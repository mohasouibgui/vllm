# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jet-Nemotron model executor."""
# model https://huggingface.co/jet-ai/Jet-Nemotron-4B


from collections.abc import Iterable
from itertools import islice

import torch
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)

from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

# starts here
from vllm.model_executor.models.qwen2 import Qwen2MLP as JetNemotronMLP
from vllm.model_executor.models.qwen3 import Qwen3Attention as JetNemotronAttention





# Qwen-next
logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    output, final_state = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    return output.unsqueeze(0), final_state


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        if current_platform.is_cuda() and current_platform.is_device_capability(90):
            logger.info_once(
                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
            )
            self._forward_method = self.forward_cuda
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    

@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        if current_platform.is_cuda() and current_platform.is_device_capability(90):
            logger.info_once(
                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
            )
            self._forward_method = self.forward_cuda
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


# Jet 

from dataclasses import dataclass
from typing import Optional, Tuple
from vllm.transformers_utils.configs import JetNemotronConfig
import math
from torch import nn
from collections import OrderedDict
from vllm.model_executor.jet_nemotron_utils import (
    dynamic_conv_triton_cache,
    dynamic_conv_triton_autograd,
    causal_conv_step_triton,
)


@dataclass
class JetBlockConfig():
    mode: str = 'chunk'
    expand_v: int = 2.0
    num_heads: int = 6
    head_dim: int = 256
    norm_eps: float = 1e-5
    conv_size: int = 4
    dconv_generator_reduction: int = 8
    dconv_implementation: str = 'triton'


def init_linear_conv1d(weight: torch.Tensor, std: float, bias: Optional[torch.Tensor] = None) -> None:
    weight.data.normal_(mean=0.0, std=std)
    if bias is not None:
        if not getattr(bias, "_no_reinit", False):
            nn.init.zeros_(bias)


class DynamicShortConvolution(nn.Module):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: Optional[int] = None,
        generator_reduction: Optional[int] = None,
        generator_activation: str = 'silu',
        activation: Optional[str] = 'silu',
        static_conv_init: Callable = None,
        use_fast_conv1d: bool = True,
        implementation: str = "naive",
    ) -> DynamicShortConvolution:
        super().__init__()

        self.hidden_size = hidden_size
        self.generator_input_size = hidden_size if generator_input_size is None else generator_input_size
        self.generator_hidden_size = hidden_size if generator_reduction is None else (hidden_size // generator_reduction)
        self.kernel_size = kernel_size
        self.activation = None
        self.use_fast_conv1d = use_fast_conv1d
        self.implementation = implementation

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation
        
        self.static_conv_init = static_conv_init
        
        self.kernel_generator = nn.Sequential(
            OrderedDict([
                ("w1", nn.Linear(self.generator_input_size, self.generator_hidden_size, bias=False)),
                ("act", ACT2FN[generator_activation]),
                ("w2", nn.Linear(self.generator_hidden_size, self.hidden_size * self.kernel_size, bias=True)),
            ])
        )
        self._init_kernel_generator()

    def _init_kernel_generator(self):
        """
        Initialize the kernel generator.
        """
        for layer in self.kernel_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
        if self.static_conv_init is not None:
            # init for static_bias
            self.static_conv_init(self.kernel_generator.w2.bias)

    def get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        flat_kernels = self.kernel_generator(x)
        if flat_kernels.dim() == 3:
            kernels = rearrange(flat_kernels, 'b t (d w) -> b t d w', w=self.kernel_size)
        elif flat_kernels.dim() == 2:
            kernels = rearrange(flat_kernels, 'b (d w) -> b d w', w=self.kernel_size)
        else:
            raise ValueError(f"Invalid kernel shape: {flat_kernels.shape}")
        return kernels

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
        Returns:
            Tensor of shape `[B, T, D]`.
        """

        """
        x: [B, T, D]
        return: [B, T, D]
        """
        
        assert cu_seqlens is None, "cu_seqlens not supported yet."
        
        B, T, D, W = *x.shape, self.kernel_size
        N = B

        input_dtype = x.dtype

        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))

        implementation = self.implementation
        if implementation == "triton" and not self.training:
            implementation = "triton_cache"

        # during the decoding phase, we assume the batch is composed of sequences of length 1
        if cache is not None and B * T == N:
            assert T == 1
            if implementation in ["naive", "triton_training"]:
                x, cache = self._step_naive(x, cache, cu_seqlens, generator_input=generator_input)
            elif implementation in ["triton", "triton_cache", "triton_decoding"]:
                x, cache = self._step_triton(x, cache, cu_seqlens, generator_input=generator_input)
            else:
                raise ValueError(f"Unknown implementation: {implementation}")
            return x, cache

        if output_final_state:
            new_cache = rearrange(x[..., -min(W, T):, :], 'n w d -> n d w')
        else:
            new_cache = None
        
        if implementation in ["naive", "triton_decoding"]:
            x = self._forward_naive(x, generator_input=generator_input)  # [B, T, D]
        elif implementation in ["triton", "triton_training"]:
            assert cache is None, "Cache not supported in pure triton mode. Please set model.eval() or use triton_cache mode."
            x = self._forward_triton(x, generator_input=generator_input)
        elif implementation == "triton_cache":
            x = self._forward_triton_cache(x, generator_input=generator_input, cache=cache)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        if self.activation is not None:
            x = ACT2FN[self.activation](x)
        
        x = x.to(input_dtype)
        if output_final_state:
            if cache is None:
                cache = x.new_zeros(N, D, W)
            cache[:, :, -min(W, T):].copy_(new_cache)

        return x, cache

    def _forward_naive(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.kernel_size
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        x = F.pad(x.transpose(1, 2), (W - 1, 0))  # [B, D, T+W-1]
        x = x.unfold(dimension=2, size=W, step=1)  # [B, D, T, W]
        x = x.permute(0, 2, 1, 3)  # [B, T, D, W]
        x = (x * kernels).sum(dim=-1)  # [B, T, D]
        return x

    def _forward_triton(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        output_triton = dynamic_conv_triton_autograd(x, kernels)
        return output_triton

    @torch.no_grad
    def _forward_triton_cache(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        generator_input = x if generator_input is None else generator_input
        assert not self.training, "Triton implementation is only available in eval mode."
        # cache: [B, D, T(W)]
        CHUNK_SIZE = 2048
        n_chunk = (x.shape[1] + CHUNK_SIZE - 1) // CHUNK_SIZE
        output_triton = torch.zeros_like(x)
        if cache is not None:
            cache = rearrange(cache, "b d t -> b t d")  # [B, T(W), D]
        for i in range(n_chunk):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, x.shape[1])
            kernels = self.get_kernel(generator_input[:, start:end])
            out = dynamic_conv_triton_cache(x[:, start:end], kernels, cache=cache)
            output_triton[:, i*CHUNK_SIZE:end, :] = out
            cache = x[:, end-self.kernel_size:end, :]
        return output_triton

    def _step_naive(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape
        generator_input = x if generator_input is None else generator_input
        x = x.squeeze(1)
        generator_input = generator_input.squeeze(1) # Shape [B, D]
        B, D, W = *x.shape, self.kernel_size

        # we follow the fast mode that updates the cache in-place
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x # [B, D, T(W)]
        
        kernels = self.get_kernel(generator_input) # [B, D, W]
        x = torch.sum(cache * kernels, dim=-1)
        
        if self.activation is not None:
            x = ACT2FN[self.activation](x)
        
        return x.view(shape), cache
    
    def _step_triton(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Triton Implementation ---
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape # Keep original shape [B, 1, D] for return
        generator_input = x if generator_input is None else generator_input

        # 1. Generate kernels
        kernels_triton = self.get_kernel(generator_input.squeeze(1)) # [B, D, W]

        # 2. Call Triton kernel without activation
        x_out_triton = causal_conv_step_triton(
            x,
            cache,
            kernels_triton,
        )
        
        # Apply activation (if any) after kernel execution
        if self.activation is not None:
            x_out_triton = ACT2FN[self.activation](x_out_triton)

        # 3. Return reshaped output and the *same cache tensor* (it was updated in-place)
        return x_out_triton.view(shape), cache
    


class JetBlock(nn.Module):
    def __init__(
        self,
        config: Optional[JetNemotronConfig] = None,
        layer_type: str = 'jet',
        layer_idx: Optional[int] = None,
        hidden_size: Optional[int] = None,
        initializer_range: Optional[float] = None,
        jet_block_config: Optional[JetBlockConfig] = None
    ) -> JetBlock:
        super().__init__()

        if jet_block_config is None:
            assert config.efficient_attention_config is not None, "Efficient attention config must be provided in JetConfig."
            assert layer_type in config.efficient_attention_config, \
                f"{layer_type} configuration must be provided in efficient_attention_config."
            jet_block_config = JetBlockConfig(**config.efficient_attention_config[layer_type])

        hidden_size = hidden_size or config.hidden_size
        initializer_range = initializer_range or config.initializer_range

        self.mode = jet_block_config.mode

        self.hidden_size = hidden_size
        self.expand_v = jet_block_config.expand_v

        self.conv_size = jet_block_config.conv_size

        self.head_dim = jet_block_config.head_dim
        self.num_heads = jet_block_config.num_heads

        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = jet_block_config.head_dim
        self.head_v_dim = int(jet_block_config.head_dim * self.expand_v)
        self.layer_idx = layer_idx

        self.autotune_interval = 32 * 16 * 1024 # 32 batch size * 16 num head * 1024 sequence length

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.key_dim * self.expand_v}, which is invalid for nn.Linear."
            )
        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * self.expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert self.mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{jet_block_config.mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim,
            kernel_size=self.conv_size,
            generator_input_size=self.hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=lambda x: init_linear_conv1d(x, std=initializer_range),
            implementation=jet_block_config.dconv_implementation,
        )

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=float(jet_block_config.norm_eps), autotune_interval=self.autotune_interval)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[JetNemotronCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[JetNemotronCache]]:
        if attention_mask is not None:
            if len(attention_mask.shape) > 2:
                attention_mask = attention_mask.squeeze(1)
                attention_mask = torch.where(attention_mask[:, -1] > -1, 1, 0)

            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None and q_len > 1:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])

        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None

        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))

        conv_state = None
        if last_state is not None:
            conv_state = last_state['conv_state']
        v, conv_state = self.dynamic_conv1d(
            x=self.v_proj(hidden_states),
            generator_input=hidden_states,
            mask=conv_mask,
            cache=conv_state,
            output_final_state=use_cache,
        )

        if attention_mask is not None and q_len > 1:
            q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            k = index_first_axis(rearrange(k, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            v = index_first_axis(rearrange(v, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        
        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                autotune_interval=self.autotune_interval
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
        o = self.o_norm(o, g)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None and q_len > 1:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, past_key_value