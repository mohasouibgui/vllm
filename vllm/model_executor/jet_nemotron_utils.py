
from vllm.triton_utils import tl, triton
import torch
from torch.autograd import Function

# Helper function to ensure tensors are contiguous for Triton
def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()

@triton.jit
def _dynamic_conv_fwd_kernel(
    X_ptr, K_ptr, Out_ptr,
    B, T, D,
    X_stride_b, X_stride_t, X_stride_d,
    K_stride_b, K_stride_t, K_stride_d, K_stride_w,
    Out_stride_b, Out_stride_t, Out_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch_time = tl.program_id(0)
    pid_d_block = tl.program_id(1)

    batch_idx = tl.cast(pid_batch_time // T, tl.int64)
    time_idx = pid_batch_time % T

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W)

    # Load Kernels
    k_ptrs = K_ptr + (batch_idx * K_stride_b + time_idx * K_stride_t +
                      offs_d[:, None] * K_stride_d + offs_w[None, :] * K_stride_w)
    k_vals = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)

    # Load Input X with implicit padding
    t_in_offs = time_idx + offs_w - W + 1
    t_in_mask = (t_in_offs >= 0) & (t_in_offs < T)
    x_ptrs = X_ptr + (batch_idx * X_stride_b + t_in_offs[None, :] * X_stride_t +
                      offs_d[:, None] * X_stride_d)
    x_load_mask = d_mask[:, None] & t_in_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

    # Compute and Accumulate
    product = k_vals * x_vals
    accumulator += tl.sum(product, axis=1)

    # Store Result
    out_ptrs = Out_ptr + (batch_idx * Out_stride_b + time_idx * Out_stride_t +
                          offs_d * Out_stride_d)
    tl.store(out_ptrs, accumulator, mask=d_mask)

# --- Backward Kernel for Input Gradient (dX) ---
@triton.jit
def _dynamic_conv_bwd_dx_kernel(
    GradOut_ptr, K_ptr, GradX_ptr, # Note: GradX is accumulated into
    B, T, D,
    GradOut_stride_b, GradOut_stride_t, GradOut_stride_d,
    K_stride_b, K_stride_t, K_stride_d, K_stride_w,
    GradX_stride_b, GradX_stride_t, GradX_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Computes gradient w.r.t. input X.
    Grid: (B * T, cdiv(D, BLOCK_SIZE_D)) - covering GradX output
    GradX[b, t_x, d] = sum_{w=0}^{W-1} GradOut[b, t, d] * K[b, t, d, w]
                       where t = t_x + W - 1 - w
    """
    pid_batch_time_x = tl.program_id(0) # Covers B * T for output GradX
    pid_d_block = tl.program_id(1)

    batch_idx = pid_batch_time_x // T
    time_idx_x = pid_batch_time_x % T # This is t_x

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    # Accumulator for GradX elements
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W) # [W]

    # Loop over W to accumulate contributions
    # Calculate the 't' index needed for GradOut and K based on t_x and w
    # t = t_x + W - 1 - w
    t_k_gradout_offs = time_idx_x + W - 1 - offs_w # Shape [W]

    # Mask for valid 't' indices [0, T)
    t_k_gradout_mask = (t_k_gradout_offs >= 0) & (t_k_gradout_offs < T) # Shape [W]

    # --- Load GradOut ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    gradout_ptrs = GradOut_ptr + (batch_idx * GradOut_stride_b +
                                  t_k_gradout_offs[None, :] * GradOut_stride_t +
                                  offs_d[:, None] * GradOut_stride_d)
    # Combined mask for loading GradOut (valid D and valid t)
    gradout_load_mask = d_mask[:, None] & t_k_gradout_mask[None, :]
    # Shape: [BLOCK_SIZE_D, W]
    gradout_vals = tl.load(gradout_ptrs, mask=gradout_load_mask, other=0.0)

    # --- Load Kernels ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    k_ptrs = K_ptr + (batch_idx * K_stride_b +
                      t_k_gradout_offs[None, :] * K_stride_t +
                      offs_d[:, None] * K_stride_d +
                      offs_w[None, :] * K_stride_w) # Index K with 't' and 'w'
    # Combined mask for loading K (valid D and valid t)
    k_load_mask = d_mask[:, None] & t_k_gradout_mask[None, :]
    # Shape: [BLOCK_SIZE_D, W]
    k_vals = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

    # --- Compute product and accumulate ---
    # Shape: [BLOCK_SIZE_D, W]
    product = gradout_vals * k_vals
    # Sum contributions over the W dimension
    accumulator += tl.sum(product, axis=1) # Shape: [BLOCK_SIZE_D]

    # --- Store accumulated gradients ---
    # Note: This kernel computes the *entire* gradient value for GradX[b, t_x, d_block].
    # If this kernel could potentially be called multiple times for the same GradX elements
    # (e.g., in complex graphs), atomic adds would be needed. Here, it seems direct store is fine.
    gradx_ptrs = GradX_ptr + (batch_idx * GradX_stride_b +
                              time_idx_x * GradX_stride_t +
                              offs_d * GradX_stride_d)
    tl.store(gradx_ptrs, accumulator, mask=d_mask)


# --- Backward Kernel for Kernel Gradient (dK) ---
@triton.jit
def _dynamic_conv_bwd_dk_kernel(
    GradOut_ptr, X_ptr, GradK_ptr, # Note: GradK is written directly
    B, T, D,
    GradOut_stride_b, GradOut_stride_t, GradOut_stride_d,
    X_stride_b, X_stride_t, X_stride_d,
    GradK_stride_b, GradK_stride_t, GradK_stride_d, GradK_stride_w,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Computes gradient w.r.t. kernels K.
    Grid: (B * T, cdiv(D, BLOCK_SIZE_D)) - covering GradK output dims B, T, D
    GradK[b, t, d, w] = GradOut[b, t, d] * X[b, t + w - W + 1, d]
    """
    pid_batch_time = tl.program_id(0) # Covers B * T for output GradK
    pid_d_block = tl.program_id(1)

    batch_idx = pid_batch_time // T
    time_idx = pid_batch_time % T # This is 't' for GradK and GradOut

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D

    offs_w = tl.arange(0, W) # [W]

    # --- Load GradOut ---
    # Pointers shape: [BLOCK_SIZE_D] (only depends on b, t, d)
    gradout_ptrs = GradOut_ptr + (batch_idx * GradOut_stride_b +
                                  time_idx * GradOut_stride_t +
                                  offs_d * GradOut_stride_d)
    # Shape: [BLOCK_SIZE_D]
    gradout_vals = tl.load(gradout_ptrs, mask=d_mask, other=0.0)

    # --- Load Input X with implicit padding ---
    # Calculate X's time index: t_x = t + w - W + 1
    t_in_offs = time_idx + offs_w - W + 1 # Shape [W]
    # Mask for valid t_x index [0, T)
    t_in_mask = (t_in_offs >= 0) & (t_in_offs < T) # Shape [W]

    # Pointers shape: [BLOCK_SIZE_D, W]
    x_ptrs = X_ptr + (batch_idx * X_stride_b +
                      t_in_offs[None, :] * X_stride_t +
                      offs_d[:, None] * X_stride_d)
    # Combined mask for loading X (valid D and valid t_x)
    x_load_mask = d_mask[:, None] & t_in_mask[None, :] # Shape [BLOCK_SIZE_D, W]
    # Shape: [BLOCK_SIZE_D, W]
    x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

    # --- Compute GradK = GradOut * X ---
    # Broadcast gradout_vals: [BLOCK_SIZE_D, 1] * [BLOCK_SIZE_D, W] -> [BLOCK_SIZE_D, W]
    gradk_vals = gradout_vals[:, None] * x_vals # Shape [BLOCK_SIZE_D, W]

    # --- Store gradients for Kernels ---
    # Pointers shape: [BLOCK_SIZE_D, W]
    gradk_ptrs = GradK_ptr + (batch_idx * GradK_stride_b +
                              time_idx * GradK_stride_t +
                              offs_d[:, None] * GradK_stride_d +
                              offs_w[None, :] * GradK_stride_w)
    # Mask only needed for D dimension (W is fully computed)
    # Store computed gradient values.
    tl.store(gradk_ptrs, gradk_vals, mask=d_mask[:, None])


# --- Autograd Function ---
class DynamicConvTritonFunc(Function):

    @staticmethod
    def forward(ctx, x, kernels):
        """
        Args:
            x: Input tensor [B, T, D]
            kernels: Kernels tensor [B, T, D, W]
        """
        x = ensure_contiguous(x)
        kernels = ensure_contiguous(kernels)

        B, T, D = x.shape
        W = kernels.shape[3]
        assert W <= 4, "Kernel W > 4 not expected for this version"

        out = torch.empty_like(x) # Output shape [B, T, D]

        grid = lambda meta: (B * T, triton.cdiv(D, meta['BLOCK_SIZE_D']))
        BLOCK_SIZE_D = 128 # Consider tuning

        _dynamic_conv_fwd_kernel[grid](
            x, kernels, out,
            B, T, D,
            x.stride(0), x.stride(1), x.stride(2),
            kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Save tensors needed for backward
        # Need x for dK, need kernels for dX
        ctx.save_for_backward(x, kernels)
        # Store W and BLOCK_SIZE_D needed for backward kernel calls
        ctx.W = W
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: Gradient w.r.t. the output tensor [B, T, D]
        Returns:
            grad_x: Gradient w.r.t. input x [B, T, D]
            grad_kernels: Gradient w.r.t. kernels [B, T, D, W]
        """
        grad_out = ensure_contiguous(grad_out)
        x, kernels = ctx.saved_tensors
        W = ctx.W
        BLOCK_SIZE_D = ctx.BLOCK_SIZE_D

        B, T, D = x.shape

        # Initialize gradients
        # grad_x needs accumulation, start with zeros.
        grad_x = torch.zeros_like(x)
        # grad_kernels is computed directly, can use empty_like if kernel handles all writes.
        # Using empty and relying on kernel writing zeros via masking/other=0.0.
        grad_kernels = torch.empty_like(kernels)

        # Define grid (can often be the same as forward or similar)
        grid = lambda meta: (B * T, triton.cdiv(D, meta['BLOCK_SIZE_D']))

        # Kernel call for grad_x
        _dynamic_conv_bwd_dx_kernel[grid](
            grad_out, kernels, grad_x,
            B, T, D,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            kernels.stride(0), kernels.stride(1), kernels.stride(2), kernels.stride(3),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Kernel call for grad_kernels
        _dynamic_conv_bwd_dk_kernel[grid](
            grad_out, x, grad_kernels,
            B, T, D,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            grad_kernels.stride(0), grad_kernels.stride(1), grad_kernels.stride(2), grad_kernels.stride(3),
            W=W,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        # Return gradients in the order inputs were received by forward
        return grad_x, grad_kernels

# --- User-facing function ---
def dynamic_conv_triton_autograd(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Fused dynamic convolution with autograd support using Triton kernels.
    Assumes W <= 4.
    Args:
        x: Input tensor of shape [B, T, D].
        kernels: Dynamic kernels of shape [B, T, D, W].
    Returns:
        Output tensor of shape [B, T, D].
    """
    return DynamicConvTritonFunc.apply(x, kernels)

# --- User-facing function ---
def dynamic_conv_triton_cache(x: torch.Tensor, kernels: torch.Tensor, cache: torch.Tensor = None) -> torch.Tensor:
    """
    Fused dynamic convolution with autograd support using Triton kernels.
    Assumes W <= 4.
    Args:
        x: Input tensor of shape [B, T, D].
        kernels: Dynamic kernels of shape [B, T, D, W].
        cache: Optional past context tensor of shape [B, T_cache, D].
               If provided, treated as concatenated before x for convolution input.
    Returns:
        Output tensor of shape [B, T, D].
    """
    return DynamicConvTritonFunc.apply(x, kernels, cache)

@triton.jit
def _causal_conv_step_kernel(
    # --- Input/Output Pointers ---
    X_ptr,         # Pointer to current input x [B, D] (after squeeze)
    Cache_ptr,     # Pointer to cache [B, D, W], updated IN-PLACE
    Kernels_ptr,   # Pointer to generated kernels [B, D, W]
    Out_ptr,       # Pointer to output tensor [B, D]
    # --- Tensor Dimensions ---
    B, D,          # Batch size, Feature dimension
    # --- Tensor Strides ---
    X_stride_b, X_stride_d,
    Cache_stride_b, Cache_stride_d, Cache_stride_w,
    Kernels_stride_b, Kernels_stride_d, Kernels_stride_w,
    Out_stride_b, Out_stride_d,
    # --- Kernel Meta-Parameters ---
    W: tl.constexpr,               # Kernel width (Cache size), passed as compile-time constant (1 < W <= 4)
    BLOCK_SIZE_D: tl.constexpr,    # Block size for D dimension (tuning parameter)
    # Removed ACTIVATION: tl.constexpr
):
    """
    Triton kernel for a single step (T=1) of causal dynamic convolution.
    Updates the cache in-place and computes the output (without activation).
    Optimized for small W (1 < W <= 4) by manually unrolling the W dimension.
    Does NOT handle separate static bias.
    Grid: (B, cdiv(D, BLOCK_SIZE_D))
    Updates Cache[b, d, :] and computes Out[b, d].
    """
    # 1. --- Get Program IDs and Calculate Indices ---
    pid_b = tl.program_id(0)       # Program ID for batch dimension
    pid_d_block = tl.program_id(1) # Program ID for dimension block

    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D # Shape: [BLOCK_SIZE_D]

    # 2. --- Load Current Input X ---
    x_ptrs = X_ptr + pid_b * X_stride_b + offs_d * X_stride_d
    x_curr = tl.load(x_ptrs, mask=d_mask, other=0.0) # Shape: [BLOCK_SIZE_D]

    # --- Initialize Accumulator ---
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=x_curr.dtype) # Use input dtype

    # --- Manually Unroll Operations for W ---
    # We will load kernel values and cache values step-by-step
    # and perform the calculation and cache update.

    # --- Step w = 0 ---
    # Compute: cache_val_1 * k_val_0 (part 1)
    # Cache Update: store cache_val_1 at index 0
    if tl.constexpr(W > 1):
        # Load k_val_0
        k_ptr_0 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 0 * Kernels_stride_w
        k_val_0 = tl.load(k_ptr_0, mask=d_mask, other=0.0)

        # Load cache_val_1 (needed for computation and storing at index 0)
        cache_ptr_1 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 1 * Cache_stride_w
        cache_val_1 = tl.load(cache_ptr_1, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_1 * k_val_0

        # Cache Update: Store cache_val_1 -> cache_ptr_0
        cache_ptr_0 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 0 * Cache_stride_w
        tl.store(cache_ptr_0, cache_val_1, mask=d_mask)

    # --- Step w = 1 ---
    # Compute: cache_val_2 * k_val_1 (part 1)
    # Cache Update: store cache_val_2 at index 1
    if tl.constexpr(W > 2):
        # Load k_val_1
        k_ptr_1 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 1 * Kernels_stride_w
        k_val_1 = tl.load(k_ptr_1, mask=d_mask, other=0.0)

        # Load cache_val_2
        cache_ptr_2 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 2 * Cache_stride_w
        cache_val_2 = tl.load(cache_ptr_2, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_2 * k_val_1

        # Cache Update: Store cache_val_2 -> cache_ptr_1
        cache_ptr_1 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 1 * Cache_stride_w
        tl.store(cache_ptr_1, cache_val_2, mask=d_mask)

    # --- Step w = 2 ---
    # Compute: cache_val_3 * k_val_2 (part 1)
    # Cache Update: store cache_val_3 at index 2
    if tl.constexpr(W > 3):
        # Load k_val_2
        k_ptr_2 = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + 2 * Kernels_stride_w
        k_val_2 = tl.load(k_ptr_2, mask=d_mask, other=0.0)

        # Load cache_val_3
        cache_ptr_3 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 3 * Cache_stride_w
        cache_val_3 = tl.load(cache_ptr_3, mask=d_mask, other=0.0)

        # Accumulate Part 1
        accumulator += cache_val_3 * k_val_2

        # Cache Update: Store cache_val_3 -> cache_ptr_2
        cache_ptr_2 = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + 2 * Cache_stride_w
        tl.store(cache_ptr_2, cache_val_3, mask=d_mask)

    # --- Final Step (Part 2 and Final Cache Update) ---
    # Compute: x_curr * k_val_{W-1} (part 2)
    # Cache Update: store x_curr at index W-1

    # Load k_val_{W-1}
    k_ptr_last = Kernels_ptr + pid_b * Kernels_stride_b + offs_d * Kernels_stride_d + (W - 1) * Kernels_stride_w
    k_val_last = tl.load(k_ptr_last, mask=d_mask, other=0.0)

    # Accumulate Part 2
    accumulator += x_curr * k_val_last

    # Final Cache Update: Store x_curr -> cache_ptr_{W-1}
    cache_ptr_last = Cache_ptr + pid_b * Cache_stride_b + offs_d * Cache_stride_d + (W - 1) * Cache_stride_w
    tl.store(cache_ptr_last, x_curr, mask=d_mask)

    # Removed activation application: accumulator = _apply_activation(accumulator, ACTIVATION)

    # 6. --- Store Output ---
    out_ptrs = Out_ptr + pid_b * Out_stride_b + offs_d * Out_stride_d
    tl.store(out_ptrs, accumulator, mask=d_mask) # Store result without activation

    # Cache update is now fully handled within the unrolled steps.


# --- Python Wrapper Function ---
def causal_conv_step_triton(
    x: torch.Tensor,           # Input tensor [B, 1, D]
    cache: torch.Tensor,       # Cache tensor [B, D, W], modified in-place
    kernels: torch.Tensor,     # Kernels tensor [B, D, W]
    # Removed activation parameter
) -> torch.Tensor:             # Returns output tensor [B, D] (before activation)
    """
    Performs one step of causal dynamic convolution using Triton.
    Updates the cache in-place. Does NOT fuse activation. Assumes 1 < W <= 4.
    Uses manually unrolled kernel for W dimension.
    Args:
        x: Current input token tensor of shape [B, 1, D].
        cache: Cache tensor of shape [B, D, W]. Will be updated in-place.
        kernels: Dynamically generated kernels tensor of shape [B, D, W].
    Returns:
        Output tensor of shape [B, D] for the current step (before activation).
    """
    # --- Input Validation and Preparation ---
    assert x.dim() == 3 and x.shape[1] == 1, "Input x must have shape [B, 1, D]"
    assert cache.dim() == 3, "Cache must have shape [B, D, W]"
    assert kernels.dim() == 3, "Kernels must have shape [B, D, W]"
    B, _, D = x.shape
    W = cache.shape[2]
    # Updated assertion: W must be > 1 and <= 4
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"
    assert cache.shape[0] == B and cache.shape[1] == D, f"Cache shape mismatch: {cache.shape}"
    assert kernels.shape == cache.shape, f"Kernels shape mismatch: {kernels.shape}"
    assert x.is_cuda and cache.is_cuda and kernels.is_cuda, "Inputs must be CUDA tensors"
    # Allow different input dtypes, but ensure they are compatible or handled
    # assert x.dtype == cache.dtype == kernels.dtype, "Input dtypes must match"

    # Squeeze the time dimension from input x
    x_squeezed = x.squeeze(1) # Shape [B, D]

    # Ensure tensors are contiguous for correct stride calculations in Triton
    x_squeezed = ensure_contiguous(x_squeezed)
    # Cache MUST be contiguous for in-place updates and loads/stores to work reliably
    cache = ensure_contiguous(cache)
    kernels = ensure_contiguous(kernels)

    # Create output tensor with the same dtype as input x
    out = torch.empty_like(x_squeezed) # Shape [B, D]

    # --- Triton Kernel Launch ---
    grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_SIZE_D']))
    BLOCK_SIZE_D = 64 # Example, tune this value

    # Launch the kernel
    _causal_conv_step_kernel[grid](
        x_squeezed, cache, kernels, out,   # Tensor pointers
        B, D,                              # Dimensions
        x_squeezed.stride(0), x_squeezed.stride(1), # x strides
        cache.stride(0), cache.stride(1), cache.stride(2), # cache strides
        kernels.stride(0), kernels.stride(1), kernels.stride(2), # kernels strides
        out.stride(0), out.stride(1),      # out strides
        # --- Meta-parameters ---
        W=W,                               # Pass W as constexpr
        BLOCK_SIZE_D=BLOCK_SIZE_D,         # Pass BLOCK_SIZE_D as constexpr
        # Removed ACTIVATION=activation
    )

    return out # Return the computed output [B, D] (before activation)