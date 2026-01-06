/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"
#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

std::vector<at::Tensor>
mha_varlen_fwd_v3(at::Tensor &q,                               // total_q x num_heads x head_size
                  const at::Tensor &k,                         // total_k x num_heads_k x head_size
                  const at::Tensor &v,                         // total_k x num_heads_k x head_size
                  std::optional<at::Tensor> &out_,             // total_q x num_heads x head_size
                  const at::Tensor &cu_seqlens_q,              // b+1
                  const at::Tensor &cu_seqlens_k,              // b+1
                  std::optional<at::Tensor> &seqused_k,        // b. If given, only this many elements of each batch element's keys are used.
                  std::optional<const at::Tensor> &leftpad_k_, // batch_size
                  std::optional<at::Tensor> &block_table_,     // batch_size x max_num_blocks_per_seq
                  std::optional<at::Tensor> &alibi_slopes_,    // num_heads or b x num_heads
                  int max_seqlen_q,
                  const int max_seqlen_k,
                  const float p_dropout,
                  const float softmax_scale,
                  const bool zero_tensors,
                  bool is_causal,
                  int window_size_left,
                  int window_size_right,
                  const float /*softcap*/,
                  const bool return_softmax,
                  std::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    
    TORCH_CHECK(head_size <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // V3 does not support these features yet
    TORCH_CHECK(p_dropout == 0.0f, "v3 implementation does not support dropout");
    TORCH_CHECK(!alibi_slopes_.has_value(), "v3 implementation does not support alibi slopes yet");
    TORCH_CHECK(!block_table_.has_value(), "v3 varlen does not support block_table (paged attention) yet");
    TORCH_CHECK(!leftpad_k_.has_value(), "v3 varlen does not support leftpad_k yet");
    TORCH_CHECK(!seqused_k.has_value(), "v3 varlen does not support seqused_k yet");

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);

    int batch_size = cu_seqlens_q.numel() - 1;
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // Determine mask type
    int mask_type = 0; // no mask
    if (is_causal) {
        window_size_right = 0;
        window_size_left = -1;
        mask_type = 2; // causal mask
    }
    else if (window_size_left >= 0 || window_size_right >= 0) {
        mask_type = 2; // masked (generic)
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size);
    }
    else {
        out = torch::empty_like(q);
    }

    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    at::Tensor softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat32));

    // For varlen v3, we need to process each batch element
    // Note: This is a simplified implementation. A proper varlen implementation would
    // use variable-length kernel traits. For now, we'll use the regular batch mode
    // with proper cu_seqlen pointers.
    
    TORCH_CHECK(false, "mha_varlen_fwd_v3: Variable-length v3 implementation is not yet complete. "
                       "Please use batch mode for now.");
    
    // Return dummy values (never reached, but need valid initialization to avoid warnings)
    at::Tensor S_dmask = torch::empty({0}, opts.dtype(torch::kUInt8));
    at::Tensor rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    return {out, softmax_lse, S_dmask, rng_state};
}

