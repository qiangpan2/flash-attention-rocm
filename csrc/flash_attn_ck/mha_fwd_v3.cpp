/******************************************************************************
 * Copyright (c) 2025, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"
#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"
#include "mask.hpp"

std::vector<at::Tensor>
mha_fwd_v3(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x head_size
           const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
           const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
           std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x head_size
           std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
           const float p_dropout,
           const float softmax_scale,
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

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // V3 does not support dropout yet
    TORCH_CHECK(p_dropout == 0.0f, "v3 implementation does not support dropout");
    // V3 does not support alibi yet (TODO: can be added)
    TORCH_CHECK(!alibi_slopes_.has_value(), "v3 implementation does not support alibi slopes yet");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1) { is_causal = false; }

    // Determine mask type
    // ck_tile::GenericAttentionMaskEnum:
    //   0 = NO_MASK
    //   1 = MASK_FROM_TOP_LEFT (causal)
    //   2 = MASK_FROM_BOTTOM_RIGHT
    int mask_type = 0; // NO_MASK
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
        window_size_left = -1;
        mask_type = 1; // MASK_FROM_TOP_LEFT (causal)
    }
    else if (window_size_left >= 0 || window_size_right >= 0) {
        // Local/sliding window attention
        mask_type = 2; // masked (generic)
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0;
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    }
    else {
        out = torch::empty_like(q);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    at::Tensor softmax_lse;
    softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

    // Construct fmha_fwd_v3_args
    ck_tile::fmha_fwd_v3_args args;
    
    // Set data type
    args.data_type = (q_dtype == torch::kFloat16) 
        ? ck_tile::fmha_fwd_v3_args::data_type_enum::fp16
        : ck_tile::fmha_fwd_v3_args::data_type_enum::bf16;
    
    // Set dimensions
    args.batch = batch_size;
    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.nhead_q = num_heads;
    args.nhead_kv = num_heads_k;
    args.hdim_qk = head_size;
    args.hdim_v = head_size;
    
    // Set scale
    float scale = softmax_scale;
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    }

    args.softmax_scale = scale;
    
    // Set mask
    args.mask_type = mask_type;
    args.window_size_left = window_size_left;
    args.window_size_right = window_size_right;
    
    // Set Q tensor
    args.q_ptr = q.data_ptr();
    args.stride_q = q.stride(1);
    args.nhead_stride_q = q.stride(2);
    args.batch_stride_q = q.stride(0);
    
    // Set K tensor
    args.k_ptr = k.data_ptr();
    args.stride_k = k.stride(1);
    args.nhead_stride_k = k.stride(2);
    args.batch_stride_k = k.stride(0);
    
    // Set V tensor
    args.v_ptr = v.data_ptr();
    args.stride_v = v.stride(1);
    args.nhead_stride_v = v.stride(2);
    args.batch_stride_v = v.stride(0);
    
    // Set O tensor
    args.o_ptr = out.data_ptr();
    args.stride_o = out.stride(1);
    args.nhead_stride_o = out.stride(2);
    args.batch_stride_o = out.stride(0);
    
    // No cu_seqlen overrides for batch mode
    args.cu_seqlen_q_ptr = nullptr;
    args.cu_seqlen_kv_ptr = nullptr;
    
    // Launch kernel
    auto stream = at::cuda::getCurrentHIPStream().stream();
    ck_tile::stream_config stream_config{stream};
    
    auto [success, time] = ck_tile::fmha_fwd_v3(args, stream_config);
    
    TORCH_CHECK(success, "fmha_fwd_v3 kernel failed to launch");
    
    // Reshape output if we swapped
    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * ngroups, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * ngroups, head_size});
    }
    
    // Return dummy tensors for compatibility
    // Note: S_dmask would contain dropout mask, but v3 doesn't support dropout yet
    at::Tensor S_dmask = torch::empty({0}, opts.dtype(torch::kUInt8));
    at::Tensor rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    
    return {out, softmax_lse, S_dmask, rng_state};
}

