/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Stub implementation for backward pass - not implemented in v3
 ******************************************************************************/

#include "flash_common.hpp"

// Stub for backward pass - provides duck-typed interface but not implemented
std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &softmax_lse,
        std::optional<at::Tensor> &dq_,
        std::optional<at::Tensor> &dk_,
        std::optional<at::Tensor> &dv_,
        std::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state)
{
    TORCH_CHECK(false, "Backward pass is not implemented in ck_tile v3. "
                       "This build only supports forward inference.");
    return {};
}

// Stub for varlen backward pass
std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,
               const at::Tensor &q,
               const at::Tensor &k,
               const at::Tensor &v,
               const at::Tensor &out,
               const at::Tensor &softmax_lse,
               std::optional<at::Tensor> &dq_,
               std::optional<at::Tensor> &dk_,
               std::optional<at::Tensor> &dv_,
               const at::Tensor &cu_seqlens_q,
               const at::Tensor &cu_seqlens_k,
               std::optional<at::Tensor> &alibi_slopes_,
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state)
{
    TORCH_CHECK(false, "Backward pass is not implemented in ck_tile v3. "
                       "This build only supports forward inference.");
    return {};
}

