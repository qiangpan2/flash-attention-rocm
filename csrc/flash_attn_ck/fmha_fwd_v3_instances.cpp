// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel instantiations for fmha_fwd_v3


#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_wmma_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "fmha_fwd_v3_head_dim_policies.hpp"
#include "fmha_fwd_v3.hpp"

namespace ck_tile {


using kernel_traits_fp16_nomask_legacy =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, false>;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_nomask_legacy);
using kernel_traits_fp16_mask_legacy =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::fp16, false, true>;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_mask_legacy);

using kernel_traits_bf16_nomask_legacy =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, false>;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_nomask_legacy);
using kernel_traits_bf16_mask_legacy =
    fmha_fwd_v3_kernel_traits<fmha_fwd_v3_args::data_type_enum::bf16, false, true>;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_mask_legacy);

using kernel_traits_fp16_64_nomask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::fp16, false, 64>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_64_nomask);
using kernel_traits_fp16_64_mask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::fp16, true, 64>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_64_mask);

using kernel_traits_bf16_64_nomask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::bf16, false, 64>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_64_nomask);
using kernel_traits_bf16_64_mask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::bf16, true, 64>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_64_mask);

// head_dim = 128 (default v3 path w/ K/V vector-load policy)
using kernel_traits_fp16_nomask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::fp16, false, 128>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_nomask);

using kernel_traits_fp16_mask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::fp16, true, 128>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_fp16_mask);

using kernel_traits_bf16_nomask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::bf16, false, 128>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_nomask);

using kernel_traits_bf16_mask =
    fmha_fwd_v3_kernel_traits_selector<fmha_fwd_v3_args::data_type_enum::bf16, true, 128>::type;
INST_FMHA_FWD_V3_DISPATCH(kernel_traits_bf16_mask);

} // namespace ck_tile

