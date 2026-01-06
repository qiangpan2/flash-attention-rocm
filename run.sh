#!/bin/bash
source .venv/bin/activate 

# - dtype: only FP16 / BF16 are supported
# - head_dim: <= 256 and must be a multiple of 8
# - causal: supported (mapped to mask params internally)
# - GQA/MQA: supported, requires (num_heads % num_heads_k == 0)
# - seqlen_q != seqlen_k: supported in batch-mode forward
# - dropout: NOT supported (must be 0.0)
# - alibi_slopes: NOT supported (must not be passed)

export GPU_ARCHS="gfx1100;gfx1201"
export BUILD_TARGET=rocm
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE

python setup.py bdist_wheel 2>&1 | tee build.log

export LD_LIBRARY_PATH="/workspace/repo/flash-attention-rocm/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
python -c "import flash_attn_2_cuda; print('ok', flash_attn_2_cuda.__file__)"

python test_v3_basic.py 2>&1 | tee test.log
