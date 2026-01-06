#!/usr/bin/env python3


import torch
import torch.nn.functional as F

print("=" * 80)
print("Testing ck_tile v3 FlashAttention Implementation")
print("=" * 80)

# Check if CUDA/ROCm is available
if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available")
    exit(1)

device = torch.device('cuda')
print(f"Using device: {torch.cuda.get_device_name(0)}")
print(f"Device properties: {torch.cuda.get_device_properties(0).gcnArchName}")
print()

# Test configuration
batch = 2
seqlen = 1024
nheads = 8
headdims = [64, 128]
dtype = torch.float16

print(f"Test configuration:")
print(f"  Batch size: {batch}")
print(f"  Sequence length: {seqlen}")
print(f"  Number of heads: {nheads}")
print(f"  Head dimensions: {headdims}")
print(f"  Data type: {dtype}")
print()

# Make tests reproducible
torch.manual_seed(0)

# Import flash_attn
try:
    from flash_attn import flash_attn_func
    print("✓ flash_attn module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import flash_attn: {e}")
    exit(1)

def attention_ref(q, k, v, causal: bool):
    """
    Reference attention in fp32:
      out = softmax(q @ k^T / sqrt(d) + mask) @ v
    Shapes:
      q, k, v: [B, S, H, D]
      out:     [B, S, H, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    b, s_q, h, d = q.shape
    b2, s_k, h2, d2 = k.shape
    assert b == b2 and h == h2 and d == d2
    assert v.shape == k.shape

    qf = q.float()
    kf = k.float()
    vf = v.float()
    scale = 1.0 / (d ** 0.5)

    # (B, S, H, D) -> (B*H, S, D)
    q2 = qf.permute(0, 2, 1, 3).reshape(b * h, s_q, d)
    k2 = kf.permute(0, 2, 1, 3).reshape(b * h, s_k, d)
    v2 = vf.permute(0, 2, 1, 3).reshape(b * h, s_k, d)

    scores = torch.bmm(q2, k2.transpose(1, 2)) * scale  # (B*H, S_q, S_k)
    if causal:
        # Only valid for s_q == s_k in this reference (matches typical causal self-attn use).
        assert s_q == s_k, "reference causal mask assumes s_q == s_k"
        mask = torch.triu(torch.ones(s_q, s_k, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.bmm(p, v2)  # (B*H, S_q, D)
    out = out.reshape(b, h, s_q, d).permute(0, 2, 1, 3).contiguous()
    return out

def attention_ref_sdpa(q, k, v, causal: bool):
    """
    Reference via PyTorch SDPA (math kernel preferred).
    Shapes:
      q, k, v: [B, S, H, D]
      out:     [B, S, H, D]
    """
    b, s, h, d = q.shape
    assert k.shape == (b, s, h, d)
    assert v.shape == (b, s, h, d)

    q2 = q.float().permute(0, 2, 1, 3)  # (B, H, S, D)
    k2 = k.float().permute(0, 2, 1, 3)
    v2 = v.float().permute(0, 2, 1, 3)

    # Try to force SDPA to use the math implementation (avoid calling flash kernels).
    try:
        # PyTorch >= 2.0 provides this context manager on CUDA; on ROCm it may still exist.
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            out2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None, dropout_p=0.0, is_causal=causal)
    except Exception:
        out2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None, dropout_p=0.0, is_causal=causal)

    out = out2.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D)
    return out

def assert_close_fp32(out, ref, *, name: str, atol: float, rtol: float):
    out_f = out.float()
    ref_f = ref.float()
    max_abs = (out_f - ref_f).abs().max().item()
    denom = ref_f.abs().max().item()
    max_rel = max_abs / denom if denom != 0 else float("inf")
    print(f"  {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e} (atol={atol}, rtol={rtol})")
    torch.testing.assert_close(out_f, ref_f, atol=atol, rtol=rtol)

def tensor_stats(x: torch.Tensor, name: str):
    xf = x.float()
    nan = torch.isnan(xf).sum().item()
    inf = torch.isinf(xf).sum().item()
    absmax = xf.abs().max().item() if xf.numel() else 0.0
    minv = xf.min().item() if xf.numel() else 0.0
    maxv = xf.max().item() if xf.numel() else 0.0
    print(f"  {name}: nan={nan}, inf={inf}, absmax={absmax:.6e}, range=[{minv:.6e}, {maxv:.6e}]")

def test_basic_forward():
    """Test basic forward pass"""
    print("\n" + "=" * 80)
    print("Test 1: Basic Forward Pass (no mask)")
    print("=" * 80)

    ok = True
    for headdim in headdims:
        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

        try:
            out = flash_attn_func(q, k, v, causal=False)
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            print(f"✓ Basic forward pass successful (headdim={headdim})")
            print(f"  Output shape: {out.shape}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        except Exception as e:
            ok = False
            print(f"✗ Basic forward pass failed (headdim={headdim}): {e}")
            import traceback
            traceback.print_exc()
    return ok

def test_reference_correctness():
    """
    Compare flash_attn_func against a fp32 reference on a small problem.
    This is an actual correctness check (not just NaN/Inf).
    """
    print("\n" + "=" * 80)
    print("Test 0: Reference Correctness (small fp32 reference)")
    print("=" * 80)

    # Reference is O(S^2), keep it small.
    seqlen_ref = min(seqlen, 128)
    atol = 5e-2 if dtype == torch.float16 else 8e-2
    rtol = 5e-2 if dtype == torch.float16 else 8e-2

    ok = True
    for headdim in headdims:
        q = torch.randn(batch, seqlen_ref, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen_ref, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen_ref, nheads, headdim, device=device, dtype=dtype)

        for causal in [False, True]:
            try:
                out = flash_attn_func(q, k, v, causal=causal)
                # Prefer SDPA as reference; fall back to explicit softmax if unavailable.
                try:
                    ref = attention_ref_sdpa(q, k, v, causal=causal)
                except Exception:
                    ref = attention_ref(q, k, v, causal=causal)

                tensor_stats(out, "out(fp32)")
                tensor_stats(ref, "ref(fp32)")
                assert_close_fp32(out, ref, name=f"headdim={headdim}, causal={causal}", atol=atol, rtol=rtol)
                print(f"✓ Reference match (headdim={headdim}, causal={causal})")
            except Exception as e:
                ok = False
                print(f"✗ Reference mismatch (headdim={headdim}, causal={causal}): {e}")
                import traceback
                traceback.print_exc()
    return ok

def test_causal_mask():
    """Test causal mask"""
    print("\n" + "=" * 80)
    print("Test 2: Causal Mask")
    print("=" * 80)
    
    ok = True
    for headdim in headdims:
        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

        try:
            out = flash_attn_func(q, k, v, causal=True)
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            print(f"✓ Causal mask forward pass successful (headdim={headdim})")
            print(f"  Output shape: {out.shape}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        except Exception as e:
            ok = False
            print(f"✗ Causal mask forward pass failed (headdim={headdim}): {e}")
            import traceback
            traceback.print_exc()
    return ok

def test_gqa():
    """Test Grouped Query Attention (GQA)"""
    print("\n" + "=" * 80)
    print("Test 3: Grouped Query Attention (GQA)")
    print("=" * 80)
    
    ok = True
    nheads_k = 4  # Half the number of query heads
    for headdim in headdims:
        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, nheads_k, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, nheads_k, headdim, device=device, dtype=dtype)

        try:
            out = flash_attn_func(q, k, v, causal=False)
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            print(f"✓ GQA forward pass successful (headdim={headdim})")
            print(f"  Q heads: {nheads}, KV heads: {nheads_k}")
            print(f"  Output shape: {out.shape}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        except Exception as e:
            ok = False
            print(f"✗ GQA forward pass failed (headdim={headdim}): {e}")
            import traceback
            traceback.print_exc()
    return ok

def test_different_seqlens():
    """Test different Q and K sequence lengths"""
    print("\n" + "=" * 80)
    print("Test 4: Different Q and K Sequence Lengths")
    print("=" * 80)
    
    ok = True
    seqlen_q = 512
    seqlen_k = 1024
    for headdim in headdims:
        q = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen_k, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen_k, nheads, headdim, device=device, dtype=dtype)

        try:
            out = flash_attn_func(q, k, v, causal=False)
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            print(f"✓ Different sequence lengths forward pass successful (headdim={headdim})")
            print(f"  Q seqlen: {seqlen_q}, KV seqlen: {seqlen_k}")
            print(f"  Output shape: {out.shape}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        except Exception as e:
            ok = False
            print(f"✗ Different sequence lengths forward pass failed (headdim={headdim}): {e}")
            import traceback
            traceback.print_exc()
    return ok

def test_bfloat16():
    """Test with bfloat16 dtype"""
    print("\n" + "=" * 80)
    print("Test 5: BFloat16 Data Type")
    print("=" * 80)

    if not torch.cuda.is_bf16_supported():
        print("! BF16 not supported on this device/runtime, skipping")
        return True
    
    ok = True
    for headdim in headdims:
        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16)

        try:
            out = flash_attn_func(q, k, v, causal=False)
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert out.dtype == torch.bfloat16, f"Output dtype mismatch: {out.dtype}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            print(f"✓ BFloat16 forward pass successful (headdim={headdim})")
            print(f"  Output shape: {out.shape}")
            print(f"  Output dtype: {out.dtype}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        except Exception as e:
            ok = False
            print(f"✗ BFloat16 forward pass failed (headdim={headdim}): {e}")
            import traceback
            traceback.print_exc()
    return ok

# Run all tests
if __name__ == "__main__":
    results = []
    
    results.append(("Reference Correctness", test_reference_correctness()))
    results.append(("Basic Forward", test_basic_forward()))
    results.append(("Causal Mask", test_causal_mask()))
    results.append(("GQA", test_gqa()))
    results.append(("Different Seqlens", test_different_seqlens()))
    results.append(("BFloat16", test_bfloat16()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        exit(1)

