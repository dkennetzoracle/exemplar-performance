# TE attention backend selection under NVFP4BlockScaling

Captured with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 on 4 x GB300 (sm_103),
nvcr.io/nvidia/nemo:26.06.01, llama31_8b nvfp4 preset, fp8 DPA left at
the preset default of true. Training cannot start.

## TE debug output
```
DEBUG:DotProductAttention:Available backends = {FlashAttention=False, FusedAttention=False, UnfusedDotProductAttention=False}
DEBUG:DotProductAttention:Disabling FlashAttention 2 for FP8 attention
DEBUG:DotProductAttention:Disabling FusedAttention for NVFP4BlockScaling
DEBUG:DotProductAttention:Disabling UnfusedDotProductAttention for FP8 attention
DEBUG:DotProductAttention:Selected backend = NoBackend.
```

## The deciding line in TE source

transformer_engine/pytorch/attention/dot_product_attention/utils.py:607
```python
if use_fused_attention and (fp8_recipe.float8_block_scaling() or fp8_recipe.nvfp4()):
    logger.debug("Disabling FusedAttention for %s", fp8_recipe.__class__.__name__)
    use_fused_attention = False
```

No compute-capability check. The MXFP8 branch immediately above it is
gated ("Disabling FusedAttention for MXFP8 on arch < sm100"), and there
is an sm120 check below, so TE does arch-gate elsewhere -- this disable
is unconditional on the recipe. FlashAttention 2 and
UnfusedDotProductAttention independently refuse FP8 attention, leaving
no backend on ANY architecture.
