# Blackhole Destination-Reuse Synchronization Fix

## Problem Analysis

Issue #46523: Nondeterministic back-to-back FP32 Welford layernorm output on Blackhole for large tensors.

### Root Cause
When `acc_to_dest=true` and `binary_reuse_dest != NONE`, the destination register is:
1. Written by the math engine (accumulator → destination)
2. Immediately read by the unpacker as a source operand (DEST_TO_SRCA or DEST_TO_SRCB)

The race occurs because there is no explicit pipeline synchronization between the math engine's write to destination and the unpacker's read from destination. With WATCHER enabled, the additional instrumentation introduces sufficient delay to hide the race. With WATCHER disabled, the unpacker may read stale or partially-written destination data.

### Affected Paths
- `DEST_TO_SRCA`: Destination reused as SrcA operand
- `DEST_TO_SRCB`: Destination reused as SrcB operand
- Column-broadcast path with `DEST_TO_SRCA` + FP32 accumulation
- `llk_unpack_A.h`: `_llk_unpack_A_mop_config_` function, acc_to_dest + reuse paths

### Solution Architecture

#### 1. Synchronization Barrier
Add a `tensix_sync()` or targeted stall before unpacker reads destination when:
- `acc_to_dest == true`
- `binary_reuse_dest != NONE`

This ensures the math engine completes the destination write before the unpacker reads.

#### 2. Targeted Fix in llk_unpack_A.h
In `_llk_unpack_A_mop_config_`, after configuring acc_to_dest operations, insert a synchronization stall:
```cpp
if constexpr (acc_to_dest && binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::STALL_MATH);
}
```

#### 3. Re-enable Skipped Tests
Remove Blackhole-specific skips for the FP32 Welford layernorm cases in:
- `tests/ttnn/nightly/unit_tests/operations/fused/test_layer_norm_ulp.py`

## Verification

- Back-to-back executions produce identical numerical output
- All LLK destination-reuse tests pass
- Existing non-reuse paths unaffected
- The original large-shape Welford layernorm cases pass on Blackhole without #46523 skips
