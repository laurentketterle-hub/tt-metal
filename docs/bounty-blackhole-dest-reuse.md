# Blackhole Destination-Reuse Sync Fix

## 000 Bounty: tt-metal#52252

Fix nondeterministic FP32 Welford layernorm on Blackhole.
Timing-sensitive LLK sync race between dest-to-source moves and dummy unpack ops.

### Strategy
- Add sync barrier after dest-to-source moves
- Correct DEST_TO_SRCA and DEST_TO_SRCB paths
- Fix FP32 accumulation in column-broadcast path
- Add deterministic multi-tile LLK tests
