# MiniMax-M3 Step 7 — local digest replay

## Scope

This is a baseline replay of the existing test-only M3 layer-output digest
witness. It verifies that the current checkout still produces the established
57 sparse-layer sequence before any future M3 shadow hook is considered. It is
not shadow-on/off equivalence evidence.

## Command

Run from `c/` in the checkout:

```text
SNAP=/c/Users/jaapj/.lmstudio/models/minimax_m3_i4 \
PROMPT=hi NGEN=1 TEMP=0 PIPE=0 RAM_GB=18 \
COLI_RAM_OVERCOMMIT=1 AUTOPIN=0 COLI_M3_DAG_DIGEST=1 \
./colibri.exe 2 4 8
```

## Result

- Process exit code: `0`
- Model: MiniMax-M3 grouped-int4, CPU, `S=1`
- Prompt: `hi`
- Prefill: `1` token
- Digest records: `57`, layers `3..59`
- First digest: layer `3`, `0b5dca1f6fa0cd96`
- Final digest: layer `59`, `c7cd7ef0e12065bb`
- Observed output token: `{aligned`

The replay loaded successfully and reported `M3_DAG active`. It does not
exercise a Step-7 planner snapshot, a shadow-on/off comparison, or a real
storage-topology mapping; those remain unimplemented evidence.
