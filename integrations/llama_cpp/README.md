# llama.cpp Apple M4 Integration

This integration inserts an opt-in Q4_K x Q8_K kernel into a local llama.cpp
source tree. It is disabled unless `GGML_M4_Q4K_CUSTOM=1` is present.

The installer changes two boundaries:

- the ARM Q4_K dot function dispatches to `kevin_m4_q4k.h` for `nrc=1`;
- Q4_K tensor repacking is disabled only for the opt-in process so the custom
  kernel receives original GGUF `block_q4_K` bytes.

Install and remove manually:

```bash
/usr/bin/python3 integrations/llama_cpp/install_kevin_m4_q4k.py \
  --llama-root build/llama.cpp

/usr/bin/python3 integrations/llama_cpp/install_kevin_m4_q4k.py \
  --llama-root build/llama.cpp \
  --uninstall
```

The build helper always removes the source patch when it exits:

```bash
LLAMA_ROOT=build/llama.cpp scripts/build_llama_cpp_m4_q4k.sh
```

Use `GGML_M4_Q4K_TRACE=1` with the custom flag to emit one first-hit message.
The current real-model result is parity, not a speedup. The repacked llama.cpp
path remains the default and the production recommendation.
