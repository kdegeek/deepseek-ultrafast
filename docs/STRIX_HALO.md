# Strix Halo migration

Target verified on November 14, 2026: GMKtec EVO-X2, Ryzen AI Max+ 395,
Radeon 8060S, Windows 11 Pro. Vulkan driver reports `26.10.1 (LLPC)`.
The existing AMD PyTorch installation reports `2.11.0+rocm7.14.0`.

The fork currently has two paths:

- `crates/strix`: native Rust/Vulkan compute using wgpu. Packed FP4, FP8, BF16,
  and FP32 matrix-vector kernels, RMSNorm, SiLU, device reporting, and strict
  checkpoint inspection. This is not yet a token-generation engine.
- `ports/strix-reference`: the official model architecture with pure PyTorch
  replacements for TileLang kernels and a bounded lazy checkpoint loader.
  It supplies a correctness baseline for the native migration. A short target-only
  text completion has passed with the full original released checkpoint.

The original CUDA implementation remains available. Neither the NVIDIA throughput
figures in the upstream README nor its 1M context claim have been validated here.

## Native build and validation

The pinned Rust toolchain and installed AMD Vulkan driver are sufficient for this
crate. CUDA, HIP SDK, WSL, Redis, and RDMA are not required for these commands.

```powershell
cargo build -p deepseek-ultrafast-strix --bin strix
cargo run -p deepseek-ultrafast-strix --bin strix -- device
cargo run -p deepseek-ultrafast-strix --bin strix -- verify
cargo test -p deepseek-ultrafast-strix
cargo test -p deepseek-ultrafast-strix --lib amd_hardware_acceptance -- --ignored
cargo clippy -p deepseek-ultrafast-strix --all-targets -- -D warnings
```

Adapter selection requires actual AMD Vulkan hardware and never silently falls
back to a CPU adapter. Device buffer limits are not a measurement of free GPU
memory. Uploaded matrices stay resident until dropped; diagnostic operations
read their output back to the host. This is a correctness baseline, not a tuned
resident inference graph.

```powershell
cargo run -p deepseek-ultrafast-strix --bin strix -- inspect MODEL_DIRECTORY 64
cargo run -p deepseek-ultrafast-strix --bin strix -- check-matrix MODEL_DIRECTORY layers.6.ffn.experts.0.w1.weight
```

The optional `inspect` budget is in GiB and counts stored weights only, excluding
KV cache, activations, staging and OS memory. Single-matrix diagnostic extraction
is capped at 256 MiB. Original I8-packed FP4 experts are accepted only when model
configuration, tensor name, dimensions, and typed scales agree. Arbitrary integer
weights are not guessed to be FP4.

## Original checkpoint and AMD reference

The original checkpoint is pinned to
`deepseek-ai/DeepSeek-V4.1-Flash@314113f8a7dadb388215396ba5fbee2c610a04d6`.
The selected files total about 510 GB. The downloader resumes into a separate
directory, records source hashes, checks space, and validates file lengths and
LFS SHA256 hashes.
It does not replace existing models. No Flash-1017 weights are substituted.

Use a Python environment with AMD HIP PyTorch, safetensors 0.8 or newer, transformers,
huggingface_hub, numpy, sympy, Pillow, and tokenizers. PyTorch's `cuda` device
name also applies to its AMD HIP build. On the acceptance machine, the existing
interpreter is used without changing installed packages:

```powershell
$python = 'C:\Users\ncrawford\.unsloth\studio\unsloth_studio\Scripts\python.exe'
& $python tools/download-strix-checkpoint.py target/models/DeepSeek-V4.1-Flash
& $python -m unittest discover -s ports/strix-reference/tests -v
& $python ports/strix-reference/run.py smoke
& $python -u ports/strix-reference/run.py generate --checkpoint target/models/DeepSeek-V4.1-Flash --cache-gib 8 --context 256 --tokens 8 --prompt 'The capital of France is' --output target/strix-full-generation.json
```

The loader constructs parameters on meta, initializes and moves runtime buffers,
then loads parameters for active modules through an LRU with a byte budget.
Dense parameters use safetensors' `pread` backend, reading one tensor at a time.
The original many-shard PyTorch mmap path crashed with a native access violation
while loading layer 30 on this Windows runtime; per-tensor reads avoid that path.
Engram tables retain CPU mappings and copy only requested rows. They must not use
the safetensors 0.8 CPU `pread` slicing path, which reads a whole tensor before
slicing it. OS page cache, CPU read buffers, GPU allocator reservations and
temporary dequantization buffers are additional to the weight-cache budget.
Files must not change while mapped. The reference is single-device and
single-threaded; its model globals are not safe for concurrent requests.

Generation uses greedy target decoding and raw completion prompts, with bounded
token-by-token prefill. DSpark, image input, API serving, concurrent users and
long-context acceptance are not implemented in this reference entry point.
The result JSON records prompt and generated token IDs, finite-logit checks and
elapsed time for every forward, AMD/PyTorch identity, peak device allocation,
weight-cache statistics, and hashes of the local provenance and tokenizer files.
Initialization time is reported separately from generation including prefill.

The release has no Jinja chat template. For a future chat entry point, use the
[pinned official encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding)
or its maintained `deepseek-recipe` implementation; a raw completion is not a
chat-template acceptance test.

## Evidence and acceptance boundaries

- Rust unit tests and Clippy pass. The explicitly requested AMD hardware test
  passes all 30 initial numerical checks.
- A real original FP4 expert matrix, `layers.6.ffn.experts.0.w1.weight`, was fetched
  using bounded HTTP ranges from the pinned checkpoint. Its 2,304 outputs passed
  against an independent CPU reference on the Radeon 8060S. No model weights are
  checked into Git.
- All 33 Python tests pass without skips on the acceptance machine, including
  CPU and actual AMD kernels, the real expert matrix, sparse attention with
  learned sinks, indexed multi-shard eviction, and original Engram rows at large
  file offsets under a default GPU device context.
- A reduced synthetic architecture using quantized projections, FP4 experts,
  compressed/shared KV, sparse attention and hyper-connections completes prefill
  and decode on CPU and AMD with matching greedy tokens. It is not evidence of
  full-model answer quality. The lazy-loader fixture also compares single-token
  and batched prefill with Engram and nonzero hyper-connections on both devices.
- All 48 original weight shards (510,296,708,312 file bytes) passed LFS SHA256
  validation. The native inspector read 96,085 checkpoint tensors; the reference
  model's 93,338 target parameter shapes and 99,092 compressed Engram IDs match.
- The full 40-layer model completed `The capital of France is` with ` Paris.`
  and then EOS. The eight-token cap produced two text tokens naturally. All
  seven forwards had finite logits. Generation including prefill took 176.10 s,
  with 7.71 s of initialization reported separately. Peak cached weights were
  8,589,922,760 bytes, below the 8 GiB budget (8,589,934,592 bytes).
  PyTorch reported 8,749,762,048 peak allocated device bytes and 14,237,564,928
  peak reserved bytes; the weight-cache budget is not a total process-memory cap.
- This is one short raw-completion acceptance, not broad model-quality or
  performance acceptance. Chat generation, DSpark, vision, long context, native
  decoding and API serving remain unverified. No full-model tokens/s claim is made.

The [full-model evidence](validation/strix-halo-full-model-2026-11-14.json) records
the exact output, token IDs, per-forward logits checks, timings, AMD identity,
cache statistics, storage fix and remaining boundaries. The
[checkpoint manifest](validation/strix-original-checkpoint.json) records every
original shard hash. Local detailed logs are in
`target/strix-download-verify.log`, `target/strix-full-generation-pread.log`, and
`target/strix-python-tests-final.log`. These results were recorded on November 14
in America/Chicago (November 15 UTC).

## Next native integration work

1. Connect the verified checkpoint naming, dtype conversion, sparse Engram reads,
   and bounded weight lifecycle to the Rust model execution path.
2. Implement the remaining native graph operations: RoPE, activation/KV
   quantization, compressed/shared sparse attention, routing, Engram hashing and
   lookup, hyper-connections, and the final logits head. Compare intermediate
   tensors and greedy tokens against the AMD reference using original weights.
3. Integrate target decoding and official prompt encoding with the native server,
   then validate request cancellation, KV lifetime, memory bounds, and API
   behavior on the actual Radeon hardware.
4. Tune weight reuse, transfers and kernels only after matching the correctness
   baseline. DSpark, vision and long-context support require separate acceptance.

The unmodified official model files and their MIT license are kept under
`ports/strix-reference/upstream`. The Rust fork retains the upstream Elastic
License in the repository root.

Sources: [official model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6),
[safetensors 0.8 storage backends](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs),
[wgpu 30 API](https://docs.rs/wgpu/30.0.2/wgpu/),
[WGSL specification](https://www.w3.org/TR/WGSL/),
[AMD Windows compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html).
