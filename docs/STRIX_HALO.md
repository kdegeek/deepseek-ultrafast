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

The original `run.py` entry point uses greedy target decoding and raw completion
prompts, with bounded token-by-token prefill. The separate `serve.py` entry point
adds a persistent local text API, described below. DSpark, image input and
long-context acceptance remain unimplemented.
The result JSON records prompt and generated token IDs, finite-logit checks and
elapsed time for every forward, AMD/PyTorch identity, peak device allocation,
weight-cache statistics, and hashes of the local provenance and tokenizer files.
Initialization time is reported separately from generation including prefill.

The release has no Jinja chat template. The server uses the
[pinned official encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/encoding)
from the exact original revision, vendored unmodified with its attribution and
SHA256. Chat uses `thinking_mode="chat"` (the official `</think>` assistant prefix),
and the official completion parser. Thinking and tool calls are not supported.

## Persistent local text API

This is an interim Python/PyTorch HIP server, not a native optimized Rust decoder.
It runs one persistent model on one worker thread. The event loop serves health
checks while inference is active. It does not change other model servers or startup.
The existing acceptance interpreter already has the serving dependencies;
`ports/strix-reference/requirements-server.txt` records their verified versions.

```powershell
$python = 'C:\Users\ncrawford\.unsloth\studio\unsloth_studio\Scripts\python.exe'
& $python -u ports/strix-reference/serve.py --checkpoint target/models/DeepSeek-V4.1-Flash --port 18080
```

The listener is fixed to **127.0.0.1**. A port collision fails rather than stopping
another application. Use `--port` to choose another local port. There is no
authentication or browser CORS access. Do not expose this reference on a network.
Base URL for OpenAI clients: `http://127.0.0.1:18080/v1`; a dummy API key is sufficient.

```powershell
Invoke-RestMethod http://127.0.0.1:18080/readyz
$body = @{ model = 'deepseek-ai/DeepSeek-V4.1-Flash'; prompt = 'The capital of France is'; max_tokens = 8; temperature = 0 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:18080/v1/completions -Method Post -ContentType application/json -Body $body
$chat = @{ model = 'deepseek-ai/DeepSeek-V4.1-Flash'; messages = @(@{ role = 'user'; content = 'Reply with only OK.' }); max_tokens = 8 } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:18080/v1/chat/completions -Method Post -ContentType application/json -Body $chat
```

- `GET /healthz`: process/worker state, PID and device identity; stays responsive
  during loading and inference. `GET /readyz`: 200 once initialized, 503 on startup
  or an engine failure. `busy` reports current admission state independently.
- `GET /v1/models`: the single pinned original model ID. No model aliases or hot swaps.
- `POST /v1/completions`: one raw string prompt. `POST /v1/chat/completions`:
  system/user/assistant messages with string content, ending in a user message.
- Greedy only (`temperature=0`), `n=1`. Optional `stop` accepts up to four strings.
  Unknown fields, sampling, tools, thinking, images, arrays of prompts, logprobs,
  JSON-mode requests and multimodal content fail explicitly with 400.
- `stream=true` emits OpenAI SSE deltas as tokens are decoded, final finish reason,
  and `[DONE]`. `stream_options={"include_usage":true}` adds the final usage chunk.
  `: working` comments keep idle streams alive during slow prefill. Stop prefixes
  and incomplete UTF-8 are withheld; generated control/tool markup in chat is an
  engine error. Errors after streaming headers use an `error` event and `[DONE]`.
- Usage counts prompt IDs and sampled completion IDs, including sampled EOS or a
  stop-containing token. EOS/stop => `finish_reason="stop"`; cap => `"length"`.
  For cap/stop-truncated chat, an EOS sentinel is supplied only to the official
  parser; it is neither generated nor counted in usage.
- One active inference, **no waiting queue**. Additional inference requests get
  429 with `Retry-After: 5`. Initialization/failure gives 503. GPU failures quarantine
  the worker until restart; they are not reported as successful completions.
- Default context 256, response cap 64 tokens, body cap 64 KiB, body deadline 10 s,
  inference deadline 1800 s. CLI context is capped at 512 and response cap at 256;
  these ceilings are bounds, not long-context acceptance. Prompt+max_tokens must fit.
- Disconnect/deadline cancellation is checked before every layer and forward.
  An already launched operation finishes; cancellation is not a GPU preemption.
  The slot stays busy until cleanup. KV ring/compressed/index buffers, compressor
  state, Engram history and shared attention globals reset before/after every job.
  Cached immutable weights survive. No conversation cache is shared across requests.

Stop a foreground server with Ctrl+C. For a background instance, read its PID from
`/healthz`, verify its command line, and use `Stop-Process -Id <that PID>`; no startup
task is installed. Validation tools stop their own subprocess on success or failure.
Optional `--audit-log target/api.jsonl` records local token IDs, text, per-forward
timings, cache counters and memory; it is disabled by default because prompts are data.

The default is 8 GiB cache, zero retained share, and an initial prefill batch of
at most 16 tokens. `--prefill-batch 1 --retain-gib 0` restores the original forward
schedule. The cache budget counts resident weights, not all GPU/CPU memory. `--retain-gib`
reserves at most half of that same budget for the smallest non-routed parameters;
the rest remains LRU-managed. Retention is experimental: it reduced loaded bytes
in the measured run but did not reduce total elapsed time, so it is off by default.
No precision or checkpoint changes are involved.
`--prefill-batch` batches only the initial chunk (at most 32 tokens); subsequent
positions use the official one-token decode path. The original CLI and its 8 GiB
baseline remain available unchanged. See the HTTP validation evidence below for
the measured configuration and its limits. Batched and single-token logits are
not bit-identical; acceptance compares finite logits and greedy output tokens on
the recorded short prompts. This does not establish equivalence for other prompts.

## Evidence and acceptance boundaries

- Rust unit tests and Clippy pass. The explicitly requested AMD hardware test
  passes all 30 initial numerical checks.
- A real original FP4 expert matrix, `layers.6.ffn.experts.0.w1.weight`, was fetched
  using bounded HTTP ranges from the pinned checkpoint. Its 2,304 outputs passed
  against an independent CPU reference on the Radeon 8060S. No model weights are
  checked into Git.
- The original 33 Python tests passed without skips on the acceptance machine, including
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
- That original result is one short raw-completion acceptance. The subsequent
  HTTP evidence adds raw/chat serving, request isolation and protocol checks.
  Neither result establishes broad model quality or optimized throughput.
  DSpark, vision, long context and native decoding remain unverified. No
  full-model tokens/s claim is made.

The [full-model evidence](validation/strix-halo-full-model-2026-11-14.json) records
the exact output, token IDs, per-forward logits checks, timings, AMD identity,
cache statistics, storage fix and remaining boundaries. The
[checkpoint manifest](validation/strix-original-checkpoint.json) records every
original shard hash. Local detailed logs are in
`target/strix-download-verify.log`, `target/strix-full-generation-pread.log`, and
`target/strix-python-tests-final.log`. These results were recorded on November 14
in America/Chicago (November 15 UTC).

## HTTP acceptance and measured prefill improvement

The [HTTP evidence](validation/strix-halo-http-api-2026-11-14.json) records the
original revision, device, source hashes, exact token IDs/output, every forward,
cache counters, GPU allocation/reservation, streaming event times and limits.
These runs used the original 48 verified shards, existing AMD environment and
8 GiB total weight cache throughout. Initialization is separate from the timings.

| Same raw prompt, same ` Paris.` + EOS | Seconds including prefill/cleanup | Cache hits |
| --- | ---: | ---: |
| Single-token prefill, before | 164.36 | 0 |
| Single-token prefill, 4 GiB protected within 8 GiB total | 164.37 | 6,549 |
| Batched prefill, fresh trial | 105.82 | 0 |
| Batched prefill, persistent A / after chat B / after disconnect | 122.25 / 107.84 / 117.07 | 0 |
| Single-token prefill, after | 190.08 | 0 |

The initial batch is the measured improvement; protected caching did not improve
elapsed time. Retention reduced materialized device weight bytes from
119,366,627,392 to 93,751,772,608, while batched prefill used 60,790,506,048 for the
same output. Timing varies on this machine; these are short acceptance runs, not
a general throughput benchmark. The original 176.10-second CLI evidence is retained.

Peak PyTorch device allocation was 8,749,762,048 bytes for the single-token
baseline and 8,728,689,664 bytes for the fresh batched trial. Peak reservation
increased from 14,237,564,928 to 16,854,810,624 bytes (16,911,433,728 across the
persistent acceptance sequence). Reported device capacity was 89,108,070,400 bytes.
All cached-weight peaks stayed below 8,589,934,592 bytes. These counters do not
include other applications, OS page cache, or CPU staging memory.

Real HTTP acceptance passed raw completions, official chat (`OK` then EOS),
raw/chat incremental SSE, EOS/user-stop/token-cap usage, context rejection,
health while busy, 429 admission, consecutive A→B→A isolation and recovery after
an early-prefill disconnect. The three repeated A requests had identical token
IDs and recorded logit extrema. Chat content arrived at 87.96 seconds, before
EOS/final usage at 115.51 seconds. Non-streaming chat also returned `OK` and EOS.
Health answered in 2.18 ms during the main acceptance run. Observed disconnect
cleanup took about 0.69 seconds; cancellation latency depends on the current
operation. A late-decode disconnect on the full model remains untested.

All **43 Python tests passed with no skips**, including fixture API contracts,
CPU/AMD kernels, the real expert matrix and synthetic full-architecture state
reset. Installed OpenAI 3.2.0 also consumed captured actual SSE through its stream
consumer; this was client replay, not another generation run. The evidence
separates fixture, client-replay and real-model results and records harness fixes.

```powershell
& $python -m unittest discover -s ports/strix-reference/tests -v
# Opt-in: several minutes of actual GPU inference; starts/stops only its own server.
# Choose a new label each time so prior evidence is not overwritten.
& $python -u tools/validate-strix-api.py --label strix-http-new-run --prefill-batch 16 --acceptance
```

Validation left no server running and installed no startup service. Detailed
local logs/results remain under `target/strix-http-*`; the concise JSON above is
checked into Git. No models, drivers, OS settings or existing services were replaced.

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
