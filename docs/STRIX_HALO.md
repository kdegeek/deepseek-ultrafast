# Strix Halo migration

Target verified on November 14, 2026: GMKtec EVO-X2, Ryzen AI Max+ 395,
Radeon 8060S, Windows 11 Pro. Vulkan driver reports `26.10.1 (LLPC)`.
The existing AMD PyTorch installation reports `2.11.0+rocm7.14.0`.

For current native serving, start `pwsh -NoProfile -File tools/serve-strix.ps1`
and use `http://127.0.0.1:18081/v1`. The [native API guide](STRIX_NATIVE_API.md)
covers setup, supported text requests, streaming, cancellation and hardware
acceptance. This explicitly selects Rust/C++ ATen/HIP; the historical Vulkan
decoder and Python reference results below remain separate. Native serving is
slow, greedy and limited to 128 total tokens; routing ties can vary across runs.

The fork has the following paths:

- `crates/strix`: native Rust/Vulkan compute using wgpu. Packed FP4, FP8, BF16,
  and FP32 matrix-vector kernels plus an **experimental native target-only token
  loop**. It executes the original 40-layer checkpoint without Python, HTTP,
  CUDA, or HIP. Reduced-model tests and the original France completion pass;
  broader numerical/quality equivalence is not established.
- `ports/strix-reference`: the official model architecture with pure PyTorch
  replacements for TileLang kernels and a bounded lazy checkpoint loader.
  It supplies a correctness baseline for the native migration. A short target-only
  text completion has passed with the full original released checkpoint.
- Optional `native-aten` feature: Rust/C++ ATen/HIP compatibility, selected
  explicitly and validated separately. See [native ATen guide](STRIX_ATEN.md).
  The Vulkan failures below are preserved and are not claims about this backend.

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

## Experimental native text decoder

The native decoder has passed **scoped original-model generation acceptance**:
the France completion produces `[11111, 16, 1]`, exactly matching the reference,
including EOS. The officially serialized “Reply with only OK.” prompt produces
`[11932, 1]` (`OK`, EOS). The broader comparison has five next-token mismatches
across 28 tested positions, including the single-token `The` prefix. Logit
differences remain substantial. This is not acceptance as a replacement for the
Python/HIP reference; the command stays explicitly experimental:

```powershell
cargo build -p deepseek-ultrafast-strix --release
.\target\release\strix.exe generate-experimental target/models/DeepSeek-V4.1-Flash 'The capital of France is' 3 8 128
```

Arguments after the raw prompt are generated-token limit, packed GPU weight-cache
budget in integer GiB (default 8), and context bound (default/maximum 128).
An optional final directory records intermediate tensors and logits as JSON.
EOS is included in `generated_tokens` and omitted from rendered text. Prompt
tokenization uses the original tokenizer with no inserted special tokens.
This Vulkan CLI has no chat template or API mode. The separate ATen backend now
has a [native API](STRIX_NATIVE_API.md); the Python API remains a fallback.
The native CLI performs no subprocess or HTTP call.
For the chat acceptance, the pinned official encoder serialized the prompt once;
that string was passed directly to the native executable, which generated both
output token IDs. Python did not participate in native inference.

Implemented text graph: all 40 layers, delayed hyperconnection pre-mixes,
Sinkhorn mixing with source/destination axes preserved, normalized Engram gates,
compressed tokenizer IDs and NumPy-compatible PCG64 hashes, sparse high-offset
Engram row reads, low-rank/grouped attention projections, YaRN and inverse RoPE,
per-layer sliding KV, shared compressed KV and index keys, learned attention
sinks, sqrt-softplus routing with correction bias, six selected FP4 experts and
one FP8 shared expert, and the full original output head.

All learned matrix products run on the AMD Vulkan adapter, including the router,
hyperconnection projections, compressor, Engram projection, and output head.
There is no CPU matrix fallback. Host Rust code schedules the graph and computes
activation quantization, small vector operations, norms, rotations, hyperconnection
mixing, routing selection, and attention over the bounded KV. Each projection
currently reads its result back; this is not a fully resident or tuned GPU graph.

FP8 linear inputs use E4M3/E8M0 blocks of 32. Compressed KV uses FP4 blocks of 16
with E4M3 scales; index keys/queries use FP4 blocks of 32 with E8M0 scales.
Window KV and grouped output-projection inputs follow the official FP8 roundtrip.
BF16 staging, learned zero-value sinks, and 64-key probability staging are explicit.
`wo_a` is grouped by output rows, not treated as a single dense projection.
The E8M0 code-zero decoder was corrected from zero to `2^-127`, and code 255 to
NaN, matching the pinned reference. Nonfinite uploaded scales remain rejected.

The enforced 128-position limit keeps every compressed position inside both
`index_topk=512` and the candidate-block budget. Index projection/key arithmetic
still executes, but selection is provably the complete set in position order.
Long-context pruning, vision, DSpark, batching, stochastic sampling and native
serving are unsupported. Different model configurations and unhandled text
tensors fail at admission; vision-only routing biases and vision/MTP tensors
remain on disk and are explicitly outside target-only text execution.

The checkpoint catalog reads only bounded safetensors headers, validates metadata,
payload extent and index membership, then uses `seek`/`read_exact` for bounded
dense chunks and individual Engram rows. It never maps or duplicates the full
checkpoint. On Windows, its open handles deny writes/deletion during inference.
The packed GPU LRU counts both weights and scales, including alignment and the
dummy scale buffer for unquantized matrices. Eviction precedes allocation; the
prior operation has completed before its matrix is dropped. Host upload buffers,
KV capacity and process/system memory are reported separately. GPU driver
allocation overhead and other processes are not covered by the weight-cache cap.
Process private-byte counters overlap GPU allocations and are not additive to
the cache byte count. Process memory is sampled once per second; system-wide
available-memory readings also include other processes and filesystem cache.
An optional 16 GiB cache was tested against the same raw prompt: generated IDs,
text, and every recorded top-ten logit list were unchanged. It is an opt-in
weight-reuse setting; the default remains 8 GiB.

Reproducible validation:

```powershell
cargo test -p deepseek-ultrafast-strix
cargo test -p deepseek-ultrafast-strix --lib amd_hardware_acceptance -- --ignored
cargo test -p deepseek-ultrafast-strix --lib amd_reduced_decoder_oracle -- --ignored
cargo clippy -p deepseek-ultrafast-strix --all-targets -- -D warnings
cargo fmt -p deepseek-ultrafast-strix -- --check
& ports/strix-reference/validate_native.ps1 -OutputPrefix target/strix-native-final
```

The last command compares **exactly** `[11111, 16, 1]`, including EOS, and returns
exit code 2 if that original-model gate fails. It retains JSON, stderr and process
memory evidence even on a mismatch. An earlier implementation produced
`[11111, 603, 671]` and failed this gate; a text prefix containing “Paris” was not
counted as a pass. The wider FMA reduction tree passed the exact sequence.

`ports/strix-reference/native_oracle.py` is separate validation tooling. Its CPU
fixtures cover exact activation quantization, hyperconnection axes and attention
padding/block boundaries. Its synthetic five-layer checkpoint covers Engram,
both compression ratios, shared KV/index state, and four sequential positions.
The reduced gate uses the existing reference smoke-test tolerances (`rtol=0.03`,
`atol=0.01`) plus exact greedy IDs; these were not widened for the native port.
`--checkpoint ... --trace ... --positions 2` records independent original-model
HIP tensors without changing vendored official sources. These fixture/trace
commands use Python only for the oracle, never as the native decoder backend.

The full-checkpoint result and remaining numerical limitation are recorded in
[`validation/strix-halo-native-2026-11-14.json`](validation/strix-halo-native-2026-11-14.json).

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
Engram tables use direct bounded file reads for requested rows. Headers are
validated before use, with a 32 MiB header limit and a 1 MiB payload limit per
row request. Shape/dtype inspection uses metadata only. Neither sparse rows nor
dense tensors use PyTorch mmap storage: a later layer-14 diagnostic also exposed
an access violation in the former sparse mmap path. The safetensors 0.8 CPU
`pread` payload-slicing path is also avoided because it reads a whole tensor
before slicing it. OS page cache, CPU read buffers, GPU allocator reservations and
temporary dequantization buffers are additional to the weight-cache budget.
Files must not change while the loader is open. The reference is single-device and
single-threaded; its model globals are not safe for concurrent requests.

The [sparse-row I/O evidence](validation/strix-row-io-2026-11-14.json) records
independent byte comparisons for both original Engram tables, including large
file offsets, dtype/order preservation, bounded-read failures and handle lifetime.
All 50 Python tests pass. A fresh original-model run with the new reader produced
` Paris.` and EOS; every recorded forward prediction and logit extremum matched
the previous acceptance. `row_read_bytes`, `peak_row_read_bytes` and
`row_read_limit_bytes` expose sparse-read accounting separately from weight memory.

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

## Remaining native work

1. The Vulkan numerical failures remain unresolved. The optional ATen backend
   has separate full-vocabulary and conditional-route evidence in
   [STRIX_ATEN.md](STRIX_ATEN.md). Exact tied routing can vary in the unchanged
   reference and native runtime; unconditional determinism is not promised.
2. Keep the exact-token gates and full-vocabulary comparisons visible. Broader
   original-model acceptance requires more than locally accurate components or
   two easy completions. Do not change the reference, tune reduction width to
   prompts, or widen tolerances to conceal the remaining disagreement.
3. Extend the native API's bounded text coverage beyond its current 128-token
   cap and test broader prompts. Its official text encoding, cancellation,
   request-state lifetime and HTTP acceptance are recorded separately in the
   [native API guide](STRIX_NATIVE_API.md).
4. Tune weight reuse, transfers and kernels only after matching the correctness
   baseline. DSpark, vision and long-context support require separate acceptance.

## Identical-input numerical investigation (November 14, 2026)

New evidence is recorded separately in
[`validation/strix-halo-numerics-2026-11-14.json`](validation/strix-halo-numerics-2026-11-14.json).
Earlier traces and acceptance records remain historical evidence. The native
decoder remains **experimental**, with no native API or publication in this slice.

Two bounded arithmetic corrections follow the pinned equations:

- Attention now multiplies scores by the FP32 conversion of the model's scalar
  `head_dim**-0.5`, instead of dividing by an FP32 square root. An independent
  one-product, 512-dimensional boundary fixture returns `0.5` on CPU and HIP;
  the previous native expression returned `0.50390625`. The Rust and Python
  regressions require exact output, with no tolerance change.
- Engram now forms the product of the two RMS inverses before applying it to the
  dot product, multiplies by the scalar inverse square root, and uses a pairwise
  FP32 sum for the 5,120 products. With the original layer-14 residual and
  projected KV held identical, gate
  maximum error against CPU float64 falls from `9.07e-8` to `1.43e-8`.
  Nine differing BF16 output elements become zero. This is component evidence,
  not a claim of whole-model parity.

The first hyperconnection projection provides a separate, measured explanation
for early amplification. With identical original embedding and matrix bytes,
native versus HIP maximum difference is `5.15e-5`. Against independent CPU
float64, native maximum error is `7.68e-7`, while HIP's is `5.07e-5`.
Holding the attention result and residual fixed, the resulting coefficient
differences cross 19 BF16 residual boundaries, then 18 FFN normalization
boundaries. The following FP8 quantization removes those particular FFN-input
differences; the residual stream still carries the earlier changes.

Two unchanged HIP executions of `The` produce identical **all-vocabulary** logits
and greedy token `455`. In a separately labelled diagnostic variant, replacing
only layer 0's attention hyperconnection projection with CPU float64 accumulation,
then rounding back to FP32 before the unchanged HIP normalization/mixing, changes
the next token to `201`. Relative L2 logit difference is `0.841294`, maximum absolute
difference `13.684872`. This variant is a sensitivity experiment, never a new
acceptance oracle. The original reference files, weights and environment are unchanged.

The reusable component replay covers original layers 0, 1, 2, 14 and 20, with
473 probes: hyperconnection stages, router projections/selection/weights, learned
projections, grouped `wo_a`, normalization, RoPE, FP8/FP4 quantization, Engram and
sparse attention with sinks. Ratios 1 and 2 are replayed at positions 0 and 1;
the longer teacher-forced sequences cross further compression boundaries.
All sampled router expert selections match. Exact FP8/FP4 and normalization
stage matches are distinguished from projection rounding differences in the JSON.
CPU float64 checks of disagreeing BF16 matrix rows find examples where either
implementation is closer; HIP is not silently treated as an exact real-arithmetic sum.

The direct layer-14 component probe exposed an access violation in the existing
Python mmap row loader. The diagnostic's optional `--bounded-rows` path reads the
same requested row bytes with `seek`/`read`, capped at 1 MiB per request. Layer-1
readback is compared with the unchanged loader. This is confined to validation
tooling; the ordinary full-model HIP comparison uses the unchanged engine/loader.
Older native trace hash IDs were serialized through FP32 and cannot be replayed
as exact large row addresses. The probe derives integer hashes from the pinned
hash module and records the source token explicitly.

Reproduction (new output paths are required):

```powershell
$python = 'C:/Users/ncrawford/.unsloth/studio/unsloth_studio/Scripts/python.exe'
$checkpoint = 'target/models/DeepSeek-V4.1-Flash'
cargo build -p deepseek-ultrafast-strix --release --examples
& $python ports/strix-reference/component_oracle.py --checkpoint $checkpoint --trace target/strix-native-fma --token 671 --layers 0 1 2 14 20 --bounded-rows --output target/new-components
& $python ports/strix-reference/analyze_components.py --checkpoint $checkpoint --components target/new-components --output target/new-components-f64.json
& $python ports/strix-reference/engram_variance.py --checkpoint $checkpoint --components target/new-components --output target/new-engram-calibration
& $python ports/strix-reference/reference_variance.py --checkpoint $checkpoint --prompt The --output target/new-reference-variance
& $python ports/strix-reference/validate_numerics.py --checkpoint $checkpoint --output target/new-full-comparison
```

`teacher_force` is a Rust validation example, not a serving API. It checks Rust
tokenization against supplied original prompt IDs, preserves the 8 GiB packed
weight cache and 128-token bound, and writes all 129,280 logits per position.
The suite includes raw France, officially serialized chat, `The`, arithmetic,
story and greeting inputs. Continuations are teacher-forced; exact predicted
IDs are reported for every position so reference continuations cannot hide a
native generation mismatch. Native code never invokes Python.

The final suite matches **23 of 28 next-token predictions** across six inputs.
France's `[11111, 16, 1]` and chat's `[11932, 1]` remain exact; all arithmetic,
story and greeting positions match. `The` still predicts native `1` versus HIP
`455`, with all-logit maximum absolute error `11.6160233` and relative L2
`0.7047771`. Earlier France and chat prefixes also disagree. All 129,280 logits
are compared at every position; the repeated `The` prefix is bit-identical
across the two native requests. The longest original sequence here is 10
positions, so this is not acceptance of a full 128-position original run.

The comparison first restores FP32 from JSON's round-trip decimals, then computes
errors in float64. True float64 component oracles retain their extra precision.
The report reader uses UTF-8 for native serialized-chat summaries, checks exact
input IDs and vocabulary length, and can recover completed artifacts with
`--reuse-reference` plus `--native-output`. The earlier Windows-code-page reporting
failure and its original inference artifacts remain preserved separately.

Validation passed 31 regular Rust tests, both AMD hardware tests, 46 Python
tests without skips, the original-checkpoint hash-constant check, clippy with
warnings denied, formatting, and diff checks. The native packed-cache peak was
8,589,933,136 bytes against the 8 GiB budget; largest checkpoint read was 20 MiB.
These counters exclude total process/device overhead. No publication or native
API acceptance follows from these results.

PyTorch documents why different reduction and batching choices need not be
bit-identical in its [2.11 numerical accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).
That fact is context for the measured sensitivity, not an acceptance tolerance.

The unmodified official model files and their MIT license are kept under
`ports/strix-reference/upstream`. The Rust fork retains the upstream Elastic
License in the repository root.

Sources: [official model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6),
[safetensors 0.8 storage backends](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs),
[wgpu 30 API](https://docs.rs/wgpu/30.0.2/wgpu/),
[WGSL specification](https://www.w3.org/TR/WGSL/),
[AMD Windows compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html).
