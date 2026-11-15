# Resident Qwen on Windows Strix Halo

This fork includes an in-process Qwen inference engine for the Ryzen AI Max+
395 / Radeon 8060S. `qwen-resident.exe` runs the Rust/Axum API and loads our
`qwen_native.dll`, which statically links the pinned llama.cpp/GGML Vulkan or HIP
backend and common Jinja chat library. There is no llama-server subprocess,
HTTP proxy, Python serving runtime, ATen conversion, or DeepSeek compute path.

## Build, start, check

Run from the repository root in PowerShell 7:

```powershell
pwsh -NoProfile -File tools/build-qwen.ps1 -DownloadModel
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" tools/download-qwen.py `
  --lock ports/qwen-resident/model-dflash2.lock.json
pwsh -NoProfile -File tools/serve-qwen.ps1
pwsh -NoProfile -File tools/serve-qwen.ps1 -Check
```

The start command launches a hidden process, waits for actual native warmup and
readiness, then prints its PID, log directory and identity. `-Foreground` keeps
the server attached to the shell, where Ctrl+C shuts it down. No startup task or
service is installed. The launcher refuses an occupied port and never stops an
existing server. It stops only the exact process it just created if startup
fails. In an automation host that owns child process lifetimes, keep that host
alive for the duration of validation.

The default is Q8, Vulkan, 32K context, Q8 KV and DFlash depth 4 with probabilistic
draft verification for positive-temperature requests. `-DFlashDraft 0` explicitly
selects ordinary decoding; `-MtpDraft 2` selects MTP instead. Explicit Q4 or HIP
selection defaults to ordinary decoding. `-Profile q4-fallback` uses the frozen
pre-optimization Q4 executable and DLL. See the [optimization report](QWEN_OPTIMIZATION.md)
for measured results, precision limits and populated-context acceptance.

For an explicitly measured 64K prefill acceleration, build HIP separately with
`tools/build-qwen.ps1 -Backend HIP`, then select
`tools/serve-qwen.ps1 -Profile extended64 -Backend HIP -DFlashDraft 4 -DraftSampling probabilistic`.
HIP passed the 61,076-token retrieval/reconciliation task with a 231.71-second
prefill, versus 388.94 seconds on Vulkan. Its 128K arithmetic check failed, so
the accepted 128K/256K profiles remain Vulkan. The largest accepted input was
251,702 tokens; this is separate from the 262,144-token allocation capacity.

The build discovers the existing VS2022 Build Tools and the read-only CMake/Ninja
in Strata's `.rocm-win` environment. `-CMake`, `-Ninja` and `-VcVars` can supply
explicit tool locations. The pinned Vulkan SDK is verified by SHA256 and publisher
signature, then copied into `target/qwen-resident` using LunarG's documented
`copy_only=1` mode. This does not install drivers or change registry, global PATH,
services or the managed Unsloth environment. VS tool variables exist only in the
build's child shell. All weights, tools, runtime binaries and fetched source stay
under the ignored `target` directory.

`-Backend HIP` builds a separate `build-hip` directory for `gfx1151`. The build
helper uses the existing ROCm 7.14 runtime read-only, with a hash-pinned development
wheel extracted into the workspace. HIP dependencies are loaded from that private
SDK using process-local DLL directories and exact preload paths. Managed Python
packages, global PATH and the system HIP installation are unchanged. The build
manifest records compiler, runtime libraries, flags and binary hashes. Alternative
HIP flags use `-BuildVariant` so they cannot overwrite the ordinary backend.

The Python download helper uses only the standard library. It downloads one
file directly to its final directory, resumes its `.part` file with validated
HTTP Range responses, verifies size and SHA256, then renames it atomically.
There is no second Hugging Face cache copy. An existing verified model is reused.
To recheck the complete file:

```powershell
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" tools/download-qwen.py --quant q8 --verify-only
pwsh -NoProfile -File tools/serve-qwen.ps1 -Quant q8 -VerifyModel -Check
```

Ordinary startup validates the verification receipt, file size, modification time
and lock hash. `-VerifyModel` also rereads and hashes the model. Direct invocation
of the executable requires a previously verified GGUF; use the launcher for the
pinned-model and memory-admission checks.

## Model and backend identity

| Input | Pin |
|---|---|
| Official model | `Qwen/Qwen3.8-27B` |
| GGUF source | `bartowski/Qwen3.8-27B-GGUF` |
| GGUF revision | `87410e7b9a5c21e6ca70c0c9cc5237de909fe621` |
| Q8 file | `Qwen3.8-27B-Q8_0.gguf` |
| Q8 file size | 29,116,388,960 bytes, approximately 27.12 GiB |
| Q8 SHA256 | `a830e736efa71ce29589ff20b8713333a427e0d46ab774723be23800ad54362d` |
| Preserved Q4 file | `Qwen3.8-27B-Q4_K_M.gguf`, 17,442,399,968 bytes |
| Q4 SHA256 | `880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673` |
| llama.cpp | `b54752df40e4d8e53d34ef5a598336b4e4871d39` (MIT) |
| Vulkan SDK | `1.4.363.1`; checksum in backend lock |

The original [Q4 model lock](../ports/qwen-resident/model.lock.json) is preserved as
supplied, including its original pre-validation status. Current acceptance is
recorded separately in validation evidence. The [Q8 lock](../ports/qwen-resident/model-q8.lock.json)
binds its independent file and verification receipt. [Backend lock](../ports/qwen-resident/backend.lock.json)
and `target/qwen-resident/build-manifest.json` record source/tool/binary identity.

The GGUF reports `general.architecture=qwen35`. The pinned source has the dense
QWEN35 graph for its 64-layer hybrid of Gated DeltaNet and gated attention. This
port does not invent a `qwen38` graph enum. The quant includes an MTP block.
`-MtpDraft 1`, `2` or `3` loads that head into a separate draft context sharing the
resident target weights. `-DFlashDraft 1` through `7` instead loads the pinned,
compatible 1.92 GiB Q8 DFlash2 checkpoint. The two modes are mutually exclusive;
zero disables speculation. Only target-verified, consumed tokens count toward
completion usage and decode throughput. Packed weights remain allocated on GPU
and host buffers instead of expanding to FP32.

The adapter uses pinned upstream drafting and verification, bounded recurrent
snapshots and attention-KV rollback. Every request clears target and draft state;
there is no implicit prefix reuse. DFlash2 requires its separate download:

```powershell
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" tools/download-qwen.py `
  --lock ports/qwen-resident/model-dflash2.lock.json
```

Sources: [official model card](https://huggingface.co/Qwen/Qwen3.8-27B),
[quant repository](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF),
[pinned build documentation](https://github.com/ggml-org/llama.cpp/blob/b54752df40e4d8e53d34ef5a598336b4e4871d39/docs/build.md),
[pinned C API](https://github.com/ggml-org/llama.cpp/blob/b54752df40e4d8e53d34ef5a598336b4e4871d39/include/llama.h).

## API and behavior

Base URL `http://127.0.0.1:18083/v1`; the model name is `Qwen/Qwen3.8-27B`.
`GET /health`, `/healthz`, `/readyz` and `/v1/models` expose real readiness,
PID, backend, device, context and last-request timing. Readiness follows a real
128-token synthetic prefill plus a decode step; that warmup is not a quality test.

```powershell
$body = @{
  model = 'Qwen/Qwen3.8-27B'
  messages = @(
    @{role='system'; content='Give concise, practical answers.'}
    @{role='user'; content='Explain how to check whether an exterior door needs new weatherstripping.'}
  )
  max_tokens = 384
} | ConvertTo-Json -Depth 6
Invoke-RestMethod http://127.0.0.1:18083/v1/chat/completions `
  -Method Post -ContentType application/json -Body $body
```

`POST /v1/chat/completions` takes text messages with `system`, `user` and
`assistant` roles; an optional system message must be first and the last message
must be from the user. The actual GGUF Jinja template formats messages; user
content is tokenized separately from template control tokens. Multi-turn requests
send the complete conversation. Each request clears attention KV, recurrent state
and the sampler, so one conversation cannot inherit another's state.

`POST /v1/completions` takes one nonempty raw `prompt` string. Raw prompts do not
parse embedded special-token spellings. Both routes support SSE (`stream:true`),
`stream_options:{"include_usage":true}`, `stop` (one string or up to four), and
`n:1`. Stops are withheld across token and UTF-8 boundaries; their text is omitted.
Usage counts prompt and sampled completion tokens, including sampled termination
tokens and tokens consumed by a stop. It is not the retokenized visible-text count.
SSE ends with a finish-reason chunk, optional usage chunk, and `[DONE]`.

| Parameter | Supported values / default |
|---|---|
| `max_tokens` | 1–8192; default 512; prompt + maximum must fit context |
| `temperature` | 0–2; 0 selects greedy; non-thinking default 0.7 |
| `top_p` | greater than 0 through 1; default 0.8 |
| `top_k` | integer 0–1000; default 20; 0 disables filtering |
| `min_p` | 0–1; default 0 |
| `seed` | integer 0–4294967294; omitted means random |
| `presence_penalty` | -2–2; non-thinking chat default 1.5; raw default 0 |
| `frequency_penalty` | -2–2; default 0 |
| `repeat_penalty` | 0.01–2; default 1 |
| `enable_thinking` | chat only; default false |

Penalties use this request's generated-token history, bounded by `max_tokens`.
Thinking mode defaults to temperature 1.0, top-p 0.95 and presence penalty 0;
explicit sampling values override these defaults. It returns raw `<think>...</think>`
text in `content`, followed by the answer, rather than a separate reasoning field.
The output token limit includes thinking. This deliberately makes ordinary chat
respond directly, while offering an explicit model-supported reasoning mode.

Unknown and unsupported parameters return 400; none are silently ignored.
Unsupported features include tools/function calls, JSON-schema constraints,
logprobs, multiple choices, embeddings, images/video, `reasoning_effort`,
`preserve_thinking`, and persistent/prefix KV caching. The original model's vision
encoder is not loaded. The text-only API is OpenAI-compatible within this subset.

One request owns the GPU; another gets 429 rather than queueing. Loading or failed
workers are unready. Oversized or invalid requests are rejected before decode;
streaming headers wait for native prompt validation. Dropping either a streaming
or non-streaming connection requests cancellation. Cancellation is checked between
GPU batches/tokens; an in-flight GPU batch finishes before admission reopens.
The body limit scales with configured context (32 bytes per context token plus
64 KiB, bounded between 1 and 16 MiB). Default deadlines are one hour through
64K, two hours through 128K, and three hours above 128K. An explicit
`-RequestTimeoutSeconds` overrides these defaults within 1–21600 seconds.
Loopback Host checks
and rejection of browser Origin headers follow the existing Strix serving pattern.

## Preserved Q4 hardware acceptance — 2026-11-15

Machine: Windows, Ryzen AI Max+ 395, Radeon 8060S, 128 GiB installed,
111.65 GiB OS-visible; AMD display driver `32.0.32015.7029`, Vulkan loader
`1.4.357.0`. The tested configuration is 32768 context, full GPU offload,
Q8_0 K/V, Flash Attention, batch 2048, microbatch 512. Timings are measured in
this integrated engine. The selected default is **8 CPU threads**. These are not
upstream NVIDIA numbers or measurements of the separate Strata fallback.

| Workload, 8 CPU threads | Actual tokens | Measured result |
|---|---:|---|
| Warm short prompt, 3 identical runs | 80 input; 384 output each | Median 12.04 decode tokens/s; 223.55 prefill tokens/s; client TTFT 0.407 s |
| Populated prompt plus sustained decode | 4100 input; 384 output | 11.48 decode tokens/s; 290.55 prefill tokens/s; client TTFT 14.16 s |
| Front/middle/end key retrieval | 4082 input; 24 output | All three keys correct; prefill 13.27 s / 307.65 tokens/s |
| Near-32K key retrieval | 31014 input; 24 output | All three keys correct; prefill 141.44 s / 219.27 tokens/s |

Decode rate counts actual one-token decode steps divided by the timed decode
interval; the first sampled token comes from prefill. The 31K retrieval emitted
only 24 tokens, so its 11.03 tokens/s decode observation is not a sustained
near-32K throughput benchmark. The complete suite passed 19 bounded checks,
including arithmetic, structured instructions, summary constraints, Unicode,
multi-turn inventory arithmetic, generated Python tested on eight fixtures,
explicit thinking, greedy/seeded repeatability, exact streaming equivalence,
cross-token stops, raw completions, busy admission, cancellation/recovery and
context overflow. These are useful acceptance checks, not a general model score.

The same three 384-token requests with **16 threads** measured median **11.53
decode tokens/s**, **219.97 prefill tokens/s** and **0.402 s client TTFT**. Their
full output SHA256 matched the 8-thread run. Eight threads were selected from
this modest same-prompt comparison; the small speed difference is not a general
hardware benchmark. Returning to the final 8-thread build produced **12.04
tokens/s** and **0.399 s client TTFT** on another 384-token run, with the same
output SHA256. Batch/microbatch remain 2048/512. Q8 KV was retained after
correctness and headroom checks; the F16 experiment did not fix the driver crash,
so consuming more KV memory provided no demonstrated benefit in that initial Q4
phase. These historical measurements predate the Q8/HIP/speculative experiments.

Memory observation covered 144 samples over approximately seven minutes, including
generation and long prefill. Minimum available physical RAM was **69.388 GiB**;
maximum system commit was **65.262 GiB**. Process working set peaked at **4.315 GiB**,
process private commit at **19.296 GiB**, and Windows GPU committed allocation at
**18.410 GiB** (dedicated allocation peak 14.327 GiB, shared peak 4.084 GiB).
These counters overlap on this UMA system; adding them would double-count memory.
Process working set alone substantially understates the GPU's resident allocation.

The loader reported 15713.94 MiB of Vulkan model buffers, 682.03 MiB of host model
buffers, 1088.00 MiB of Q8 attention KV, 149.62 MiB of recurrent state and
557.02 MiB of Vulkan compute buffers. It logged full layer offload. Warmup and
sustained inference exercised these allocations. Native load mode is explicitly
non-mmap, lazy tensor loading is disabled, and the model remains loaded across
requests. During the measured window **process read and write byte deltas were
both zero**, and system pages-out and page-write operations were both zero.
This supports resident packed-weight inference with no observed disk weight
streaming, beyond merely comparing model file size to RAM.

Windows did record system-wide page-ins (98,749 pages), and the process had
50,594 page faults; these counters include unrelated activity and soft faults.
They do not establish zero paging everywhere. Allocated/committed GPU memory is
not an OS residency lock. Future memory pressure can change Windows' behavior;
the engine does not disable the pagefile or promise immunity from later paging.

### Vulkan driver workaround

Unmodified upstream Flash Attention selection reproducibly crashed this installed
AMD driver during compute-pipeline creation for a longer prefill. The native stack
showed a null read in `amdvlk64.dll`, through `ggml_vk_create_pipeline_func`,
`ggml_vk_load_shaders` and `ggml_vk_flash_attn`. It occurred with both Q8 and F16
KV. The adapter now sets `GGML_VK_DISABLE_FA_FLIP=1` inside its own process before
initializing GGML. That selects upstream's ordinary cooperative-matrix Flash
Attention implementation while keeping Flash Attention, quantized KV and GPU
offload enabled. Source pin and driver remain unchanged. `/health` exposes the
workaround, and the 128-token startup prefill exercises the affected size range
before readiness. Optional `QWEN_NATIVE_DIAGNOSTICS=1` prints native exception
addresses and module offsets; it makes no WER or registry changes.

An initial Python validation-client error consumed the first 2000 bytes of a
successful stream while constructing an error message. It was fixed and all
stream parity/stop checks were rerun successfully; the failed early client
reports are not used as final API or TTFT evidence.

The final default build also passed all nine contract checks, including a
disconnect one second into a populated prefill followed by successful recovery;
cleanup took 2.30 seconds after disconnect. The earlier admission-only cancellation
check is retained in the full-suite report with its narrower scope.

Evidence: [summary](validation/qwen-strix-2026-11-15.json),
[full hardware suite](validation/qwen-strix-2026-11-15/acceptance-t8.json),
[final contract run](validation/qwen-strix-2026-11-15/delivery-contract.json),
[16-thread comparison](validation/qwen-strix-2026-11-15/comparison-t16.json),
[8-thread return run](validation/qwen-strix-2026-11-15/delivery-bench-a2.json),
[memory samples](validation/qwen-strix-2026-11-15/memory-t8.json),
[loader allocations](validation/qwen-strix-2026-11-15/loader.log),
[driver failure trace](validation/qwen-strix-2026-11-15/driver-failure.log), and
[final build identity](validation/qwen-strix-2026-11-15/delivery-build-manifest.json).

## Reproduce acceptance

With the engine running, use the already installed interpreter (serving itself
does not use Python):

```powershell
$python = "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe"
& $python tools/validate-qwen.py --suite all --output target/qwen-resident/my-acceptance.json
pwsh -NoProfile -File tools/measure-qwen-memory.ps1 `
  -DurationSeconds 180 -Output target/qwen-resident/my-memory.json
cargo test -p deepseek-ultrafast-qwen --features native-qwen
cargo test -p deepseek-ultrafast-qwen
cargo fmt -p deepseek-ultrafast-qwen -- --check
cargo clippy -p deepseek-ultrafast-qwen --all-targets --features native-qwen -- -D warnings
cargo test -p deepseek-ultrafast-strix --features native-serve --lib --tests
```

The initial Q4 phase passed: 14 Qwen feature tests, 5 feature-free contract tests, native
CTest, scoped Qwen formatting and warning-free Clippy for Qwen and Strix. Existing
DeepSeek checks passed 45 library tests plus the independent math fixture; three
existing hardware-specific tests remained ignored. Workspace-wide formatting
reports pre-existing changes in untouched CUDA/engine/server files; those files
were preserved. The CUDA GPU suite and full DeepSeek generation were not rerun.

Run the read-only memory observer in a second PowerShell while the acceptance or
benchmark suite generates. It records physical/commit/process/GPU accounting and
raw paging/I/O counters, and never stops or modifies a process. The native backend
refuses requests when available physical RAM drops below 20 GiB; startup uses an
additional conservative allocation estimate. No pagefile setting is changed.

## Boundaries and preserved paths

This is an experimental single-user text engine, not a broad quality evaluation
or multi-user scheduler. Profiles select 32768 (`fast`), 65536 (`extended64`),
131072 (`extended128`) or the official native limit 262144 (`extended256`).
Allocation and startup alone do not validate populated context quality. Context
acceptance and performance are recorded separately for actual populated requests.
YaRN/1M quality, vision and Q6 weights are not established by native-profile boot.

`-Profile q4-fallback` selects the frozen executable and DLL from
`target/qwen-resident/fallback-ab990916`, together with the original Q4 checkpoint.
Their hashes are checked independently of current source/build freshness. This
fallback remains limited to 32768 context and has no speculation. The launcher
refuses to load a second full model from this workspace while one is running.

Original CUDA sources, the DeepSeek 510 GB checkpoint, DeepSeek native/reference
servers, managed Unsloth and the separate Strata 1M fallback remain untouched.
The fallback configuration and executable hashes were independently rechecked:
`34a2c5043d6d6146cfb47ff3b4918429638f2865df7f727d4bafded6513b1449` and
`0e465cb9f92b3d9a51231354300620a13b2365c2494a3852e262b32756215933`.
Port 8081 and its Codex process were not stopped or reconfigured.
