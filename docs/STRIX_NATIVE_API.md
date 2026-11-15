# Native Strix Halo API

On the tested 128 GB Strix Halo machine, from this repository in PowerShell 7:

```powershell
pwsh -NoProfile -File tools/serve-strix.ps1 -CacheGiB 32
```

This starts a native Rust/C++ ATen/HIP API at **http://127.0.0.1:18081/v1** on
the Radeon 8060S. The server loads no Python runtime. The launcher defaults to
this machine's existing Unsloth native libraries and original checkpoint; it
builds the shim and `strix.exe` if either artifact or its source manifest is
missing/stale. `-Build` forces a rebuild; `-Check` checks freshness without
building or starting a server. `-BuildOnly` builds and records the manifest
without starting inference. `-SitePackages`, `-Checkpoint`, `-Port`,
`-CacheGiB` (3–32) and `-Context` override the defaults. The minimum cache holds
the full FP32 output head plus the HC reserve. It installs no packages and
changes no driver, OS or persistent environment setting.

Without `-CacheGiB`, the conservative default remains 8 GiB. The tested 32 GiB
configuration avoids weight-cache thrashing on this machine; it does not change
model arithmetic or the routing policy. The comparison below is specific to
the original France prompt, not a guarantee for every workload.

Freshness checks cover Rust/C++ sources, the executable/shim, selected compiler
and ABI inputs, all Torch DLLs and the bundled HIP/ROCm runtime directories,
including rocBLAS/hipBLAS and their kernel data. A source or runtime change
during compilation fails the build check instead of certifying that artifact.

The supported subset is **text only, one greedy completion, at most 128 total
prompt-plus-output tokens**. Generation is slow and experimental: exact ATen
router-score ties can choose different experts across runs. The API does not
promise bit-identical responses. This is separate from the Vulkan experiment
and the validated Python reference on port 18080.

Wait for `GET /readyz` to return 200. `GET /health` (also `/healthz`) reports
initialization, busy state, runtime identity, completed/cancelled counts and the
last request's timing, sampled IDs and memory metrics. `GET /v1/models` lists
`deepseek-ai/DeepSeek-V4.1-Flash`.

```powershell
$body = @{
    model = 'deepseek-ai/DeepSeek-V4.1-Flash'
    messages = @(@{ role = 'user'; content = 'Reply with only OK.' })
    max_tokens = 8
    temperature = 0
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:18081/v1/chat/completions -Method Post -ContentType application/json -Body $body
```

Raw prompts use `/v1/completions` with `prompt` instead of `messages`. Supported
parameters are `model`, `prompt`/`messages`, `max_tokens` (1–64, default 16),
`temperature: 0`, `n: 1`, `stop` (up to four strings), `stream`, and
`stream_options: {"include_usage": true}`. Unknown options fail with 400.
Chat accepts only system/user/assistant string messages and requires a final
user message. It follows the pinned official encoder's text chat mode,
including adjacent user merging and mid-conversation system transitions.
Control-token injection, vision, tools, reasoning, batching, sampling and
logprobs are unsupported.

Streaming uses incremental OpenAI SSE chunks, an optional final usage chunk,
and `[DONE]`. Stop text and incomplete UTF-8 prefixes are withheld. Usage counts
all sampled tokens, including EOS and the token that completes a user stop.
Once streaming starts, later errors are SSE error objects followed by `[DONE]`.

Only one request is admitted. Concurrent inference gets 429 with `Retry-After`;
health remains responsive. Disconnects, deadline expiry and Ctrl+C cancel the
owning request. Cancellation is checked between complete native forwards, so a
GPU forward already underway finishes before cleanup. Request state resets
before admission reopens, while weights and immutable constants stay warm.
Unsupported generated chat controls reject that request after cleanup; a
decoder/runtime failure puts readiness into a failed state and requires restart.

The request deadline is 3,600 seconds, allowing for this slow backend's bounded
context. A client may need a similarly long timeout. The server binds only IPv4
loopback, rejects browser `Origin` headers and non-loopback `Host` names, and
limits JSON bodies to 64 KiB. It provides no remote serving/authentication mode.

The default eight-GiB budget is for cached weights, including the HC reserve;
it does not cap native allocator reservations, temporary tensors, KV, host
staging or other processes. Health reports those separately where measured.
The launcher stays attached to the console; Ctrl+C stops its server. It does
not install a background service or start at boot.

## Verification

The contract suite uses a fixture engine and includes actual TCP disconnects
for streaming and nonstreaming requests, health/busy admission, errors,
request-local failure recovery, usage, timeout, and incremental text. Forty
fixtures are generated from the pinned unchanged official chat encoder,
covering role sequences through four messages, empty text and Unicode.
These fixture checks do not establish model correctness. The complete
`native-serve` CPU suite passes 46 tests, including 12 serving/contract tests;
three separately configured hardware tests are excluded from that command.

Original-checkpoint HTTP acceptance on this Radeon 8060S passed raw France
(` Paris.` then EOS), streamed chat (`OK` then EOS), raw → chat → raw isolation,
SSE usage/DONE, busy rejection, user-stop accounting, disconnect recovery and
context rejection. The native process loaded no Python runtime. At an 8 GiB
weight cache, the France request took 189 seconds and chat took 271 seconds;
these include prefill. The tracked
[evidence](validation/strix-halo-native-api-2026-11-14.json) distinguishes the
full API run from a final-build cancellation/recovery check. This does not
establish broad prompt quality or original-model behavior at 128 positions.

```powershell
cargo test -p deepseek-ultrafast-strix --features native-serve
cargo clippy -p deepseek-ultrafast-strix --features native-serve --all-targets -- -D warnings
```

The opt-in hardware validator starts/stops only its own native process and
combines raw → chat → raw isolation, SSE/usage, user-stop accounting, busy
rejection, responsive health, disconnect cleanup and subsequent generation.
It records source/executable/shim hashes, runtime identity, token IDs and memory
statistics, and checks the loaded-module list for Python runtimes.

```powershell
$env:STRIX_ATEN_SITE_PACKAGES = "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Lib/site-packages"
$env:STRIX_ATEN_SHIM = (Resolve-Path target/strix-aten-server-build/strix_aten.dll).Path
& "$env:USERPROFILE/.unsloth/studio/unsloth_studio/Scripts/python.exe" tools/validate-strix-native-api.py --label native-api-new-run
```

Python in that command is the external HTTP test client, never part of the
native server. Each run records evidence in `target/<label>/report.json`.
Numerical fidelity and its
remaining boundaries are described in [STRIX_ATEN.md](STRIX_ATEN.md).

## Cache comparison on this machine

Both 32 GiB requests returned ` Paris.` and EOS, preserving all seven expected
next-token predictions. The second request reused all 122,773 cached matrix
entries with zero new uploads. It used the same delivery executable as the final
cancellation/recovery check. The 8 GiB API run used the preceding build, whose
only later serving change was the startup minimum-cache guard.

| Weight-cache budget | First France request | Repeated France request | Peak native allocation | Peak native reservation |
|---|---:|---:|---:|---:|
| 8 GiB | 189.17 s | 189.58 s, after chat | 8.90 GB | 11.84 GB |
| 32 GiB | 130.95 s | 114.75 s | 27.79 GB | 28.01 GB |

Times include five prefill positions and two subsequent forwards. The 32 GiB
process started with an empty GPU weight cache; the filesystem cache was not
flushed. The 8 GiB memory peaks cover the full HTTP acceptance suite; the 32 GiB
peaks cover the two France requests. Other system workloads were uncontrolled, and these are two requests,
not a general throughput benchmark. Approximately 68 GiB of OS-visible RAM
remained available after the first 32 GiB request. Total allocation/reservation
includes more than cached weights.

[Cache evidence](validation/strix-halo-cache-2026-11-15.json) records exact
tokens, per-forward timings, memory counters and source/executable hashes.
`tools/benchmark-strix-cache.py --cache-gib 32 --label NEW_LABEL` reproduces the
bounded two-request experiment using the SDK environment shown above. It stops
only its own server when finished.
