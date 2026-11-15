# Qwen Q8 optimization on Strix Halo

Accepted: resident Q8 inference, native populated context through 251,702 input
tokens, and measured MTP/DFlash/backend acceleration. The interactive default is
Vulkan Q8/DFlash4 with probabilistic verification. Final same-binary A/B/A measured
13.49 verified output tokens/s versus 7.37 for ordinary Q8: **1.83x throughput**
and **45.0% less full-request time**. HIP is an explicit faster-prefill option at
64K; its 128K arithmetic failure is retained and that profile is rejected.

The [machine-readable evidence](validation/qwen-optimization-2026-11-15.json)
contains exact model/build/binary identities, all measured alternatives, memory
and paging observations, accepted/rejected checks and the final running-server
receipt. Bounded full acceptance reports and manifests are in the adjacent
`validation/qwen-optimization-2026-11-15/` directory. Historical Q4 evidence is
unchanged.

This report covers the embedded Rust/C++ engine on the Ryzen AI Max+ 395 and
Radeon 8060S. It does not use an external model server. The target is the pinned
Qwen3.8-27B Q8_0 GGUF, with all packed weights resident and full GPU layer offload.
The original Q4 engine and checkpoint remain available through `q4-fallback`.

## Implementation

The downloader selects an independent pinned Q4/Q8 lock, resumes one `.part`
file in the final model directory, verifies size and SHA256, and writes a receipt.
Existing Q4 receipts remain valid. The Q8 file is 29,116,388,960 bytes and its SHA256
is `a830e736efa71ce29589ff20b8713333a427e0d46ab774723be23800ad54362d`.
The compatible DFlash2 Q8 file is 2,056,414,816 bytes, SHA256
`651a1b2930c9825080480f79cf38057e681c9b175d580a357c14f67cf77ccf49`.
Both were downloaded completely and checksum-verified before loading.

Native health reports the loaded GGUF quantization, actual device backend,
context, KV type, speculative mode, draft model and residency settings. Launcher
readback also checks the loaded native DLL path, executable, batch sizes, threads,
Flash Attention and deadline. Each new run preserves its build manifest, model
receipts and binary hashes. Custom context sizes round up to the backend's
256-token allocation granularity before admission and identity comparison.

MTP shares the target's resident weights and loads its trained prediction head.
DFlash2 uses a separately pinned small resident model. Both use the pinned upstream
drafting implementations, target verification, bounded recurrent snapshots and
attention-KV rollback. The adapter retains the current token and accepted draft
prefix, leaves the verified bonus token pending, and clears target/draft state at
request termination or cancellation. There is no implicit conversation cache.

Greedy drafting uses upstream target sample-and-match, including for stochastic
target requests: a proposal is accepted only when it equals the genuine target
sample. `-DraftSampling probabilistic` enables DFlash2's actual conditional draft
distributions and the upstream rejection verifier for positive temperatures.
That verifier accepts with `min(1,p/q)` and samples normalized `(p-q)+` on rejection;
the target's penalties and filters remain part of `p`. Temperature zero continues
to use target sample-and-match. Usage and throughput count only consumed verified
output tokens, never rejected or unused proposals.

The HIP backend builds separately with ROCm 7.14's clang/MSVC ABI for `gfx1151`.
Missing development files came from a checksum-pinned wheel extracted under
`target`; installed vendor runtime packages were read only. Runtime DLLs are
preloaded from the private SDK. An initial real-inference failure exposed ROCm's
invalid joining of `\\?\` drive paths with forward slashes. The Rust loader now
passes normal absolute drive paths, preserving Unicode; subsequent HIP inference
succeeded. No driver, global PATH, pagefile, OS or managed-package change was made.

Kernel flags were checked against the pinned implementation. On `gfx1151`, the
Q8 path already selects the RDNA WMMA/MMQ kernels at every batch size; forcing
MMQ would not change that path. `HIP_MMQ_MFMA` controls the CDNA MFMA path, not
this RDNA GPU. HIP graphs and `HIP_NO_VMM` are already enabled by the upstream
defaults. The graph-disabled experiment is recorded separately rather than
combining unrelated switches and attributing any change to one of them.

Two host-side changes were measured: output-buffer capacity is bounded to the
required verification rows instead of the entire prefill batch, and a dirty-state
guard skips a second physical clear when the previous request already cleared
everything. The latter retains physical clearing and MTP carryover reset.
`QWEN_BATCH_LOGIT_CAPACITY=1` and `QWEN_ALWAYS_CLEAR=1` preserve comparison paths.
Timing fields distinguish prefill/decode from terminal cleanup and full request
time. Sampling timings include synchronization and must not be read as pure CPU
cost.

The final same-binary HIP clear control reproduced the identical 384-token output
in both modes. Skipping a redundant initial clear reduced median client TTFT from
2.322 seconds to 0.261 seconds over two repeats; decode rates 6.99/7.03 tokens/s
were effectively unchanged. Terminal physical clearing remains enabled. This is
a latency improvement, not a claim that host clearing increased decode throughput.

Disabling HIP graphs changed ordinary greedy decode from 6.99 to 7.12 tokens/s
and DFlash4 greedy decode from 12.99 to 13.18 over two repeats, with identical
outputs for each paired comparison. These small differences are not enough to
establish a robust graph-setting advantage. Graph-disabled probabilistic DFlash
measured 13.89 tokens/s at temperature .7, but its full request took 30.73 seconds,
versus 28.80 seconds for the earlier Vulkan probabilistic candidate. HIP's roughly
three-second terminal clear makes highest decode rate and shortest full request
different selection criteria. Physical terminal clearing remains enabled.

## Exploratory measurements

The short benchmark uses the same 80-token maintenance prompt, seed, temperature
zero and 384-token budget. The 4K workload emits 384 tokens after a 4100-token
prompt, following a separate three-key retrieval check. Rates exclude the first
token, which comes from prefill. Different numeric execution paths can produce
different text; these are workload measurements, not a universal speed ratio.

| Backend / mode | Short decode tokens/s | Populated 4K decode tokens/s |
|---|---:|---:|
| Vulkan Q8, original batch-sized output capacity | 7.18 | 7.25 |
| Vulkan Q8, bounded output capacity | 7.30 | 7.33 |
| HIP Q8, bounded output capacity | 7.21 | 7.08 |
| Vulkan MTP 1 | 11.08 | 11.97 |
| Vulkan MTP 2 | 12.96 | 13.75 |
| Vulkan MTP 3 | 12.34 | 13.20 |
| Vulkan DFlash2 depth 2 | 12.27 | 13.11 |
| Vulkan DFlash2 depth 4 | 13.23 | 14.14 |
| Vulkan DFlash2 depth 7 | 9.88 | 10.96 |
| HIP MTP 1 | 10.43 | 10.84 |
| HIP DFlash2 depth 4 | 12.81 | 14.51 |

Ordinary baseline rows use three repeated short runs; speculative rows use two.
An isolated two-job CPU fixture compilation overlapped parts of the exploratory
DFlash2 measurements (17:28–17:32 UTC). Final selection measurements run after
builds stop. The preserved Q4 baseline was 12.04 tokens/s; Q8's larger weights make
its ordinary decode slower. Q8 acceleration is compared against Q8 controls.

HIP's slightly faster 4K DFlash decode did not make its full request faster:
42.48 seconds versus Vulkan's 38.46 seconds, including prefill and cleanup. HIP
MTP's client TTFT fell to about 0.27 seconds after skipping the redundant initial
clear, while terminal cleanup remained approximately three seconds.

The subsequent same-binary prefill comparison retained batch 2048, microbatch 512
and Q8 KV. At 4100 populated tokens, microbatches 512/1024/2048 measured
365.22/355.41/326.48 prefill tokens/s and 14.12/13.80/13.56 decode tokens/s.
F16 KV with microbatch 512 measured 362.28 prefill and 13.93 decode tokens/s while
requiring more cache memory. All four configurations passed the three-key check;
none justified replacing the existing batching/KV defaults.

With the API's normal temperature .7 / top-k 20 / top-p .8 / presence 1.5 settings,
two repeated short runs measured 7.39 tokens/s for ordinary Vulkan Q8, 11.89 for
MTP depth 2, 12.96 for DFlash2 depth 4 with greedy proposals, and 13.50 for DFlash2
depth 4 with probabilistic proposals and exact rejection verification. Each mode
reproduced its own complete output at the fixed seed. This comparison uses the
same prompt and output budget, but different modes need not produce identical
sampled text.

## Final same-binary comparison

The final serial A/B/A used the same executable, Vulkan DLL and build manifest,
the same 80-token prompt, seed 2026, temperature .7, top-k 20, top-p .8 and presence
penalty 1.5. Each phase generated three 384-token responses after warmup. No CPU
build or competing model benchmark ran during this window. No direct GPU
temperature/clock telemetry was collected; the ordinary return phase checks
timing drift instead of claiming thermal equivalence.

| Phase | Median verified decode tokens/s | Median full HTTP seconds | Median client TTFT seconds |
|---|---:|---:|---:|
| Ordinary Q8, before | 7.4497 | 51.8359 | 0.3828 |
| DFlash4, probabilistic, selected | 13.4886 | 28.7957 | 0.3894 |
| Ordinary Q8, after | 7.2276 | 53.4359 | 0.4207 |

The six pooled ordinary samples have median 7.3691 tokens/s and 52.3847 seconds
per request. The candidate is 1.8304 times that rate and takes 45.03% less full
request time. The return control is 2.98% slower than the first control and has
the identical complete output hash. Every within-mode repeat also matches.
The candidate accepted 231 of 592 proposed draft tokens (39.02%) per repeated
request; output usage counts only its 384 consumed verified tokens, and decode
rate uses the 383 tokens after prefill. It does not count proposals
as generated output. Available physical RAM stayed above 58.50 GiB across all
three phases; the observer detected no reserve violation.

The selected default minimizes measured full short-request latency. HIP's earlier
13.89-token/s sampled result had a slower 30.73-second full request because of
terminal cleanup. Larger contexts remain explicit so their allocation and cleanup
cost do not burden ordinary interactive requests.

## Correctness and precision evidence

All seven short greedy control cases matched ordinary Vulkan across every tested
MTP/DFlash configuration: arithmetic, prose, code, multi-turn edits, cross-token
stop, EOS and a one-token budget. Exact sampled comparisons across configurations
failed and are retained as failures. The 384-token greedy benchmark also differs
between ordinary and speculative execution; there is no claim of bitwise equality.

Bounded native logit diagnostics localized the first divergence on identical
accepted-token prefixes. MTP first differed at output index 88: the ordinary
target's leading margin was 0.05629, whereas the MTP execution's target ranked the
other candidate first by 0.04597. DFlash first differed at index 23, where the
ordinary margin was only 0.000309 and the DFlash execution's opposite margin was
0.05205. In both cases the verifier chose its own penalized target argmax. The
difference was present in target logits before verification; the diagnostic does
not establish broad numerical equivalence or model quality. Trace timings are
explicitly invalid for benchmarking.

An independent post-run check asked: "A cabinet shop starts with 48 drawer fronts,
makes 17 more, and ships 29. Return only the number remaining." The expected
answer is 36. With seed 2026 and thinking disabled, both ordinary Q8 and selected
DFlash4 returned the same incorrect answers: 46 at temperature .7 and 16 at
temperature zero. Enabling thinking produced the correct final answer, 36, in
both modes. This failure is retained in the
[independent check](validation/qwen-optimization-parent-review-2026-11-15.json).
It is not specific to speculative decoding on this prompt, but the comparison
does not isolate whether the model, quantization or shared backend caused it.
Passing the earlier bounded suite does not establish reliable arithmetic or
general quality in non-thinking mode. No sampling defaults or acceptance
thresholds were changed to hide this result.

An independent source review found the retained-prefix calculation, target/draft
KV rollback, MTP hidden-row selection and recurrent snapshot bound consistent
with the pinned upstream server. CPU tests call the real upstream rejection
helper using a tiny in-memory model and controlled target logits. Five thousand
trials cover equal, disjoint, point-mass, sparse selector-like and dense draft
distributions under temperature .7, top-k 20, top-p .8 and presence penalty 1.5.
There were no target-support violations; the largest empirical deviation was
2.655 standard errors against a predeclared six-error bound. Seed, penalty-history
and accepted-prefix/bonus contracts also passed. This fixture does not substitute
for Qwen GPU acceptance.

## Context and residency methodology

Profiles allocate 32K, 64K, 128K or native 256K capacity. Body limits scale with
context and deadlines can be extended to six hours. Allocation and warmup are
reported separately from populated-context quality. The extended fixture uses
seeded random secrets in front/middle/end depth bands, a stock reconciliation
requiring transactions from all three regions, and sustained procedural output.
Reports retain actual token usage, fixture seed/positions/hash, exact expected and
observed answers, TTFT and generation rates. A passed retrieval is a bounded
acceptance check, not a general long-context benchmark.

At 65,536 allocated capacity, the same seeded fixture populated **61,076 prompt
tokens**. Both ordinary Q8 and DFlash4 recovered all three secrets and the correct
closing stock of 72, then produced 384 tokens of procedural output. Ordinary Q8
prefilled in 389.93 seconds at 156.63 tokens/s and decoded at 6.62 tokens/s.
DFlash4 prefilled in 388.94 seconds at 157.03 tokens/s and decoded at 10.89 tokens/s,
a 64.56% decode-rate improvement. Its consumed draft acceptance was 236/572
(41.26%). This greedy fixture uses target sample-and-match even though the profile
also enables probabilistic proposals for positive-temperature requests. DFlash did
not materially accelerate the long prefill. Disconnecting a repeat of the known
populated fixture after five seconds recovered in 1.67 seconds ordinary / 0.79
seconds DFlash, followed by the correct isolated arithmetic answer.

Observed physical-memory minima were 59.874 GiB ordinary / 57.574 GiB DFlash.
Both observation windows recorded zero process-read bytes after loading and no
system page outputs. System page inputs were nonzero and are retained in the
evidence; they cannot be attributed to this process from those counters alone.

At 131,072 capacity, the next independent fixture populated **123,299 prompt
tokens**. It recovered all three new secrets and closing stock 125, then emitted
384 tokens. Prefill took 1325.85 seconds (93.00 tokens/s); client TTFT was 1325.90
seconds and sustained decode was 8.97 tokens/s. Consumed draft acceptance was
241/560 (43.04%). Cancellation of a repeat recovered in 0.46 seconds and the
isolated recovery answer remained correct. This is completed populated-context
acceptance, not merely successful allocation or warmup.

At the native **262,144 capacity**, the final fixture populated **251,702 prompt
tokens** (252,086 including output). All three randomized secrets and closing
stock 164 were correct. It then emitted 384 tokens at **6.18 decode tokens/s**.
Prefill took **4919.77 seconds / 82.00 minutes** at 51.16 tokens/s; client TTFT was
4919.88 seconds, the worst completed TTFT in this phase. Draft acceptance was
234/588 (39.80%). The known populated fixture also passed cancellation/recovery,
with 0.83 seconds from disconnect to idle and the correct recovery answer.

The near-256K observation retained at least **41.633 GiB** available physical
memory, with maximum GPU committed 44.185 GiB and process private commit 45.051 GiB
(overlapping views). There were zero process-read bytes during the observation,
but system-wide counters recorded 1,246,287 pages input and 3,950 pages output;
these are not process-attributed paging evidence. No reserve-pressure event was
observed. The 82-minute prefill makes this an explicit extended/offline profile,
while interactive startup retains 32K allocation. Native-capacity acceptance here
is one seeded retrieval/arithmetic fixture plus sustained output and cancellation,
not a general claim about every 256K task.

The matched 61,076-token HIP run then passed the same retrieval, stock 72 and
384-token output gates. Its prefill took 231.71 seconds at 263.59 tokens/s, versus
388.94 seconds on Vulkan: 40.43% less prefill time. Decode was 9.98 tokens/s,
slower than Vulkan's 10.89, but total native time including cleanup fell from
424.23 to 274.84 seconds. HIP cancellation recovered in 6.00 seconds and preserved
the correct isolated answer. Physical-memory reserve remained at least 55.944 GiB,
with zero process reads and no system page outputs during that window. This
measured long-prefill win triggered the matched HIP 128K acceptance stage.

HIP at 123,299 input tokens completed prefill in 625.14 seconds, versus Vulkan's
1325.85 seconds, and retrieved all three randomized keys. It nevertheless failed
the distributed stock reconciliation: 155 instead of the expected 125. Its
384-token continuation decoded at 7.07 tokens/s. This profile is rejected for
long-context acceptance; its faster timing is not an accepted quality result.
The serial runner stopped before HIP 256K, which remains untested. The full
failed response and fixture are preserved alongside the passing evidence.
No kernel defect is inferred from this single model-quality failure. Vulkan
remains the accepted 128K/256K path, while HIP is an explicit accepted 64K option.

Startup estimates packed weights, KV and runtime memory, retaining at least
20 GiB of physical reserve. Native admission repeats the reserve check between
batches and decode steps. Long prefill logs completed GPU token milestones without
prompt text. The observer records physical/commit/process/GPU counters and paging
and I/O deltas. UMA allocations overlap: process working set and GPU memory must
not be added together. No mmap, lazy weights or disk/expert offload is used.

## Verification

After integrating probabilistic verification, 17 Qwen feature tests and five
feature-free contract tests passed. Both Vulkan and HIP native CTest builds
passed two tests, including the 5,000-trial rejection fixture. Existing DeepSeek
checks passed 45 library tests and the independent math fixture; three existing
hardware tests remained ignored. Qwen and Strix Clippy passed with warnings
denied, and scoped Qwen formatting passed. Workspace-wide formatting still
reports pre-existing issues in untouched files; no unrelated formatter changes
were applied and no 510 GB generation was run.

The downloader passed seven offline regressions, including resume integrity,
hash failure retention, independent Q4/Q8 receipts and default selection. Launcher
tests passed offline identity/freshness checks and ten live read-only profile
combinations. All four staged extended runs passed both populated task quality
and disconnect cancellation/recovery on Vulkan. HIP 64K also passed both gates;
HIP 128K's arithmetic failure is retained above.

The selected Vulkan Q8/DFlash4 profile passed all 19 existing hardware checks,
including the nine API checks. This covers arithmetic, structured instructions,
Unicode, summary, multi-turn inventory, eight code fixtures, opt-in thinking,
strict unsupported-parameter rejection, seed repeatability, streaming/stop/usage,
raw completion, cancellation during decode and prefill, context overflow recovery,
short sustained generation and populated 4K/31K retrieval. HIP with graphs disabled
also passed all 19. Separate 17-request speculative regressions passed on Vulkan
32K, HIP 32K with graphs disabled and HIP 64K with graphs enabled: filtered and
unfiltered stochastic A/unrelated-B/A/stream-A checks, plus output budgets 1–9.

A five-second deadline on the accepted 61,076-token fixture returned HTTP 504 in
5.023 seconds, incremented native cancellation accounting, became idle and then
answered the recovery arithmetic correctly. The current launcher also started
the frozen Q4 executable/DLL and passed its arithmetic smoke test. Five preserved
fallback/legacy binary, configuration and manifest hashes were rechecked unchanged.

At delivery, the bare launcher started the selected Q8/Vulkan/32K/DFlash4 default
on `http://127.0.0.1:18083/v1` (PID 60736). Its arithmetic smoke, independent
`/readyz` readback and full launcher profile comparison passed with the engine
idle. The receipt records the observation time, process start time, hashes and
loaded identity; it is a delivery snapshot, not a permanent availability claim.

## One-million-token investigation

The official extension uses YaRN factor 4 with original context 262144 and a
1,000,000-token deployment target. This is distinct from 1,048,576 tokens. YaRN
is not enabled or accepted in this native engine; the current upper bound is
262144. The pin exposes the required scaling APIs, but source support does not
establish numerical or task quality at extrapolated positions.

Logged buffers and the pinned cache geometry give 34,816 bytes per target KV
token and 10,880 bytes per DFlash KV token. At 1,000,000 tokens these caches alone
would occupy 32.42 and 10.13 GiB. Context-sized masks add further memory; recurrent
state and the 200 MiB hidden-extraction buffer are fixed at this batch size.
Projecting from the completed 64K observation leaves about 27.8 GiB available for
ordinary Q8, but only 14.2–15.1 GiB for the current DFlash4 configuration. These
are estimates under comparable system load, not measured 1M allocations. Current
DFlash1M would violate the 20 GiB reserve, and its conservative admission formula
also exceeds OS-visible physical memory. It was not loaded.

An upstream compact sliding-window draft cache could reduce that cache to about
26.56 MiB. It remains unadopted: eviction, partial rollback, cancellation and
quality need separate verification. The draft model is trained for 262144 with a
2048-token local window and a different rotary layout from the target; simply
copying target YaRN settings is not draft acceptance. Ordinary Q8 is the sensible
first path for any future extrapolation test, subject to fresh memory admission,
an intermediate populated stage and a bounded latency budget.

Using the completed 123,299- and 251,702-token prefills, a nonnegative
linear-plus-quadratic fit estimates **19.66 hours** to prefill 1,000,000 tokens on
this Vulkan path. Scaling the larger measurement by token count alone gives 5.43
hours, while quadratic-only scaling gives 21.57 hours. These are extrapolations,
not measured performance or guaranteed bounds. Together with DFlash's failed
memory-reserve estimate and unvalidated YaRN quality, they do not justify a 1M
profile in this phase. The largest populated acceptance delivered is 251,702 input
tokens within the native 262,144 profile; the separate legacy Strata configuration
and executable remain untouched.

## Launch and reproduce

With the port free, these commands select the interactive default, an extended
profile, ordinary Q8, or the frozen fallback respectively:

```powershell
pwsh -NoProfile -File tools/serve-qwen.ps1
pwsh -NoProfile -File tools/serve-qwen.ps1 -Profile extended64 -Backend HIP -DFlashDraft 4 -DraftSampling probabilistic
pwsh -NoProfile -File tools/serve-qwen.ps1 -Profile extended128
pwsh -NoProfile -File tools/serve-qwen.ps1 -Profile extended256
pwsh -NoProfile -File tools/serve-qwen.ps1 -DFlashDraft 0
pwsh -NoProfile -File tools/serve-qwen.ps1 -Profile q4-fallback
```

The default is Q8/Vulkan/32K, batch 2048, microbatch 512, eight threads, Q8 KV,
DFlash4 and probabilistic draft verification for positive temperatures. Explicit
`-MtpDraft 2` selects MTP with DFlash disabled. Explicit `-MtpDraft 0` or
`-DFlashDraft 0` selects ordinary decoding. Q4 and HIP selections default to
ordinary decoding unless a speculative mode is explicitly requested. Default
deadlines are one hour through 64K, two hours through 128K, and three hours above
128K; an explicit deadline overrides them. No command silently stops an existing
engine or loads another full model beside it.

The benchmark runner owns only the processes it starts and serializes GPU work.
It records exact run identities and memory observations, then verifies PID, path
and start time before stopping each engine. Output directories must be new:

```powershell
pwsh -NoProfile -File tools/benchmark-qwen-profiles.ps1 `
  -Profiles tools/qwen-final-profiles.json -Repeats 3 -BenchmarkTemperature 0.7 `
  -OutputDirectory target/qwen-resident/my-final-comparison
pwsh -NoProfile -File tools/benchmark-qwen-profiles.ps1 `
  -Profiles tools/qwen-extended-profiles.json -Extended -Cancellation `
  -OutputDirectory target/qwen-resident/my-extended-acceptance
pwsh -NoProfile -File tools/benchmark-qwen-profiles.ps1 `
  -Profiles tools/qwen-accepted-profile.json -Acceptance `
  -OutputDirectory target/qwen-resident/my-final-acceptance
pwsh -NoProfile -File tools/test-qwen-launch.ps1 -LiveDefaults
```

`optimize-qwen.py --suite spec-contract` checks the active engine's seeded
sampling, interleaved-request isolation, streaming equivalence and output-budget
boundaries. `--suite deadline` requires an intentionally short server deadline and
`--cancel-from` pointing to a passed extended fixture; it verifies a real HTTP 504,
native cancellation and recovery. Diagnostic `--suite trace` requires explicit
`QWEN_TRACE_TARGET_LOGITS=1` and is never performance evidence. The native CTest
distribution fixture runs without loading the Qwen checkpoint.

## Sources

- [Official model and native context](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Official configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
- [Pinned Q8 quant](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/87410e7b9a5c21e6ca70c0c9cc5237de909fe621)
- [Pinned compatible DFlash2 quant](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/tree/144eb06583b4fb0b6e54bc814b16a206a240f4d2)
- [Pinned upstream speculative documentation](https://github.com/ggml-org/llama.cpp/blob/b54752df40e4d8e53d34ef5a598336b4e4871d39/docs/speculative.md)
- [Pinned upstream build documentation](https://github.com/ggml-org/llama.cpp/blob/b54752df40e4d8e53d34ef5a598336b4e4871d39/docs/build.md)
- [Primary-author ROCm research reference](https://github.com/AIwork4me/Qwen3.8-27B-ROCm); its performance claims are not used as measurements of this engine.
