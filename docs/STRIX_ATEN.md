# Optional native ATen/HIP compatibility

This Windows-only experiment uses Rust → a small C ABI DLL → the existing native
ATen/HIP libraries. It does not start, import, link, or communicate with Python.
The separate Python validation scripts are independent oracles. The existing
`strix generate-experimental` command still selects Vulkan and the Python API is
unchanged. No native HTTP API is enabled.

The backend is pinned to the original Flash checkpoint/revision, Radeon 8060S,
PyTorch 2.11.0 native ABI and HIP runtime 71460850. Ordinary Cargo builds/tests do
not require ATen. Build and selection are both explicit:

```powershell
$site = 'C:/Users/ncrawford/.unsloth/studio/unsloth_studio/Lib/site-packages'
pwsh -NoProfile -File crates/strix/native/build-aten.ps1 -SitePackages $site -Output target/strix-aten-build
cargo build -p deepseek-ultrafast-strix --release --features native-aten --examples
$env:STRIX_ATEN_SITE_PACKAGES = $site
$env:STRIX_ATEN_SHIM = (Resolve-Path target/strix-aten-build/strix_aten.dll).Path
target/release/examples/aten_probe.exe target/models/DeepSeek-V4.1-Flash target/strix-components-final/requests.json target/new-aten-components
target/release/examples/aten_teacher_force.exe target/models/DeepSeek-V4.1-Flash target/strix-numerics-20261114/cases.json target/new-aten-suite aten-ops
```

Output directories must be new. Set `STRIX_ATEN_TRACE=1` to save layer traces.
`aten-ops` selects native ATen for every learned projection, HC, RMS, attention,
Engram gates, routing, pooling, SwiGLU and complex RoPE. It does not initialize Vulkan or fall
back to CPU model execution. `hc-vulkan` is an explicit diagnostic mode: HC/RMS/
pre/post use ATen while other operations retain Vulkan/scalar scheduling.
Tokenization, hash calculation, bounded checkpoint reads, activation quantization,
bounded index selection, accumulation and KV scheduling still use Rust.
Constant rotary tables are constructed with native CPU ATen operations and then
uploaded, matching the unchanged reference's buffer initialization. This does not
perform learned matrix arithmetic on CPU; all learned projections remain on HIP.
Activations/KV cross the host boundary; this is not a fully resident ATen graph.

The build script uses the bundled clang-cl and VS2022 Build Tools environment in
a child process. It links only `torch_cpu.lib` and `c10.lib`, records input hashes
and versions in `build-manifest.json`, and installs nothing. The wheel, drivers,
OS, global PATH, original sources and model are not modified.

## Loading and resource ownership

Rust has no static ATen DLL imports. Before loading the shim it checks already
loaded modules for conflicting SDK paths, adds three process-local DLL search
directories, explicitly loads the bundled `amdhip64_7.dll`, then `torch_hip.dll`
and the shim by absolute path. Runtime ABI/version/device checks fail closed.
The completed process inventory is checked again for late-loaded conflicts and
Python/torch_python modules. No System32 HIP fallback is permitted.

The successful bundled HIP file version is `10.0.3581.0`; the feasibility probe
had crashed with System32 `10.0.3683.0`. Another preserved failed attempt exposed
rocBLAS's rejection of Windows verbatim `\\?\` paths. The production loader
canonicalizes and validates paths, then uses ordinary absolute local-drive paths.

Tensor handles and DLL search cookies use RAII. All exported operations catch
C++ exceptions and return bounded caller-owned error strings; failed outputs are
null. Handles retain their owning API. Registered ATen DLLs remain loaded until
process exit because their dispatch callbacks are process-global. Dropping a
tensor releases its active allocation; ATen may retain allocator reservations.
A failed decoder forward releases its cached weights and request state and
rejects reuse of partial KV state. Reopening is explicit; there is no fallback.

## Bounded weights and arithmetic

The existing Rust safetensors catalog, checked shape validation and bounded
seek/read reader supply original bytes. Quantized GPU cache entries retain packed
FP4/FP8 weights plus scales. Native ATen dequantizes full-K slabs and uses the
reference's 256-row GEMM dispatch and BF16 staging. Dense converted FP32 weights
are counted at their actual converted size. HC weights stay resident separately,
with 256 MiB reserved inside the overall eight-GiB weight budget.

The head is assembled on HIP from bounded BF16 reads and promoted to FP32. Its
linear call uses the complete original `[129280,5120]` matrix. Splitting that call
changed FP32 reductions even with identical head inputs. The temporary upload
chunks are not retained as duplicate cache entries.

Reports distinguish packed/converted resident weights, HC resident bytes, CPU
staging/read bounds, and native allocator current/peak allocated/reserved bytes.
The weight budget does not cap allocator reservations, BLAS workspace, transient
dequantization/index tensors, activations, host KV, tokenizer or other processes.
The largest host checkpoint read is 20 MiB; head upload staging is at most 60 MiB.
The final run measured peak native allocator usage of 8,904,337,408 allocated
bytes and 11,834,228,736 reserved bytes. Its completed inventory contains 76
modules, with no Python or `torch_python` module.

## Validation

New [machine-readable evidence](validation/strix-halo-aten-2026-11-14.json) is
recorded separately from the historical Vulkan failures. The HC-only path matched
60 original-input component probes exactly but still predicted `201` for `The`
instead of HIP `455`. Extending the operations
matched 282 original-input probes exactly. All 40 recorded block traces and the
head input for `The` then matched; the initially slabbed head still differed in
108,620 logits (maximum `3.814697265625e-5`). Restoring the original head shape
made all 129,280 `The` logits bit-identical.

The intermediate, pre-RoPE-fix 28-position suite matched **27 next-token predictions**; **16
positions have bit-identical full logits**. The remaining greedy mismatch is
chat position 3 (`260` versus saved HIP `9045`). France's `[11111,16,1]` and chat's
`[11932,1]` continuations match. Full numerical parity is **not accepted**.

A fresh unchanged HIP chat trace matches native through layer 19 and at layer
20's FFN input. Its first difference is in that FFN. Identical-input replay
isolates an exact top-k cutoff tie: experts 96 and 338 both have corrected score
`14.184751510620117`. All 384 raw logits, softplus scores and corrected scores
match. In 100 repeats, HIP selected expert 96 60 times and 338 40 times; native
selected them 65 and 35 times. Their unbiased routing weights differ, so choosing
between them changes the subsequent graph. A fresh HIP full result also differs
from its saved golden result. PyTorch explicitly documents that
[top-k tie indices can vary between invocations](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html).
No tie-breaking policy, score perturbation, tolerance, or oracle change is used.
This identifies the first isolated chat divergence; it does not establish that
every remaining difference has the same cause.

A separate original France position-5 trace identified a deterministic RoPE
boundary: layer 23's attention input matched, but scalar Rust rotation differed
from HIP complex multiplication in one of 4,096 BF16 query components (absolute
difference `0.00048828125`), even with identical supplied frequencies. Native ATen
complex multiplication and the reference-equivalent frequency factory match all
4,096 components exactly. This is integrated into `aten-ops`; the earlier failed
full-model artifacts are preserved and the final suite is recorded separately.

With that correction, the fresh 28-position suite matches **28/28 next-token
predictions** and **18/28 complete logit vectors** exactly: `The`, all seven France
positions, five arithmetic positions, four story positions and the greeting.
The ten serialized-chat positions still differ numerically (maximum absolute
logit difference `0.325541019439697`), despite matching every next token in this
run. Full numerical parity remains **unaccepted**. The identified top-k tie can
choose different experts on repeated identical calls; this single run does not
establish deterministic greedy agreement for future runs.

The delivery build also passes **312 original-input component requests** exactly,
including attention, and a fresh complete `The` vocabulary comparison. The
62-position runner was built before the final integer-parameter FFI guards and
validation-runner input/reporting checks; tensor equations and scheduling are
identical. Executable and shim hashes distinguish these runs. The final guards
are covered by native error/lifecycle tests and the delivery component replay.

The fresh **34-position geography prefix matches every next token and all
129,280 logits at every position exactly**, including repeated compression and
shared KV reuse beyond the previous ten-position limit. Across the two suites,
62/62 next-token predictions and 52/62 complete logit vectors match. This does
not exercise the 128-position window boundary or candidate pruning.

These measurements do not promote this backend to the serving default or
establish arbitrary-prompt/128-position equivalence. The Vulkan `The` failure and
its 23/28 next-token result remain unchanged historical facts.

### Conditional routing validation

A subsequent paired run captures the unchanged HIP reference's ten-position
chat sequence, then checks every native router's raw logits, unbiased scores,
corrected scores and weights bit for bit. The native diagnostic resamples the
same ATen `topk` operation only when exact tied scores permit another legal
sorted selection. It never supplies recorded weights or perturbs scores.
Non-tied differences, nonfinite values, invalid/duplicate expert IDs or unequal
weights fail immediately. This diagnostic is enabled only by the standalone
`aten_route_replay` example; the serving API cannot enable it.

**All 400 router checks and all 1,292,800 output logits match bit for bit.**
Two selections needed resampling, both at position zero:

- Layer 20: the first native selection chose expert 96; the reference chose
  338. Their corrected scores both equal `14.184751510620117`. The second
  unchanged native call selected the recorded branch.
- Layer 36: the first native selection chose 337; the reference chose 186.
  Experts 186, 239, 270 and 337 share cutoff `32.76362991333008`. The third
  unchanged native call selected the recorded branch.

The resulting chat continuation remains `OK` followed by EOS. This establishes
conditional fidelity for the tested sequence and accounts for its observed
divergence using permitted tied selections. It does **not** establish
unconditional bit-identical production output, every historical mismatch,
arbitrary prompt quality or the complete 128-position boundary. Production
continues to use one ordinary ATen top-k call per router, with its documented
tie variability. The earlier saved-reference failures remain recorded.

See [the routing evidence](validation/strix-halo-routing-2026-11-14.json) and
the separate [native API guide](STRIX_NATIVE_API.md) for serving and its own
hardware acceptance. The comparison enforces exact unique coverage of all
40 layers at each position and compares FP32 bit patterns, including signed
zero. Reference capture and native replay took 249.6 and 286.0 seconds,
respectively; these are diagnostic timings, not serving benchmarks.

```powershell
& $python ports/strix-reference/validate_routing_replay.py capture --checkpoint target/models/DeepSeek-V4.1-Flash --cases target/strix-numerics-20261114/cases.json --label chat --output target/new-routing-reference
target/release/examples/aten_route_replay.exe target/models/DeepSeek-V4.1-Flash target/new-routing-reference target/new-routing-native
& $python ports/strix-reference/validate_routing_replay.py compare --reference target/new-routing-reference --native target/new-routing-native --executable target/release/examples/aten_route_replay.exe --shim $env:STRIX_ATEN_SHIM --output target/new-routing-comparison.json
```

```powershell
cargo test -p deepseek-ultrafast-strix
cargo test -p deepseek-ultrafast-strix --lib -- --ignored
cargo test -p deepseek-ultrafast-strix --features native-aten
cargo test -p deepseek-ultrafast-strix --features native-aten --lib aten::tests::hip_ffi_lifecycle_and_errors -- --ignored
cargo clippy -p deepseek-ultrafast-strix --all-targets -- -D warnings
cargo clippy -p deepseek-ultrafast-strix --features native-aten --all-targets -- -D warnings
cargo fmt -p deepseek-ultrafast-strix -- --check
$python = 'C:/Users/ncrawford/.unsloth/studio/unsloth_studio/Scripts/python.exe'
& $python -m unittest discover -s ports/strix-reference/tests -v
& $python ports/strix-reference/validate_aten.py compare --checkpoint target/models/DeepSeek-V4.1-Flash --reference target/strix-numerics-20261114 --native target/new-aten-suite --identity target/strix-components-final/identity.json --executable target/release/examples/aten_teacher_force.exe --shim $env:STRIX_ATEN_SHIM --output target/new-aten-comparison.json
& $python ports/strix-reference/validate_aten.py compare-components --checkpoint target/models/DeepSeek-V4.1-Flash --reference target/strix-components-final --native target/new-aten-components --executable target/release/examples/aten_probe.exe --shim $env:STRIX_ATEN_SHIM --output target/new-aten-component-comparison.json
& $python ports/strix-reference/validate_aten.py long-reference --checkpoint target/models/DeepSeek-V4.1-Flash --output target/new-aten-long-reference
& $python ports/strix-reference/validate_aten_router.py --checkpoint target/models/DeepSeek-V4.1-Flash --components target/strix-aten-chat-l20-components --label l20-ffn-p0/layers.20.ffn.gate --probe target/release/examples/aten_probe.exe --output target/new-router-tie-evidence
```

The comparison verifies original provenance hashes, exact input/tokenizer IDs,
saved greedy IDs and full vocabulary lengths. It reconstructs FP32 JSON values,
computes errors in FP64, and records executable/shim/source/golden hashes. It does
not widen a tolerance. The longer reference uses unchanged official model/kernel
math, one token per forward and a 34-position natural-language prompt.

Primary API references checked before implementation:
[PyTorch 2.11 tensor ownership](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/cpp/source/notes/tensor_basics.md),
[C++ installation and Windows ABI](https://docs.pytorch.org/cppdocs/installing.html),
[HIP device semantics](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/hip.rst).
Installed ATen and caching-allocator headers were also inspected directly.
