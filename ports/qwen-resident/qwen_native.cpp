// Embedded GGML/llama.cpp backend. No llama-server child, Python, or HTTP bridge.
#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "speculative.h"
#include "stream.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#define QR_API extern "C" __declspec(dllexport)
#else
#define QR_API extern "C" __attribute__((visibility("default")))
#endif

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
using event_callback = int (*)(const char *, void *);
constexpr const char * BACKEND_COMMIT = "b54752df40e4d8e53d34ef5a598336b4e4871d39";
constexpr const char * MODEL_ID = "Qwen/Qwen3.8-27B";
#ifdef _WIN32
static LONG CALLBACK native_exception(EXCEPTION_POINTERS * info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    std::fprintf(stderr, "Qwen native access violation at %p, operation %llu, address %p\n",
        info->ExceptionRecord->ExceptionAddress,
        static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[0]),
        reinterpret_cast<void *>(info->ExceptionRecord->ExceptionInformation[1]));
    void * frames[32];
    const auto count = CaptureStackBackTrace(0, 32, frames, nullptr);
    for (unsigned i = 0; i < count; ++i) {
        HMODULE module = nullptr;
        char path[MAX_PATH]{};
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            static_cast<LPCSTR>(frames[i]), &module);
        GetModuleFileNameA(module, path, MAX_PATH);
        std::fprintf(stderr, "  %p %s + 0x%llx\n", frames[i], path,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) - reinterpret_cast<uintptr_t>(module)));
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif
struct invalid_request : std::runtime_error { using std::runtime_error::runtime_error; };
struct cancelled {};

static double elapsed_ms(clock_type::time_point start, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
static std::string process_env(const char * name) {
#ifdef _WIN32
    size_t size = 0;
    if (getenv_s(&size, nullptr, 0, name) != 0) throw std::runtime_error("Cannot read process environment");
    if (!size) return {};
    std::vector<char> value(size);
    if (getenv_s(&size, value.data(), value.size(), name) != 0) throw std::runtime_error("Cannot read process environment");
    return value.data();
#else
    const char * value = std::getenv(name);
    return value ? value : "";
#endif
}
static void error_out(char * out, size_t size, const char * message) {
    if (out && size) {
        const auto n = std::min(size - 1, std::strlen(message));
        std::memcpy(out, message, n);
        out[n] = 0;
    }
}
static uint64_t available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) throw std::runtime_error("GlobalMemoryStatusEx failed");
    return status.ullAvailPhys;
#else
    return UINT64_MAX;
#endif
}
static void check_headroom() {
    if (available_ram() < (uint64_t(20) << 30)) {
        throw invalid_request("Available physical RAM fell below the 20 GiB operating reserve");
    }
}
static std::string meta(llama_model * model, const char * key) {
    const auto n = llama_model_meta_val_str(model, key, nullptr, 0);
    if (n < 0) return {};
    std::vector<char> value(static_cast<size_t>(n) + 1);
    llama_model_meta_val_str(model, key, value.data(), value.size());
    return value.data();
}

struct engine {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> draft_model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> draft_ctx{nullptr, llama_free};
    // Destruction order matters: the speculative implementation borrows both
    // contexts and must be freed before either context or the shared model.
    common_speculative_ptr spec;
    common_chat_templates_ptr chat;
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch{nullptr, llama_batch_ext_free};
    const llama_vocab * vocab = nullptr;
    std::string identity;
    uint32_t batch_size = 2048;
    int mtp_draft = 0;
    int dflash_draft = 0;
    bool probabilistic_draft = false;
    bool trace_target_logits_enabled = false;
    // Context initialization can perform GPU work. From construction until the
    // final warmup reset, conservatively regard the state as dirty.
    bool state_dirty = true;
    bool always_clear = false;
    // Preallocated once: exception/cancellation cleanup must not allocate.
    std::vector<uint8_t> empty_mtp_state;
    int draft_depth() const { return mtp_draft + dflash_draft; }
    event_callback callback = nullptr;
    void * callback_user = nullptr;

    void trace_target_logits(int row, size_t output_index, llama_token selected,
            const llama_tokens * verification, size_t prefix_size, size_t retained_history_size) const {
        if (!trace_target_logits_enabled || output_index >= 512) return;
        // Diagnostic only: scan existing target outputs after verification,
        // without requesting new output rows or copying retained history.
        const float * logits = llama_get_logits_ith(ctx.get(), row);
        if (!logits) throw std::runtime_error("Missing target logits for requested diagnostic");
        std::vector<std::pair<float, llama_token>> top;
        top.reserve(17);
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
            if (top.size() == 16 && logits[id] <= top.back().first) continue;
            top.emplace_back(logits[id], id);
            std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            if (top.size() > 16) top.resize(16);
        }
        auto candidates = json::array();
        for (const auto & candidate : top) candidates.push_back({candidate.second, candidate.first});
        auto prefix = json::array();
        if (verification) {
            for (size_t i = 0; i < prefix_size; ++i) prefix.push_back((*verification)[i]);
        }
        const auto line = json({{"type", "target_logits_trace"}, {"diagnostic_only", true},
            {"output_index", output_index}, {"target_row", row}, {"selected_token", selected},
            {"selected_raw_logit", logits[selected]}, {"top16_raw_logits", candidates},
            {"verification_prefix", prefix}, {"retained_history_size", retained_history_size}}).dump();
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    }

    bool poll() const { return !callback || callback("{\"type\":\"poll\"}", callback_user) != 0; }
    void emit(const json & event) const {
        const auto value = event.dump(-1, ' ', false, json::error_handler_t::replace);
        if (!callback(value.c_str(), callback_user)) throw cancelled{};
    }
    void decode(const llama_token * tokens, int count, int position, bool all_logits = false) {
        if (!poll()) throw cancelled{};
        common_batch spec_batch;
        if (spec) spec_batch = common_batch(ctx.get());
        llama_batch_ext_clear(batch.get());
        for (int i = 0; i < count; ++i) {
            const int index = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            const llama_pos pos = position + i;
            if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &pos)) {
                throw std::runtime_error("Native batch construction failed");
            }
            if (all_logits) llama_batch_ext_set_output_logits(batch.get(), index, true);
            if (spec) spec_batch.add(tokens[i], pos, 0, all_logits || i == count - 1);
        }
        llama_batch_ext_set_output_logits(batch.get(), count - 1, true);
        // Mark before submission, including submissions that fail or abort.
        // This also covers the subsequent draft catch-up and KV rollback.
        state_dirty = true;
        const int code = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (!poll()) throw cancelled{};
        if (code != 0) throw std::runtime_error("llama_process failed: " + std::to_string(code));
        // The upstream MTP implementation requests every target hidden row,
        // independently of logits selection, and catches up its one-layer KV.
        if (spec && !common_speculative_process(spec.get(), spec_batch)) {
            if (!poll()) throw cancelled{};
            throw std::runtime_error("Speculative prompt/verification catch-up failed");
        }
        if (!poll()) throw cancelled{};
    }
    void reset() {
        if (state_dirty || always_clear) {
            state_dirty = true;
            llama_synchronize(ctx.get());
            if (draft_ctx) llama_synchronize(draft_ctx.get());
            // Retain full physical clearing; this optimization only skips the
            // redundant reset of a state already cleared at the previous exit.
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            if (draft_ctx) llama_memory_clear(llama_get_memory(draft_ctx.get()), true);
            if (spec && mtp_draft) {
                // begin() resets the draft sampler, but not its cross-batch
                // hidden carryover. Clear it even after an aborted prefill.
                common_speculative_set_state(spec.get(), 0, empty_mtp_state);
            }
            // Only a completed reset may mark the state clean.
            state_dirty = false;
        }
        if (spec) common_speculative_get_draft_params(spec.get(), 0) = {};
        callback = nullptr;
        callback_user = nullptr;
    }
};

QR_API void * qr_create(const char * config_json, char * error, size_t error_size) {
    try {
        const auto config = json::parse(config_json);
        const int context = config.value("context", 32768);
        const int batch = config.value("batch", 2048);
        const int ubatch = config.value("ubatch", 512);
        const int threads = config.value("threads", 8);
        const int mtp_draft = config.value("mtp_draft", 0);
        const int dflash_draft = config.value("dflash_draft", 0);
        const auto draft_model_path = config.value("draft_model_path", std::string{});
        if (context < 512 || context > 262144 || batch < 1 || batch > context || ubatch < 1 || ubatch > batch || threads < 1 || threads > 32) {
            throw invalid_request("Invalid context/batch/ubatch/threads; native context is 512..262144");
        }
        if (mtp_draft < 0 || mtp_draft > 3 || (mtp_draft && ubatch < mtp_draft + 1)) {
            throw invalid_request("mtp_draft must be 0..3 and the verification batch must fit ubatch");
        }
        if (dflash_draft < 0 || dflash_draft > 7 || (dflash_draft && ubatch < dflash_draft + 1) ||
            (dflash_draft != 0) != !draft_model_path.empty() || (dflash_draft && mtp_draft)) {
            throw invalid_request("dflash_draft must be 1..7 with draft_model_path, or both disabled; MTP and DFlash are mutually exclusive");
        }
        const int draft_depth = mtp_draft + dflash_draft;
        const auto draft_sampling_env = config.value("draft_sampling", process_env("QWEN_DRAFT_SAMPLING"));
        if (!draft_sampling_env.empty() && draft_sampling_env != "greedy" && draft_sampling_env != "probabilistic") {
            throw invalid_request("draft_sampling must be greedy or probabilistic");
        }
        const bool probabilistic_draft = draft_sampling_env == "probabilistic";
        if (probabilistic_draft && !dflash_draft) {
            throw invalid_request("Probabilistic drafting currently requires DFlash2");
        }
        // Process-local A/B switch for measuring output-buffer sizing using
        // the same binary. It changes capacity, never requested output rows.
        const bool batch_logit_capacity = process_env("QWEN_BATCH_LOGIT_CAPACITY") == "1";
        const bool always_clear = process_env("QWEN_ALWAYS_CLEAR") == "1";
        check_headroom();
        const auto model_path = config.at("model_path").get<std::string>();
        const auto kv_type = config.value("kv_type", std::string("q8_0"));
        if (kv_type != "q8_0" && kv_type != "f16") throw invalid_request("kv_type must be q8_0 or f16");
        // Qwen35 has sixteen full-attention layers, four KV heads of 256
        // elements and separate K/V. Reserve compute/state/host overhead too.
        // Admission is deliberately conservative; runtime counters remain the
        // evidence for resident allocations and real memory pressure.
        const uint64_t kv_bytes = uint64_t(context) * 16 * 4 * 256 * 2 * (kv_type == "q8_0" ? 34 : 64) / 32;
        const uint64_t admitted_bytes = std::filesystem::file_size(std::filesystem::u8path(model_path)) +
            kv_bytes + (mtp_draft ? kv_bytes / 16 : 0) + (uint64_t(28) << 30) +
            (dflash_draft ? std::filesystem::file_size(std::filesystem::u8path(draft_model_path)) + kv_bytes + (uint64_t(2) << 30) : 0);
        if (available_ram() < admitted_bytes) throw invalid_request("Insufficient physical RAM for resident weights, context, runtime overhead and 20 GiB reserve");
        static std::once_flag init;
        std::call_once(init, [] {
#ifdef _WIN32
            // The installed gfx1151 AMD Vulkan driver AVs in pipeline creation
            // for upstream's flipped FA prefill path (both Q8 and F16 KV).
            // Select upstream's normal cooperative-matrix FA implementation.
            // This affects only this process; no driver/registry/PATH change.
            if (_putenv_s("GGML_VK_DISABLE_FA_FLIP", "1") != 0) throw std::runtime_error("Cannot set process-local Vulkan workaround");
            // Per-process opt-in diagnostics; no WER/registry/global changes.
            if (GetEnvironmentVariableA("QWEN_NATIVE_DIAGNOSTICS", nullptr, 0)) AddVectoredExceptionHandler(1, native_exception);
#endif
            llama_log_set([](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_INFO) std::fputs(text, stderr);
            }, nullptr);
            ggml_backend_load_all();
            llama_backend_init();
        });
        ggml_backend_dev_t gpu = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto device = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu = device;
                break;
            }
        }
        if (!gpu) throw std::runtime_error("No compiled GPU backend/device; refusing silent CPU fallback");
        auto instance = std::make_unique<engine>();
        instance->mtp_draft = mtp_draft;
        instance->dflash_draft = dflash_draft;
        instance->probabilistic_draft = probabilistic_draft;
        instance->trace_target_logits_enabled = process_env("QWEN_TRACE_TARGET_LOGITS") == "1";
        instance->always_clear = always_clear;
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
        mp.load_mtp = mtp_draft > 0;
        instance->model.reset(llama_model_load_from_file(model_path.c_str(), mp));
        if (!instance->model) throw std::runtime_error("Failed to load pinned GGUF; inspect native loader log");
        if (meta(instance->model.get(), "general.architecture") != "qwen35") {
            throw std::runtime_error("Expected pinned qwen35-family dense hybrid GGUF");
        }
        if (mtp_draft) {
            instance->empty_mtp_state.assign(size_t(llama_model_n_embd_out(instance->model.get())) * sizeof(float), uint8_t(0));
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = batch;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_rs_seq = draft_depth;
        // Only one final prefill logit or draft+bonus logits are required. Do
        // not reserve batch*vocabulary output rows for every prompt token.
        cp.n_outputs_max = batch_logit_capacity ? batch : draft_depth + 1;
        cp.n_outputs_max_per_seq = cp.n_outputs_max;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = config.value("flash_attention", true) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = kv_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        cp.no_perf = false;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.abort_callback = [](void * data) { return !static_cast<engine *>(data)->poll(); };
        cp.abort_callback_data = instance.get();
        instance->ctx.reset(llama_init_from_model(instance->model.get(), cp));
        if (!instance->ctx) throw std::runtime_error("Failed to allocate Qwen context; inspect native log");
        if (draft_depth) {
            const auto removal = common_context_can_seq_rm(instance->ctx.get());
            if (removal != COMMON_CONTEXT_SEQ_RM_TYPE_RS && removal != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                throw std::runtime_error("Speculation requires bounded recurrent rollback or partial KV removal");
            }
            if (mtp_draft && llama_model_n_layer_nextn(instance->model.get()) < 1) {
                throw std::runtime_error("Requested MTP but GGUF has no loaded MTP head");
            }
            llama_model * model_for_draft = instance->model.get();
            if (dflash_draft) {
                auto dmp = mp;
                dmp.load_mtp = false;
                instance->draft_model.reset(llama_model_load_from_file(draft_model_path.c_str(), dmp));
                model_for_draft = instance->draft_model.get();
                if (!model_for_draft) throw std::runtime_error("Failed to load resident DFlash2 draft GGUF");
                if (llama_model_dflash_variant(model_for_draft) != LLAMA_DFLASH_VARIANT_DFLASH2 ||
                    !common_speculative_are_compatible(instance->model.get(), model_for_draft) ||
                    llama_model_n_embd(model_for_draft) != llama_model_n_embd(instance->model.get())) {
                    throw std::runtime_error("DFlash2 model variant, vocabulary or hidden dimensions do not match target");
                }
                const auto n_extract = llama_model_target_layer_ids_n(model_for_draft);
                const auto * layers = llama_model_target_layer_ids(model_for_draft);
                if (!n_extract || !layers) throw std::runtime_error("DFlash2 target layer metadata is missing");
                for (uint32_t i = 0; i < n_extract; ++i) {
                    if (layers[i] < 0 || layers[i] >= llama_model_n_layer(instance->model.get())) {
                        throw std::runtime_error("DFlash2 requested an out-of-range target hidden layer");
                    }
                }
            }
            auto dp = cp;
            dp.ctx_type = mtp_draft ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
            dp.ctx_other = instance->ctx.get();
            dp.n_rs_seq = 0;
            instance->draft_ctx.reset(llama_init_from_model(model_for_draft, dp));
            if (!instance->draft_ctx) throw std::runtime_error("Failed to allocate speculative draft context");
            common_params_speculative sp;
            sp.types = {mtp_draft ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH2};
            sp.draft.ctx_tgt = instance->ctx.get();
            sp.draft.ctx_dft = instance->draft_ctx.get();
            sp.draft.n_max = draft_depth;
            sp.draft.n_min = 0;
            sp.draft.p_min = 0;
            sp.draft.probabilistic = probabilistic_draft;
            sp.draft.cache_type_k = cp.type_k;
            sp.draft.cache_type_v = cp.type_v;
            instance->spec.reset(common_speculative_init(sp, 1));
            if (!instance->spec) throw std::runtime_error("Upstream speculative initialization failed");
            if (common_speculative_n_max(instance->spec.get()) != draft_depth) {
                throw std::runtime_error("Requested speculative depth exceeds the draft's trained block size");
            }
            if (common_context_can_seq_rm(instance->draft_ctx.get()) != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                throw std::runtime_error("This draft context does not support partial draft KV rollback");
            }
        }
        instance->vocab = llama_model_get_vocab(instance->model.get());
        instance->chat = common_chat_templates_init(instance->model.get(), "");
        instance->batch.reset(llama_batch_ext_init(instance->ctx.get()));
        if (!instance->chat || !instance->batch) throw std::runtime_error("Native chat/batch initialization failed");
        instance->batch_size = batch;
        // Exercise batched prefill and decode before readiness. This populates
        // runtime allocations and compiles the actual device pipelines, then
        // clears the synthetic sequence. It is not model-quality acceptance.
        const auto warmup_start = clock_type::now();
        auto warmup = common_tokenize(instance->vocab, "Warmup", true, false);
        if (warmup.empty()) throw std::runtime_error("Warmup tokenization failed");
        warmup.resize(std::min(128, batch), warmup.front());
        instance->decode(warmup.data(), static_cast<int>(warmup.size()), 0);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> warm_sampler(llama_sampler_init_greedy(), llama_sampler_free);
        const auto warm_token = llama_sampler_sample(warm_sampler.get(), instance->ctx.get(), -1);
        llama_tokens warm_verify{warm_token};
        if (instance->spec) {
            common_speculative_begin(instance->spec.get(), 0, warmup);
            llama_tokens warm_draft;
            auto & dp = common_speculative_get_draft_params(instance->spec.get(), 0);
            dp = {};
            dp.drafting = true;
            dp.n_max = draft_depth;
            dp.pos0 = static_cast<int>(warmup.size());
            dp.id_last = warm_token;
            dp.prompt = &warmup;
            dp.result = &warm_draft;
            instance->state_dirty = true;
            common_speculative_draft(instance->spec.get());
            warm_verify.insert(warm_verify.end(), warm_draft.begin(), warm_draft.end());
            if (!llama_memory_seq_rm(llama_get_memory(instance->draft_ctx.get()), 0, dp.pos0, -1)) {
                throw std::runtime_error("Speculative warmup rollback failed");
            }
        }
        instance->decode(warm_verify.data(), static_cast<int>(warm_verify.size()), static_cast<int>(warmup.size()), instance->spec != nullptr);
        instance->reset();
        const auto warmup_ms = elapsed_ms(warmup_start, clock_type::now());
        check_headroom();
        size_t gpu_free = 0, gpu_total = 0;
        ggml_backend_dev_memory(gpu, &gpu_free, &gpu_total);
        const auto file_type = meta(instance->model.get(), "general.file_type");
        const auto quantization = file_type == std::to_string(LLAMA_FTYPE_MOSTLY_Q8_0) ? "Q8_0" :
            file_type == std::to_string(LLAMA_FTYPE_MOSTLY_Q4_K_M) ? "Q4_K_M" : "unknown";
        instance->identity = json({
            {"model_id", MODEL_ID}, {"backend", "embedded-llama.cpp-ggml"}, {"backend_commit", BACKEND_COMMIT},
            {"device", ggml_backend_dev_description(gpu)}, {"device_backend", ggml_backend_dev_name(gpu)},
            {"architecture", meta(instance->model.get(), "general.architecture")},
            {"quantization", quantization}, {"gguf_file_type", file_type}, {"model_parameters", llama_model_n_params(instance->model.get())},
            {"model_tensor_bytes", llama_model_size(instance->model.get())},
            {"context", llama_n_ctx(instance->ctx.get())}, {"batch", batch}, {"ubatch", ubatch}, {"threads", threads},
            {"native_context_limit", 262144}, {"context_quality_validated_by_boot", false},
            {"kv_type", kv_type}, {"flash_attention", config.value("flash_attention", true)},
            {"vulkan_fa_flip_disabled", true},
            {"startup_warmup_tokens", warmup.size()}, {"startup_warmup_ms", warmup_ms},
            {"gpu_layers_requested", 999}, {"mmap", false}, {"lazy_weights", false}, {"mtp", mtp_draft > 0},
            {"mtp_draft", mtp_draft}, {"recurrent_snapshots", cp.n_rs_seq}, {"max_logit_outputs", cp.n_outputs_max},
            {"batch_logit_capacity_experiment", batch_logit_capacity},
            {"always_clear_experiment", always_clear}, {"single_clear_request_state", !always_clear},
            {"dflash_draft", dflash_draft}, {"speculative_type", mtp_draft ? "draft-mtp" : dflash_draft ? "draft-dflash2" : "none"},
            {"draft_model_path", draft_model_path},
            {"draft_model_tensor_bytes", instance->draft_model ? llama_model_size(instance->draft_model.get()) : 0},
            {"draft_quantization", instance->draft_model ?
                (llama_model_ftype(instance->draft_model.get()) == LLAMA_FTYPE_MOSTLY_Q8_0 ? "Q8_0" :
                 llama_model_ftype(instance->draft_model.get()) == LLAMA_FTYPE_MOSTLY_Q4_K_M ? "Q4_K_M" : "unknown") :
                mtp_draft ? "shared-target" : "none"},
            {"draft_sampling", probabilistic_draft ? "probabilistic" : "greedy"},
            {"target_logits_trace", instance->trace_target_logits_enabled},
            {"speculative_verifier", probabilistic_draft ? "upstream-rejection-if-temperature-positive-otherwise-match" :
                draft_depth ? "upstream-target-sample-and-match" : "none"},
            {"thinking_supported", common_chat_templates_support_enable_thinking(instance->chat.get())},
            {"load_available_physical_bytes", available_ram()}, {"gpu_budget_free_bytes", gpu_free}, {"gpu_budget_total_bytes", gpu_total}
        }).dump();
        return instance.release();
    } catch (const std::exception & e) { error_out(error, error_size, e.what()); return nullptr; }
    catch (...) { error_out(error, error_size, "Unexpected native initialization failure"); return nullptr; }
}

QR_API const char * qr_identity(void * handle) { return handle ? static_cast<engine *>(handle)->identity.c_str() : "{}"; }
QR_API void qr_destroy(void * handle) { delete static_cast<engine *>(handle); }

QR_API int qr_generate(void * handle, const char * request_json, event_callback callback, void * user, char * error, size_t error_size) {
    if (!handle || !callback) { error_out(error, error_size, "Invalid native handle/callback"); return -1; }
    auto & e = *static_cast<engine *>(handle);
    const auto request_entered = clock_type::now();
    // This C ABI is single-owner. Rust serializes requests on its native worker.
    struct reset_guard { engine & e; ~reset_guard() { if (e.callback) e.reset(); } } guard{e};
    try {
        const auto reset_before_start = clock_type::now();
        e.reset();
        const double reset_before_ms = elapsed_ms(reset_before_start, clock_type::now());
        e.callback = callback;
        e.callback_user = user;
        check_headroom();
        const auto start = clock_type::now();
        const auto request = json::parse(request_json);
        const int max_tokens = request.value("max_tokens", 512);
        const float temperature = request.value("temperature", 0.7f);
        const float top_p = request.value("top_p", 0.8f);
        const float min_p = request.value("min_p", 0.0f);
        const int top_k = request.value("top_k", 20);
        const float repeat_penalty = request.value("repeat_penalty", 1.0f);
        const float presence_penalty = request.value("presence_penalty", 0.0f);
        const float frequency_penalty = request.value("frequency_penalty", 0.0f);
        const uint32_t seed = request.value("seed", LLAMA_DEFAULT_SEED);
        const bool use_rejection = e.probabilistic_draft && temperature > 0;
        if (max_tokens < 1 || max_tokens > 8192 || !std::isfinite(temperature) || temperature < 0 || temperature > 2 ||
            !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || !std::isfinite(min_p) || min_p < 0 || min_p > 1 ||
            top_k < 0 || top_k > 1000 || !std::isfinite(repeat_penalty) || repeat_penalty <= 0 || repeat_penalty > 2 ||
            !std::isfinite(presence_penalty) || presence_penalty < -2 || presence_penalty > 2 ||
            !std::isfinite(frequency_penalty) || frequency_penalty < -2 || frequency_penalty > 2) {
            throw invalid_request("Invalid generation/sampling bounds");
        }
        std::vector<std::string> stops = request.value("stop", std::vector<std::string>{});
        for (const auto & stop : stops) if (stop.empty()) throw invalid_request("Stop sequences must not be empty");
        std::vector<llama_token> tokens;
        std::string thinking_prefix;
        if (request.contains("messages")) {
            common_chat_templates_inputs inputs;
            inputs.enable_thinking = request.value("enable_thinking", false);
            inputs.chat_template_kwargs["enable_thinking"] = inputs.enable_thinking ? "true" : "false";
            if (inputs.enable_thinking && !common_chat_templates_support_enable_thinking(e.chat.get())) {
                throw invalid_request("Model chat template does not support enable_thinking");
            }
            for (const auto & msg : request.at("messages")) {
                common_chat_msg value;
                value.role = msg.at("role").get<std::string>();
                value.content = msg.at("content").get<std::string>();
                inputs.messages.push_back(std::move(value));
            }
            const auto formatted = common_chat_templates_apply(e.chat.get(), inputs);
            // Preserve upstream separation of user content and control tokens.
            for (const auto & part : formatted.prompt.parts) {
                auto piece = common_tokenize(e.vocab, part.text, tokens.empty(), !part.is_input);
                tokens.insert(tokens.end(), piece.begin(), piece.end());
            }
            stops.insert(stops.end(), formatted.additional_stops.begin(), formatted.additional_stops.end());
            if (inputs.enable_thinking && formatted.generation_prompt.find("<think>") != std::string::npos &&
                formatted.generation_prompt.find("</think>") == std::string::npos) thinking_prefix = "<think>\n";
        } else {
            tokens = common_tokenize(e.vocab, request.at("prompt").get<std::string>(), true, false);
        }
        if (tokens.empty()) throw invalid_request("Prompt tokenized to an empty sequence");
        if (tokens.size() + static_cast<size_t>(max_tokens) > llama_n_ctx(e.ctx.get())) {
            throw invalid_request("Prompt tokens (" + std::to_string(tokens.size()) + ") plus max_tokens exceed context (" + std::to_string(llama_n_ctx(e.ctx.get())) + ")");
        }
        common_params_sampling sampling;
        sampling.seed = seed;
        sampling.temp = temperature;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.min_p = min_p;
        sampling.min_keep = 1;
        sampling.penalty_last_n = max_tokens;
        sampling.penalty_repeat = repeat_penalty;
        sampling.penalty_freq = frequency_penalty;
        sampling.penalty_present = presence_penalty;
        sampling.backend_sampling = false;
        sampling.samplers.clear();
        // OpenAI-style presence/frequency penalties apply to generated history,
        // not input tokens. Keep the whole bounded generation, not only 64 tokens.
        if (repeat_penalty != 1.0f || presence_penalty != 0 || frequency_penalty != 0) {
            sampling.samplers.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
        }
        if (temperature != 0) {
            sampling.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K);
            sampling.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_P);
            sampling.samplers.push_back(COMMON_SAMPLER_TYPE_MIN_P);
        }
        sampling.samplers.push_back(COMMON_SAMPLER_TYPE_TEMPERATURE);
        common_sampler_ptr sampler(common_sampler_init(e.model.get(), sampling));
        if (!sampler) throw std::runtime_error("Target sampler initialization failed");
        llama_perf_context_reset(e.ctx.get());
        const auto prefill_start = clock_type::now();
        size_t next_prefill_progress = 8192;
        for (size_t pos = 0; pos < tokens.size(); pos += e.batch_size) {
            check_headroom();
            const auto count = static_cast<int>(std::min<size_t>(e.batch_size, tokens.size() - pos));
            e.decode(tokens.data() + pos, count, static_cast<int>(pos));
            const size_t completed = pos + static_cast<size_t>(count);
            if (tokens.size() >= 32768 && (completed >= next_prefill_progress || completed == tokens.size())) {
                // A progress count means completed GPU work, not merely queued
                // submission. Bound this sync/report to each 8192-token milestone.
                llama_synchronize(e.ctx.get());
                if (e.draft_ctx) llama_synchronize(e.draft_ctx.get());
                const auto progress = json({{"type", "prefill_progress"}, {"completed_tokens", completed},
                    {"total_tokens", tokens.size()}, {"elapsed_ms", elapsed_ms(prefill_start, clock_type::now())},
                    {"available_physical_bytes", available_ram()}}).dump();
                std::fprintf(stderr, "%s\n", progress.c_str());
                std::fflush(stderr);
                next_prefill_progress = (completed / 8192 + 1) * 8192;
            }
        }
        llama_synchronize(e.ctx.get());
        if (e.spec) common_speculative_begin(e.spec.get(), 0, tokens);
        const auto prefill_end = clock_type::now();
        text_stream stream(std::move(stops));
        std::string output;
        auto emit_text = [&](const std::string & text) {
            if (text.empty()) return;
            output += text;
            e.emit({{"type", "delta"}, {"text", text}});
        };
        int completion_tokens = 0, decode_tokens = 0;
        int drafted_tokens = 0, accepted_draft_tokens = 0, verification_steps = 0, target_decode_calls = 0;
        int rejection_verification_steps = 0;
        double draft_ms = 0, target_decode_ms = 0, sampling_ms = 0;
        std::string finish_reason = "length";
        auto first_token = prefill_end;
        if (!thinking_prefix.empty()) emit_text(stream.push(thinking_prefix));
        auto sample_start = clock_type::now();
        auto current = common_sampler_sample(sampler.get(), e.ctx.get(), -1);
        common_sampler_accept(sampler.get(), current, true);
        sampling_ms += elapsed_ms(sample_start, clock_type::now());
        first_token = clock_type::now();
        e.trace_target_logits(-1, 0, current, nullptr, 0, tokens.size());
        // history contains only tokens actually evaluated and retained in KV;
        // current is a target-sampled token still awaiting its decode.
        llama_tokens history = tokens;
        history.reserve(tokens.size() + max_tokens);
        auto consume = [&](llama_token token) {
            if (!e.poll()) throw cancelled{};
            ++completion_tokens;
            if (llama_vocab_is_eog(e.vocab, token)) { finish_reason = "stop"; return false; }
            emit_text(stream.push(common_token_to_piece(e.vocab, token, true)));
            if (stream.stopped) { finish_reason = "stop"; return false; }
            return completion_tokens < max_tokens;
        };
        bool continuing = consume(current);
        while (continuing) {
            if (!e.poll()) throw cancelled{};
            check_headroom();
            const int position = static_cast<int>(history.size());
            const int remaining = max_tokens - completion_tokens;
            llama_tokens draft;
            std::vector<std::vector<llama_token_data>> draft_q;
            // Use full bounded drafts only. The final few tokens use ordinary
            // decode so proposals never run beyond the request's output budget.
            if (e.spec && remaining > e.draft_depth()) {
                auto & dp = common_speculative_get_draft_params(e.spec.get(), 0);
                dp = {};
                dp.drafting = true;
                dp.n_max = e.draft_depth();
                dp.pos0 = position;
                dp.id_last = current;
                dp.prompt = &history;
                dp.result = &draft;
                // Match upstream server flow: temp 0 stays on argmax drafting
                // and target sample-and-match. Positive temperatures provide
                // the drafter's actual distributions to the exact verifier.
                dp.result_q = use_rejection ? &draft_q : nullptr;
                dp.temp = temperature;
                dp.seed = seed;
                const auto draft_start = clock_type::now();
                e.state_dirty = true;
                common_speculative_draft(e.spec.get());
                draft_ms += elapsed_ms(draft_start, clock_type::now());
                if (!e.poll()) throw cancelled{};
                check_headroom();
                if ((use_rejection && draft_q.size() != draft.size()) || (!use_rejection && !draft_q.empty())) {
                    throw std::runtime_error("Draft probability rows do not match drafted tokens");
                }
                for (const auto & row : draft_q) {
                    if (row.empty()) throw std::runtime_error("Draft probability row is empty");
                }
                // Upstream server flow discards speculative draft KV before
                // catching it up from the target's actual verification rows.
                if (!llama_memory_seq_rm(llama_get_memory(e.draft_ctx.get()), 0, position, -1)) {
                    throw std::runtime_error("Speculative draft KV rollback failed");
                }
            }
            llama_tokens verification{current};
            verification.insert(verification.end(), draft.begin(), draft.end());
            const auto decode_start = clock_type::now();
            e.decode(verification.data(), static_cast<int>(verification.size()), position, !draft.empty());
            target_decode_ms += elapsed_ms(decode_start, clock_type::now());
            ++target_decode_calls;
            sample_start = clock_type::now();
            // Both pinned upstream verifiers preserve the target sampling
            // chain and accepted history. Rejection needs the exact q used to
            // sample each draft; it accepts min(1,p/q), then samples (p-q)+ on
            // rejection. No threshold-based token acceptance is used.
            std::vector<llama_token> accepted;
            if (use_rejection && !draft.empty()) {
                std::vector<int> indices(draft.size() + 1);
                for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);
                accepted = common_sampler_sample_and_accept_n_rejection(sampler.get(), e.ctx.get(), indices, draft, draft_q);
                ++rejection_verification_steps;
            } else {
                accepted = common_sampler_sample_and_accept_n(sampler.get(), e.ctx.get(), draft);
            }
            sampling_ms += elapsed_ms(sample_start, clock_type::now());
            if (accepted.empty() || accepted.size() > verification.size()) {
                throw std::runtime_error("Invalid speculative verification result");
            }
            if (e.trace_target_logits_enabled) {
                for (size_t row = 0; row < accepted.size(); ++row) {
                    e.trace_target_logits(static_cast<int>(row), history.size() - tokens.size() + row + 1,
                        accepted[row], &verification, row + 1, history.size());
                }
            }
            if (!draft.empty()) {
                ++verification_steps;
                drafted_tokens += static_cast<int>(draft.size());
                common_speculative_accept(e.spec.get(), 0, static_cast<uint16_t>(accepted.size() - 1));
                const int keep = position + static_cast<int>(accepted.size());
                if (!llama_memory_seq_rm(llama_get_memory(e.ctx.get()), 0, keep, -1) ||
                    !llama_memory_seq_rm(llama_get_memory(e.draft_ctx.get()), 0, keep, -1)) {
                    throw std::runtime_error("Speculative accepted-prefix rollback failed");
                }
            }
            history.push_back(current);
            history.insert(history.end(), accepted.begin(), accepted.end() - 1);
            current = accepted.back();
            for (size_t i = 0; i < accepted.size(); ++i) {
                ++decode_tokens;
                if (i < accepted.size() - 1) ++accepted_draft_tokens;
                continuing = consume(accepted[i]);
                if (!continuing) break;
            }
        }
        emit_text(stream.finish());
        const auto end = clock_type::now();
        const double prefill_ms = elapsed_ms(prefill_start, prefill_end);
        const double decode_ms = elapsed_ms(first_token, end);
        const auto available_physical_bytes = available_ram();
        // Preserve prefill/decode/total_ms definitions, while separately
        // measuring the full reset before publishing the terminal callback.
        const auto cleanup_start = clock_type::now();
        e.reset();
        const auto cleanup_end = clock_type::now();
        const json result = {
            {"type", "finished"}, {"text", output}, {"finish_reason", finish_reason},
            {"prompt_tokens", tokens.size()}, {"completion_tokens", completion_tokens},
            {"metrics", {{"prefill_ms", prefill_ms}, {"decode_ms", decode_ms}, {"ttft_ms", elapsed_ms(start, first_token)},
                {"prefill_tps", tokens.size() * 1000.0 / std::max(prefill_ms, 0.001)},
                {"decode_tps", decode_tokens * 1000.0 / std::max(decode_ms, 0.001)}, {"decode_tokens", decode_tokens},
                {"mtp_draft", e.mtp_draft}, {"dflash_draft", e.dflash_draft}, {"drafted_tokens", drafted_tokens}, {"accepted_draft_tokens", accepted_draft_tokens},
                {"draft_acceptance", drafted_tokens ? double(accepted_draft_tokens) / drafted_tokens : 0.0},
                {"verification_steps", verification_steps}, {"target_decode_calls", target_decode_calls},
                {"draft_sampling", use_rejection ? "probabilistic" : "greedy"},
                {"speculative_verifier", !e.spec ? "none" : use_rejection ? "upstream-target-rejection" : "upstream-target-sample-and-match"},
                {"rejection_verification_steps", rejection_verification_steps},
                {"target_logits_trace", e.trace_target_logits_enabled},
                {"benchmark_timing_valid", !e.trace_target_logits_enabled},
                {"draft_ms", draft_ms}, {"target_submit_and_catchup_ms", target_decode_ms}, {"sampling_and_sync_ms", sampling_ms},
                {"total_ms", elapsed_ms(start, end)}, {"reset_before_ms", reset_before_ms},
                {"cleanup_ms", elapsed_ms(cleanup_start, cleanup_end)},
                {"total_including_cleanup_ms", elapsed_ms(request_entered, cleanup_end)},
                {"available_physical_bytes", available_physical_bytes}, {"seed", seed}}}
        };
        // State was fully reset before this terminal callback.
        const auto serialized = result.dump(-1, ' ', false, json::error_handler_t::replace);
        if (!callback(serialized.c_str(), user)) return 1;
        return 0;
    } catch (const cancelled &) { return 1; }
    catch (const invalid_request & ex) { error_out(error, error_size, ex.what()); return 2; }
    catch (const json::exception & ex) { error_out(error, error_size, ex.what()); return 2; }
    catch (const std::exception & ex) { error_out(error, error_size, ex.what()); return -1; }
    catch (...) { error_out(error, error_size, "Unexpected native generation failure"); return -1; }
}
