// Embedded GGML/llama.cpp backend. No llama-server child, Python, or HTTP bridge.
#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "stream.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    common_chat_templates_ptr chat;
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch{nullptr, llama_batch_ext_free};
    const llama_vocab * vocab = nullptr;
    std::string identity;
    uint32_t batch_size = 2048;
    event_callback callback = nullptr;
    void * callback_user = nullptr;

    bool poll() const { return !callback || callback("{\"type\":\"poll\"}", callback_user) != 0; }
    void emit(const json & event) const {
        const auto value = event.dump(-1, ' ', false, json::error_handler_t::replace);
        if (!callback(value.c_str(), callback_user)) throw cancelled{};
    }
    void decode(const llama_token * tokens, int count, int position) {
        if (!poll()) throw cancelled{};
        llama_batch_ext_clear(batch.get());
        for (int i = 0; i < count; ++i) {
            const int index = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            const llama_pos pos = position + i;
            if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &pos)) {
                throw std::runtime_error("Native batch construction failed");
            }
        }
        llama_batch_ext_set_output_logits(batch.get(), count - 1, true);
        const int code = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (!poll()) throw cancelled{};
        if (code != 0) throw std::runtime_error("llama_process failed: " + std::to_string(code));
    }
    void reset() {
        llama_synchronize(ctx.get());
        llama_memory_clear(llama_get_memory(ctx.get()), true);
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
        if (context < 512 || context > 32768 || batch < 1 || batch > context || ubatch < 1 || ubatch > batch || threads < 1 || threads > 32) {
            throw invalid_request("Invalid context/batch/ubatch/threads; supported context is 512..32768");
        }
        check_headroom();
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
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
        mp.load_mtp = false;
        const auto model_path = config.at("model_path").get<std::string>();
        instance->model.reset(llama_model_load_from_file(model_path.c_str(), mp));
        if (!instance->model) throw std::runtime_error("Failed to load pinned GGUF; inspect native loader log");
        if (meta(instance->model.get(), "general.architecture") != "qwen35") {
            throw std::runtime_error("Expected pinned qwen35-family dense hybrid GGUF");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = batch;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn_type = config.value("flash_attention", true) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto kv_type = config.value("kv_type", std::string("q8_0"));
        if (kv_type != "q8_0" && kv_type != "f16") throw invalid_request("kv_type must be q8_0 or f16");
        cp.type_k = cp.type_v = kv_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        cp.no_perf = false;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.abort_callback = [](void * data) { return !static_cast<engine *>(data)->poll(); };
        cp.abort_callback_data = instance.get();
        instance->ctx.reset(llama_init_from_model(instance->model.get(), cp));
        if (!instance->ctx) throw std::runtime_error("Failed to allocate Qwen context; inspect native log");
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
        instance->decode(&warm_token, 1, static_cast<int>(warmup.size()));
        instance->reset();
        const auto warmup_ms = elapsed_ms(warmup_start, clock_type::now());
        check_headroom();
        size_t gpu_free = 0, gpu_total = 0;
        ggml_backend_dev_memory(gpu, &gpu_free, &gpu_total);
        instance->identity = json({
            {"model_id", MODEL_ID}, {"backend", "embedded-llama.cpp-ggml"}, {"backend_commit", BACKEND_COMMIT},
            {"device", ggml_backend_dev_description(gpu)}, {"device_backend", ggml_backend_dev_name(gpu)},
            {"architecture", meta(instance->model.get(), "general.architecture")},
            {"quantization", "Q4_K_M"}, {"model_parameters", llama_model_n_params(instance->model.get())},
            {"model_tensor_bytes", llama_model_size(instance->model.get())},
            {"context", llama_n_ctx(instance->ctx.get())}, {"batch", batch}, {"ubatch", ubatch}, {"threads", threads},
            {"kv_type", kv_type}, {"flash_attention", config.value("flash_attention", true)},
            {"vulkan_fa_flip_disabled", true},
            {"startup_warmup_tokens", warmup.size()}, {"startup_warmup_ms", warmup_ms},
            {"gpu_layers_requested", 999}, {"mmap", false}, {"lazy_weights", false}, {"mtp", false},
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
    // This C ABI is single-owner. Rust serializes requests on its native worker.
    struct reset_guard { engine & e; ~reset_guard() { if (e.callback) e.reset(); } } guard{e};
    try {
        e.reset();
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
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        // OpenAI-style presence/frequency penalties apply to generated history,
        // not input tokens. Keep the whole bounded generation, not only 64 tokens.
        if (repeat_penalty != 1.0f || presence_penalty != 0 || frequency_penalty != 0) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(llama_vocab_n_tokens(e.vocab), max_tokens, repeat_penalty, frequency_penalty, presence_penalty));
        }
        if (temperature == 0) llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        else {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(top_k));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(min_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(seed));
        }
        llama_perf_context_reset(e.ctx.get());
        const auto prefill_start = clock_type::now();
        for (size_t pos = 0; pos < tokens.size(); pos += e.batch_size) {
            check_headroom();
            const auto count = static_cast<int>(std::min<size_t>(e.batch_size, tokens.size() - pos));
            e.decode(tokens.data() + pos, count, static_cast<int>(pos));
        }
        llama_synchronize(e.ctx.get());
        const auto prefill_end = clock_type::now();
        text_stream stream(std::move(stops));
        std::string output;
        auto emit_text = [&](const std::string & text) {
            if (text.empty()) return;
            output += text;
            e.emit({{"type", "delta"}, {"text", text}});
        };
        int completion_tokens = 0, decode_tokens = 0;
        std::string finish_reason = "length";
        auto first_token = prefill_end;
        if (!thinking_prefix.empty()) emit_text(stream.push(thinking_prefix));
        for (int i = 0; i < max_tokens; ++i) {
            if (!e.poll()) throw cancelled{};
            if (i % 64 == 0) check_headroom();
            const auto token = llama_sampler_sample(sampler.get(), e.ctx.get(), -1);
            ++completion_tokens;
            if (i == 0) first_token = clock_type::now();
            if (llama_vocab_is_eog(e.vocab, token)) { finish_reason = "stop"; break; }
            emit_text(stream.push(common_token_to_piece(e.vocab, token, true)));
            if (stream.stopped) { finish_reason = "stop"; break; }
            if (i + 1 < max_tokens) {
                e.decode(&token, 1, static_cast<int>(tokens.size()) + i);
                ++decode_tokens;
            }
        }
        emit_text(stream.finish());
        const auto end = clock_type::now();
        const double prefill_ms = elapsed_ms(prefill_start, prefill_end);
        const double decode_ms = elapsed_ms(first_token, end);
        const json result = {
            {"type", "finished"}, {"text", output}, {"finish_reason", finish_reason},
            {"prompt_tokens", tokens.size()}, {"completion_tokens", completion_tokens},
            {"metrics", {{"prefill_ms", prefill_ms}, {"decode_ms", decode_ms}, {"ttft_ms", elapsed_ms(start, first_token)},
                {"prefill_tps", tokens.size() * 1000.0 / std::max(prefill_ms, 0.001)},
                {"decode_tps", decode_tokens * 1000.0 / std::max(decode_ms, 0.001)}, {"decode_tokens", decode_tokens},
                {"total_ms", elapsed_ms(start, end)}, {"available_physical_bytes", available_ram()}, {"seed", seed}}}
        };
        // State is reset before the terminal callback makes this request complete.
        e.reset();
        const auto serialized = result.dump(-1, ' ', false, json::error_handler_t::replace);
        if (!callback(serialized.c_str(), user)) return 1;
        return 0;
    } catch (const cancelled &) { return 1; }
    catch (const invalid_request & ex) { error_out(error, error_size, ex.what()); return 2; }
    catch (const json::exception & ex) { error_out(error, error_size, ex.what()); return 2; }
    catch (const std::exception & ex) { error_out(error, error_size, ex.what()); return -1; }
    catch (...) { error_out(error, error_size, "Unexpected native generation failure"); return -1; }
}
