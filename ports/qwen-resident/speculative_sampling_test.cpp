// Exercises the pinned upstream rejection verifier, not a duplicate algorithm.
// The tiny in-memory fixture and fixed-logit trials follow upstream
// tests/test-speculative-rejection-sampling.cpp. No model download is needed.
#include "common.h"
#include "sampling.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int vocabulary = 64;
constexpr int trials_per_case = 1000;
const llama_tokens penalty_history{4, 4, 8, 11};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void fill_tensor(ggml_tensor * tensor, void *) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)), 0.01f);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(values.size(), ggml_fp32_to_fp16(0.01f));
        ggml_backend_tensor_set(tensor, half.data(), 0, ggml_nbytes(tensor));
    } else {
        throw std::runtime_error("Unexpected fixture tensor type");
    }
}

struct fixture {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};

    fixture() {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
        gguf_set_val_str(metadata.get(), "general.architecture", "llama");
        gguf_set_val_u32(metadata.get(), "llama.vocab_size", vocabulary);
        gguf_set_val_u32(metadata.get(), "llama.context_length", 256);
        gguf_set_val_u32(metadata.get(), "llama.embedding_length", 32);
        gguf_set_val_u32(metadata.get(), "llama.block_count", 1);
        gguf_set_val_u32(metadata.get(), "llama.feed_forward_length", 64);
        gguf_set_val_u32(metadata.get(), "llama.attention.head_count", 2);
        gguf_set_val_u32(metadata.get(), "llama.attention.head_count_kv", 2);
        gguf_set_val_f32(metadata.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "test");
        std::vector<std::string> names;
        for (int id = 0; id < vocabulary; ++id) names.push_back("tok_" + std::to_string(id));
        std::vector<const char *> tokens;
        for (const auto & name : names) tokens.push_back(name.c_str());
        gguf_set_arr_str(metadata.get(), "tokenizer.ggml.tokens", tokens.data(), tokens.size());
        std::vector<float> scores(vocabulary, 0);
        gguf_set_arr_data(metadata.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());

        ggml_backend_dev_t no_gpu[] = {nullptr};
        auto mp = llama_model_default_params();
        mp.devices = no_gpu;
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
        mp.progress_callback = [](float, void *) { return true; };
        model.reset(llama_model_init_from_user(metadata.get(), fill_tensor, nullptr, mp));
        require(model != nullptr, "Tiny CPU model creation failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = cp.n_ubatch = 8;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 2;
        cp.offload_kqv = cp.op_offload = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context.reset(llama_init_from_model(model.get(), cp));
        require(context != nullptr, "Tiny CPU context creation failed");
        common_batch batch = common_batch_get_one(context.get(), {1, 2, 3});
        require(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "Tiny CPU prefill failed");
        // Writable output storage is a public API. Fix the logits so the test
        // covers sampling independently of model numerics or kernel choice.
        float * logits = llama_get_logits_ith(context.get(), -1);
        require(logits != nullptr, "Missing fixture logits");
        for (int id = 0; id < vocabulary; ++id) {
            logits[id] = id >= 4 && id < 28 ? 2.5f - 0.07f * float(id - 4) : -8.0f - 0.02f * float(id);
        }
    }
};

common_sampler_ptr make_sampler(llama_model * model, uint32_t seed) {
    common_params_sampling params;
    params.seed = seed;
    params.temp = 0.7f;
    params.top_k = 20;
    params.top_p = 0.8f;
    params.min_p = 0;
    params.min_keep = 1;
    params.penalty_last_n = 128;
    params.penalty_repeat = 1;
    params.penalty_present = 1.5f;
    params.penalty_freq = 0;
    params.backend_sampling = false;
    params.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
    common_sampler_ptr result(common_sampler_init(model, params));
    require(result != nullptr, "Fixture sampler creation failed");
    // Each trial starts at this exact generated history. Reusing a sampler
    // would change presence penalties and invalidate the fixed expected p.
    for (const auto token : penalty_history) common_sampler_accept(result.get(), token, true);
    return result;
}

std::vector<llama_token_data> distribution(common_sampler * sampler, llama_context * context) {
    common_sampler_sample(sampler, context, -1);
    const auto * candidates = common_sampler_get_candidates(sampler, true);
    return {candidates->data, candidates->data + candidates->size};
}

float probability(const std::vector<llama_token_data> & values, llama_token token) {
    for (const auto & value : values) if (value.id == token) return value.p;
    return 0;
}

void normalize(std::vector<llama_token_data> & q) {
    double sum = 0;
    for (const auto & token : q) sum += token.p;
    require(sum > 0, "Invalid fixture q");
    for (auto & token : q) token.p = static_cast<float>(token.p / sum);
}

llama_token draw(const std::vector<llama_token_data> & q, std::mt19937 & rng) {
    double total = 0;
    for (const auto & token : q) total += token.p;
    double value = std::uniform_real_distribution<double>(0, total)(rng);
    for (const auto & token : q) {
        value -= token.p;
        if (value < 0) return token.id;
    }
    return q.back().id;
}

void run_case(fixture & f, const std::vector<llama_token_data> & p, int kind) {
    const char * names[] = {"equal", "disjoint", "point-mass", "selector-like-16", "random-dense"};
    std::mt19937 draft_rng(0xd17f2305u + uint32_t(kind));
    std::vector<int> counts(vocabulary, 0);
    int accepted = 0;
    for (int trial = 0; trial < trials_per_case; ++trial) {
        std::vector<llama_token_data> q;
        if (kind == 0) {
            q = p;
        } else if (kind == 1) {
            for (int id = 0; id < vocabulary; ++id) {
                if (probability(p, id) == 0) q.push_back({id, 0, 1});
            }
            normalize(q);
        } else if (kind == 2) {
            q.push_back({p.front().id, 0, 1});
        } else if (kind == 3) {
            // Include overlap and target-filtered tokens, like a selector
            // whose top-16 differs from the penalized target distribution.
            for (int id = 4; id < 20; ++id) q.push_back({id, 0, std::exp(-0.11f * float(id - 4) / 0.7f)});
            normalize(q);
        } else {
            for (int id = 0; id < vocabulary; ++id) {
                q.push_back({id, 0, std::uniform_real_distribution<float>(0.01f, 1.0f)(draft_rng)});
            }
            normalize(q);
        }
        auto sampler = make_sampler(f.model.get(), 0x34a9e005u + uint32_t(trial) * 0x9e3779b9u);
        const llama_tokens draft{draw(q, draft_rng)};
        const auto result = common_sampler_sample_and_accept_n_rejection(sampler.get(), f.context.get(), {-1, -1}, draft, {q});
        require(result.size() == 1 || result.size() == 2, "Invalid rejection result length");
        require(result.front() >= 0 && result.front() < vocabulary, "Invalid rejection token ID");
        require(probability(p, result.front()) > 0, "Rejected path emitted outside filtered target support");
        if (result.size() == 2) {
            ++accepted;
            require(result.front() == draft.front(), "Accepted token differs from proposal");
        }
        ++counts[result.front()];
    }
    if (kind == 0) require(accepted == trials_per_case, "Equal distributions did not accept every proposal");
    if (kind == 1) require(accepted == 0, "Disjoint distributions accepted a proposal");
    double worst_sigma = 0;
    for (const auto & token : p) {
        const double observed = double(counts[token.id]) / trials_per_case;
        const double standard_error = std::sqrt(double(token.p) * (1.0 - token.p) / trials_per_case);
        const double error = std::abs(observed - token.p);
        worst_sigma = std::max(worst_sigma, standard_error > 0 ? error / standard_error : 0);
        require(error <= 6.0 * standard_error + 1e-7, "Empirical target probability exceeds predeclared 6-SE bound");
    }
    std::printf("%s: %d trials, %d accepted, support correct, worst %.3f SE\n", names[kind], trials_per_case, accepted, worst_sigma);
}

void repeatability_and_history(fixture & f, const std::vector<llama_token_data> & p) {
    auto first = make_sampler(f.model.get(), 9182);
    auto second = make_sampler(f.model.get(), 9182);
    const std::vector<llama_token_data> q{{0, 0, 1}}; // certainly rejected
    const auto a = common_sampler_sample_and_accept_n_rejection(first.get(), f.context.get(), {-1, -1}, {0}, {q});
    const auto b = common_sampler_sample_and_accept_n_rejection(second.get(), f.context.get(), {-1, -1}, {0}, {q});
    require(a == b && a.size() == 1, "Rejection seeded repeatability failed");
    require(common_sampler_last(first.get()) == a.front(), "Rejected proposal polluted accepted history");
    auto expected = make_sampler(f.model.get(), 1);
    common_sampler_accept(expected.get(), a.front(), true);
    const auto actual_p = distribution(first.get(), f.context.get());
    const auto expected_p = distribution(expected.get(), f.context.get());
    for (int id = 0; id < vocabulary; ++id) {
        require(std::abs(probability(actual_p, id) - probability(expected_p, id)) < 1e-7f,
            "Presence penalty history differs after rejection");
    }
    // All accepted proposals plus bonus preserve result shape and sampler
    // history. Reusing this fixed logit row intentionally isolates sampling.
    auto accepted = make_sampler(f.model.get(), 27);
    const auto result = common_sampler_sample_and_accept_n_rejection(accepted.get(), f.context.get(), {-1, -1}, {p.front().id}, {p});
    require(result.size() == 2 && result.front() == p.front().id, "Accepted-prefix plus bonus contract failed");
    require(common_sampler_last(accepted.get()) == result.back(), "Bonus token was not accepted into history");
    std::puts("seeded repeatability, rejected history and accepted-prefix/bonus contracts passed");
}
} // namespace

int main() {
    try {
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
        }, nullptr);
        llama_backend_init();
        {
            fixture f;
            auto reference = make_sampler(f.model.get(), 18);
            const auto p = distribution(reference.get(), f.context.get());
            require(p.size() >= 2 && p.size() <= 20, "Fixture did not exercise filtered stochastic support");
            require(probability(p, 4) == 0 && probability(p, 8) == 0 && probability(p, 11) == 0,
                "Fixture presence penalties did not remove repeated candidates");
            for (int kind = 0; kind < 5; ++kind) run_case(f, p, kind);
            repeatability_and_history(f, p);
        }
        llama_backend_free();
        std::puts("Qwen rejection sampling CPU contracts passed (5000 empirical trials)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Qwen rejection sampling test failed: %s\n", error.what());
        return 1;
    }
}
