#include <gpt-oss.h>
#include <internal/model.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

#include <benchmark/benchmark.h>


constexpr std::uint32_t kNumGeneratedTokens = 100;


static void end2end(benchmark::State& state, const char* env_var_name) {
    const char* model_path = getenv(env_var_name);
    if (model_path == NULL) {
        state.SkipWithError(std::format("environment variable {} is not set", env_var_name));
        return;
    }

    gptoss_model_t model_ptr = nullptr;
    gptoss_status status = gptoss_model_create_from_file(model_path, &model_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to load model from file {}", model_path));
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_model_t>, decltype(&gptoss_model_release)> model(model_ptr, gptoss_model_release);

    gptoss_context_t context_ptr = nullptr;
    status = gptoss_context_create(model.get(), /*context_lenght=*/0, &context_ptr);
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to create Context object");
        return;
    }
    std::unique_ptr<std::remove_pointer_t<gptoss_context_t>, decltype(&gptoss_context_release)> context(context_ptr, gptoss_context_release);

    const char* prompt = "why did the chicken cross the road?";
    std::size_t num_prompt_tokens = 0;
    status = gptoss_context_append_chars(context.get(), prompt, strlen(prompt), &num_prompt_tokens);
    if (status != gptoss_status_success) {
        state.SkipWithError(std::format("failed to tokenize prompt \"{}\"", prompt));
        return;
    }

    // Prefill
    status = gptoss_context_process(context.get());
    if (status != gptoss_status_success) {
        state.SkipWithError("failed to prefill Context object");
        return;
    }
    const std::size_t num_kvcache_tokens = context->num_kv_tokens;

    std::uint64_t rng_seed = 0;
    for (std::uint32_t i = 0; i < 3; i++) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    for (auto _ : state) {
        const std::uint64_t current_rng_seed = rng_seed++;
        context->num_kv_tokens = num_prompt_tokens;
        context->num_tokens = num_prompt_tokens;

        std::array<std::uint32_t, kNumGeneratedTokens> tokens;
        std::size_t num_generated_tokens = 0;
        do {
            std::size_t num_current_generated_tokens = 0;
            status = gptoss_context_sample(context.get(), /*temperature=*/1.0f, /*rng_state=*/current_rng_seed,
                /*max_tokens=*/kNumGeneratedTokens - num_generated_tokens, tokens.data(), &num_current_generated_tokens);
            if (status != gptoss_status_success) {
                state.SkipWithError("failed to sample from the Context object");
                return;
            }
            num_generated_tokens += num_current_generated_tokens;
        } while (num_generated_tokens < kNumGeneratedTokens);
    }

    state.counters["generations"] =
        benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
    state.counters["tokens"] =
        benchmark::Counter(state.iterations() * kNumGeneratedTokens, benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(end2end, gpt_oss_20b, "GPT_OSS_20B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(end2end, gpt_oss_120b, "GPT_OSS_120B_PATH")
    ->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
