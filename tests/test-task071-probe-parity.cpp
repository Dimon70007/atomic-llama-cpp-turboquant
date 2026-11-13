// TASK-071.1: probe-parity sampler semantics.
//
// The TASK-071 probe pins temperature=0, top_p=1, top_k=1, seed=42 and claims the result
// is the argmax token regardless of those other knobs. These tests turn that claim into a
// measurement: GREEN means the chain collapses to greedy, RED means some parameter is
// misapplied (clipped, reordered, dropped when "empty", or diverging from greedy).
//
// Refs: src/llama-sampler.cpp llama_sampler_init_top_p (p>=1 -> empty sampler),
// llama_sampler_temp_impl (temp<=0 -> argmax), common/sampling.cpp (product chain uses
// temp_ext, not temp; dist(seed) terminates the chain).
//
// Tie handling: on an exact tie two different ids are both valid argmax results, so tests
// assert "the surviving token has the max logit" and pin determinism separately.

#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<llama_token_data> make_logits(const std::vector<float> & logits) {
    std::vector<llama_token_data> out;
    out.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        out.emplace_back(llama_token_data{(llama_token) i, logits[i], 0.0f});
    }
    return out;
}

static float max_logit(const std::vector<llama_token_data> & src) {
    GGML_ASSERT(!src.empty());
    float best = src[0].logit;
    for (const auto & d : src) {
        best = std::max(best, d.logit);
    }
    return best;
}

static size_t tied_max_count(const std::vector<llama_token_data> & src) {
    const float best = max_logit(src);
    size_t n = 0;
    for (const auto & d : src) {
        if (d.logit == best) {
            ++n;
        }
    }
    return n;
}

static void assert_kept_argmax(const std::vector<llama_token_data> & src, llama_token got, const char * tag) {
    const float best = max_logit(src);
    const float got_l = src[got].logit;
    if (got_l != best) {
        printf("%s: got id=%d logit=%.9g but max=%.9g\n", tag, (int) got, got_l, best);
    }
    GGML_ASSERT(got_l == best);
}

// Run a sampler chain over a copy of `src` and return the selected token id.
static llama_token sample_with(llama_sampler * smpl, const std::vector<llama_token_data> & src) {
    std::vector<llama_token_data> cur = src;
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(smpl, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);
    const llama_token tok = cur_p.data[cur_p.selected].id;
    llama_sampler_free(smpl);
    return tok;
}

// Chain shape the product uses (common_sampler_init): samplers in order, then dist(seed).
// `use_temp_ext` mirrors the product path, which passes dynatemp_range=0 to temp_ext.
static llama_sampler * build_probe_chain(
        int      top_k,
        float    top_p,
        float    temp,
        uint32_t seed,
        bool     use_temp_ext = true) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(chain, use_temp_ext
            ? llama_sampler_init_temp_ext(temp, 0.0f, 1.0f)
            : llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

// Vectors with a unique maximum: chain, greedy and profile chain must agree on the id.
static const std::vector<std::vector<float>> UNIQUE_MAX_VECTORS = {
    {0.1f, 0.2f, 0.3f, 0.4f},
    {3.0f, 1.0f, 0.5f, 0.25f, 0.1f},
    {-1.0f, -2.0f, -3.0f, -4.0f},
    {1e4f, -1e4f, 0.0f, 5.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f},
};

// Vectors that may contain exact ties: only "kept a maximum" is asserted.
static const std::vector<std::vector<float>> TIE_PRONE_VECTORS = {
    {0.1f, 0.2f, 0.3f, 0.4f},
    {3.0f, 1.0f, 0.5f, 0.25f, 0.1f},
    {0.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, -2.0f, -3.0f, -4.0f},
    {1e4f, -1e4f, 0.0f, 5.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f},
    {2.0f, 2.0f, 1.999999f, 1.0f},
};

// T1: probe chain == greedy on a unique maximum.
static void test_probe_chain_equals_greedy() {
    printf("T1 probe chain (top_k=1, top_p=1, temp=0, dist) equals greedy\n");

    for (const auto & logits : UNIQUE_MAX_VECTORS) {
        const auto src = make_logits(logits);
        const llama_token via_greedy = sample_with(llama_sampler_init_greedy(), src);

        GGML_ASSERT(sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src) == via_greedy);
        GGML_ASSERT(sample_with(build_probe_chain(1, 1.0f, 0.0f, 42, false), src) == via_greedy);
    }

    for (const auto & logits : TIE_PRONE_VECTORS) {
        const auto src = make_logits(logits);
        assert_kept_argmax(src, sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src), "T1 probe");
        assert_kept_argmax(src, sample_with(llama_sampler_init_greedy(), src), "T1 greedy");
    }

    printf("T1 PASSED\n");
}

// T2: the product profile values (top_p=0.95, top_k=64) must not change the outcome at
// temp=0. If this fails, the probe is not greedy and TASK-071 stamps need a caveat.
static void test_probe_chain_profile_top_p_top_k() {
    printf("T2 profile top_p=0.95/top_k=64 at temp=0 keeps argmax\n");

    for (const auto & logits : TIE_PRONE_VECTORS) {
        const auto src = make_logits(logits);

        const llama_token via_profile     = sample_with(build_probe_chain(64, 0.95f, 0.0f, 42), src);
        const llama_token via_profile_tmp = sample_with(build_probe_chain(64, 0.95f, 0.0f, 42, false), src);
        const llama_token via_greedy      = sample_with(llama_sampler_init_greedy(), src);

        assert_kept_argmax(src, via_profile,     "T2 profile temp_ext");
        assert_kept_argmax(src, via_profile_tmp, "T2 profile temp");
        assert_kept_argmax(src, via_greedy,      "T2 greedy");
    }

    for (const auto & logits : UNIQUE_MAX_VECTORS) {
        const auto src = make_logits(logits);
        const llama_token via_profile = sample_with(build_probe_chain(64, 0.95f, 0.0f, 42), src);
        const llama_token via_probe   = sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src);
        GGML_ASSERT(via_profile == via_probe);
    }

    printf("T2 PASSED\n");
}

// T2b: ties are deterministic (same id every run) even though the id may differ from greedy.
static void test_probe_tie_break() {
    printf("T2b tie-break determinism on exact ties\n");

    const std::vector<std::vector<float>> ties = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    };

    for (const auto & logits : ties) {
        const auto src = make_logits(logits);

        const llama_token first  = sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src);
        const llama_token second = sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src);
        const llama_token greedy = sample_with(llama_sampler_init_greedy(), src);

        GGML_ASSERT(first == second);
        assert_kept_argmax(src, first, "T2b probe");
        assert_kept_argmax(src, greedy, "T2b greedy");

        printf("T2b tie n=%zu: probe=%d greedy=%d\n", logits.size(), (int) first, (int) greedy);
    }

    printf("T2b PASSED\n");
}

// T3: sampler order does not change the surviving argmax at temp=0.
static void test_probe_chain_order_independence() {
    printf("T3 sampler order does not change the surviving argmax at temp=0\n");

    for (const auto & logits : TIE_PRONE_VECTORS) {
        const auto src = make_logits(logits);

        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(1.0f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42));

        assert_kept_argmax(src, sample_with(chain, src), "T3 temp-first");
        assert_kept_argmax(src, sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src), "T3 default-order");
    }

    printf("T3 PASSED\n");
}

// T6: seed is inert at temp=0 - a seed-dependent pick would mean sampling is still active.
static void test_probe_seed_inert() {
    printf("T6 seed is inert at temp=0\n");

    const std::vector<uint32_t> seeds = {0, 1, 42, 123456};
    for (const auto & logits : TIE_PRONE_VECTORS) {
        const auto src = make_logits(logits);
        const bool unique = tied_max_count(src) == 1;
        const llama_token ref = sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src);

        for (uint32_t seed : seeds) {
            for (int rep = 0; rep < 50; ++rep) {
                const llama_token got = sample_with(build_probe_chain(1, 1.0f, 0.0f, seed), src);
                assert_kept_argmax(src, got, "T6");
                if (unique) {
                    GGML_ASSERT(got == ref);
                }
            }
        }
    }

    printf("T6 PASSED\n");
}

// T4: sensitivity control. At temp>0 the knobs MUST change the candidate set, otherwise a
// silently-ignored top_k/top_p would be indistinguishable from a working one (this is
// exactly what the mutation runs showed at temp=0). Uses a flat-ish distribution so both
// top_k and top_p have something to cut.
static size_t surviving_candidates(llama_sampler * smpl, const std::vector<llama_token_data> & src) {
    std::vector<llama_token_data> cur = src;
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    // stop before dist so we can count survivors
    llama_sampler_apply(smpl, &cur_p);
    const size_t n = cur_p.size;
    llama_sampler_free(smpl);
    return n;
}

static llama_sampler * build_truncation_only(int top_k, float top_p) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    return chain;
}

static void test_knobs_are_applied_at_temp_positive() {
    printf("T4 knobs actually cut candidates (temp>0 control)\n");

    // 16 tokens with equal logits: top_k must cut to exactly k.
    std::vector<float> flat(16, 1.0f);
    const auto src_flat = make_logits(flat);

    GGML_ASSERT(surviving_candidates(build_truncation_only(1, 1.0f), src_flat) == 1);
    GGML_ASSERT(surviving_candidates(build_truncation_only(4, 1.0f), src_flat) == 4);
    GGML_ASSERT(surviving_candidates(build_truncation_only(8, 1.0f), src_flat) == 8);
    GGML_ASSERT(surviving_candidates(build_truncation_only(1000, 1.0f), src_flat) == 16);

    // top_p=0.5 over 16 equal tokens keeps ~8 of them (nucleus covers half the mass).
    const size_t kept_half = surviving_candidates(build_truncation_only(1000, 0.5f), src_flat);
    GGML_ASSERT(kept_half < 16);
    GGML_ASSERT(kept_half >= 1);

    // top_p=1.0 must not remove anything.
    GGML_ASSERT(surviving_candidates(build_truncation_only(1000, 1.0f), src_flat) == 16);

    printf("T4 PASSED (top_k=1/4/8/1000 -> 1/4/8/16; top_p=0.5 cuts, top_p=1 keeps all)\n");
}

static uint32_t t071_rng_state = 0x9e3779b9u;
static uint32_t t071_rng() {
    t071_rng_state ^= t071_rng_state << 13;
    t071_rng_state ^= t071_rng_state >> 17;
    t071_rng_state ^= t071_rng_state << 5;
    return t071_rng_state;
}

// T7: fuzz - probe chain and profile chain must always keep an argmax.
static void test_probe_chain_fuzz() {
    printf("T7 fuzz: probe/profile chain keeps argmax on 1000 random vectors\n");

    for (int iter = 0; iter < 1000; ++iter) {
        const size_t n = 2 + (t071_rng() % 64);
        const int shape = (int) (t071_rng() % 5);

        std::vector<float> logits(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            switch (shape) {
                case 0: logits[i] = (float) ((int32_t) (t071_rng() % 2001) - 1000) / 100.0f; break;
                case 1: logits[i] = 1.0f; break;
                case 2: logits[i] = (t071_rng() % 3 == 0) ? 5.0f : 0.0f; break;
                case 3: logits[i] = ((float) (t071_rng() % 20000) - 10000.0f); break;
                default: logits[i] = (i == n - 1) ? 1.0f : 0.0f; break;
            }
        }

        const auto src = make_logits(logits);
        assert_kept_argmax(src, sample_with(build_probe_chain(1, 1.0f, 0.0f, 42), src), "T7 probe");
        assert_kept_argmax(src, sample_with(build_probe_chain(64, 0.95f, 0.0f, 42), src), "T7 profile");
    }

    printf("T7 PASSED\n");
}

int main(void) {
    ggml_time_init();

    test_probe_chain_equals_greedy();
    test_probe_chain_profile_top_p_top_k();
    test_probe_tie_break();
    test_probe_chain_order_independence();
    test_probe_seed_inert();
    test_knobs_are_applied_at_temp_positive();
    test_probe_chain_fuzz();

    printf("OK\n");
    return 0;
}
