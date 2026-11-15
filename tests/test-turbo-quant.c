#include "ggml.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

/* Must match GGML_TQ_DOT_CHUNK in ggml/src/ggml-cpu/ggml-cpu.c. */
#define TQ_DOT_CHUNK 256

/* Block-boundary invariant behind the chunked TurboQuant vec_dot kernels.
 *
 * ggml_vec_dot_{turbo,tq}_* stage their dequant through a fixed-size chunk
 * instead of a malloc'd whole-row buffer. That is only correct because every
 * TurboQuant row dequant is a pure per-block loop with no cross-block state:
 * dequantizing [0, k) in one call must be bit-identical to dequantizing it in
 * block-aligned pieces with the source pointer advanced by
 * (piece / blck_size) * type_size.
 *
 * This asserts exactly that, at row lengths that straddle the chunk size, so a
 * future row dequant that carries state across blocks -- or a wrong stride in
 * the vec_dot pointer advance -- fails here rather than silently corrupting a
 * CPU-offloaded matmul. */
static int check_chunked_dequant(enum ggml_type type, const char * name, int64_t k) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(type);

    const int64_t blck = tr->blck_size;
    if (k % blck != 0) {
        printf("  %-9s k=%-5lld SKIP (not a whole number of blocks)\n", name, (long long) k);
        return 0;
    }

    float * src   = malloc((size_t) k * sizeof(float));
    void  * q     = malloc(ggml_row_size(type, k));
    float * whole = malloc((size_t) k * sizeof(float));
    float * piece = malloc((size_t) k * sizeof(float));

    for (int64_t i = 0; i < k; i++) {
        src[i] = sinf((float) i * 0.037f) * 3.0f;
    }
    ggml_quantize_chunk(type, src, q, 0, 1, k, NULL);

    tr->to_float(q, whole, k);

    const int64_t chunk = (TQ_DOT_CHUNK / blck) * blck;
    const char * p = (const char *) q;
    for (int64_t i = 0; i < k; i += chunk) {
        const int64_t nc = (chunk < k - i) ? chunk : (k - i);
        tr->to_float(p, piece + i, nc);
        p += (nc / blck) * tr->type_size;
    }

    int64_t bad = -1;
    for (int64_t i = 0; i < k; i++) {
        if (memcmp(&whole[i], &piece[i], sizeof(float)) != 0) {
            bad = i;
            break;
        }
    }

    if (bad >= 0) {
        printf("  %-9s k=%-5lld FAIL at [%lld]: whole=%.9g chunked=%.9g\n",
               name, (long long) k, (long long) bad,
               (double) whole[bad], (double) piece[bad]);
    } else {
        printf("  %-9s k=%-5lld ok (blck=%lld, chunk=%lld)\n",
               name, (long long) k, (long long) blck, (long long) chunk);
    }

    free(src);
    free(q);
    free(whole);
    free(piece);
    return bad >= 0 ? 1 : 0;
}

/* Codec quality gate.
 *
 * Tests 1-3 used to only printf MSE/Cosine, so the numbers AGENTS.md quotes
 * ("turbo3 cosine=1.0, turbo4 cosine=0.9956") were never enforced and a codec
 * regression passed CI. The vectors are fixed and both codecs are
 * deterministic, so each case gets a hard cosine floor and MSE ceiling.
 *
 * Both turbo mids are covered on every vector: turbo4 has a separate codec
 * (nibble centroids, no QJL) and a bug in it is invisible to turbo3. */

typedef void (*quant_row_fn)(const float * x, void * y, long long k);
typedef void (*dequant_row_fn)(const void * x, float * y, long long k);

enum fixture { FIX_BASIS, FIX_SINE, FIX_COSINE };

typedef struct {
    const char * name;
    quant_row_fn   quant;
    dequant_row_fn dequant;
    enum fixture   fixture;
    double cos_min;
    double mse_max;
} codec_case;

static void fill_fixture(enum fixture f, float * x, int d) {
    switch (f) {
        case FIX_BASIS:
            memset(x, 0, (size_t) d * sizeof(float));
            x[0] = 1.0f;
            break;
        case FIX_SINE:
            for (int i = 0; i < d; i++) x[i] = sinf(i*0.1f+0.5f) * 10.0f;
            break;
        case FIX_COSINE:
            for (int i = 0; i < d; i++) x[i] = cosf(i*0.2f) * 5.0f;
            break;
    }
}

/* Floors are the measured values of the shipped codec minus a small margin.
 * A codec that gets better passes; one that gets worse does not. */
static const codec_case kCodecCases[] = {
    { "turbo3/basis",  quantize_row_turbo3_0_ref, dequantize_row_turbo3_0, FIX_BASIS,  0.9999, 1e-5 },
    { "turbo3/sine",   quantize_row_turbo3_0_ref, dequantize_row_turbo3_0, FIX_SINE,   0.9855, 1.3820 },
    { "turbo3/cosine", quantize_row_turbo3_0_ref, dequantize_row_turbo3_0, FIX_COSINE, 0.9805, 0.4820 },
    { "turbo4/basis",  quantize_row_turbo4_0_ref, dequantize_row_turbo4_0, FIX_BASIS,  0.9999, 1e-5 },
    { "turbo4/sine",   quantize_row_turbo4_0_ref, dequantize_row_turbo4_0, FIX_SINE,   0.9950, 0.4580 },
    { "turbo4/cosine", quantize_row_turbo4_0_ref, dequantize_row_turbo4_0, FIX_COSINE, 0.9953, 0.1115 },
};

/* dequant stays in the WHT-rotated domain (Q is rotated by the graph too, so
 * <Q_rot, K_rot> is the correct score); the inverse WHT is applied here only
 * to compare against the original input. */
static int check_codec_case(const codec_case * c) {
    const int d = 128;
    char buf[256];
    float input[128], output[128];

    fill_fixture(c->fixture, input, d);
    c->quant(input, buf, d);
    c->dequant(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);

    double mse = 0.0, dot = 0.0, ni = 0.0, no = 0.0;
    for (int i = 0; i < d; i++) {
        mse += (double) (input[i] - output[i]) * (double) (input[i] - output[i]);
        dot += (double) input[i] * (double) output[i];
        ni  += (double) input[i] * (double) input[i];
        no  += (double) output[i] * (double) output[i];
    }
    mse /= d;
    const double cos = (ni > 0.0 && no > 0.0) ? dot / sqrt(ni) / sqrt(no) : 0.0;

    const int ok = cos >= c->cos_min && mse <= c->mse_max;
    printf("%s %-14s MSE=%.8f (max %.4f) Cosine=%.6f (min %.6f)\n",
           ok ? "ok  " : "FAIL", c->name, mse, c->mse_max, cos, c->cos_min);
    return ok ? 0 : 1;
}

int main(void) {
    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    printf("Tests 1-3: codec quality floors (turbo3 + turbo4)\n");
    int codec_fails = 0;
    for (size_t i = 0; i < sizeof(kCodecCases) / sizeof(kCodecCases[0]); i++) {
        codec_fails += check_codec_case(&kCodecCases[i]);
    }
    printf("\n");

    /* Test 4: chunk-boundary invariant for every type with a chunked vec_dot */
    printf("Test 4: chunked dequant == whole-row dequant\n");
    int failures = 0;
    {
        const enum ggml_type types[] = {
            GGML_TYPE_TQ3_1S, GGML_TYPE_TQ4_1S,
            GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0,
        };
        const char * names[] = { "tq3_1s", "tq4_1s", "turbo2_0", "turbo3_0", "turbo4_0" };
        /* straddle TQ_DOT_CHUNK from both sides, plus a realistic row */
        const int64_t lens[] = { 128, 256, 384, 512, 4096 };

        for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
            for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
                failures += check_chunked_dequant(types[t], names[t], lens[i]);
            }
        }
    }
    printf("\n");

    if (codec_fails || failures) {
        printf("=== FAILED: %d codec quality, %d chunk-boundary ===\n", codec_fails, failures);
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
