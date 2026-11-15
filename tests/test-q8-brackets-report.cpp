// TASK-068 10.31 - dense-mode schedule report across real models.
//
// Loads a GGUF, reads the model facts the planner actually uses (n_layer,
// has_kv, is_swa) and prints the q8 layer plan for every mode. No generation,
// no KV allocation, so it is cheap and safe to run for many models.
//
// Usage: test-q8-brackets-report model1.gguf [model2.gguf ...]
//
// Exit code 0 when every model planned cleanly, 1 on a refusal (the refusal is
// printed: a too-wide tail on a shared-KV model is a valid answer, not a crash).

#include "llama.h"
#include "llama-model.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

// Declared in src/llama-kv-cache.h; keep this unit free of the heavy KV header.
std::vector<uint8_t> llama_kv_cache_q8_brackets_plan_layers(
        uint32_t n_layer,
        const std::vector<bool> & has_kv,
        const std::vector<bool> & is_swa,
        bool dense_mode,
        uint32_t head_req,
        uint32_t tail_req);

uint32_t llama_kv_cache_q8_brackets_percent_to_layers(uint32_t n_layer, uint32_t percent);

struct llama_kv_cache_q8_report_widths {
    uint32_t head = 0;
    uint32_t tail = 0;
    uint32_t head_pct = 0;
    uint32_t tail_pct = 0;
};
llama_kv_cache_q8_report_widths llama_kv_cache_q8_brackets_report_widths(
        uint32_t n_layer, uint32_t n_kv_layer);

namespace {

// Print a plan as "0,1,6,7" plus a compact summary.
void print_plan(const char * label, const std::vector<uint8_t> & plan,
                uint32_t head, uint32_t tail, const std::vector<bool> & is_swa,
                const std::vector<bool> & has_kv) {
    uint32_t n = 0;
    uint32_t dense_q8 = 0;
    uint32_t dead = 0;
    std::string layers;
    for (size_t il = 0; il < plan.size(); ++il) {
        if (!plan[il]) {
            continue;
        }
        ++n;
        if (!has_kv[il]) {
            ++dead;
        }
        if (!is_swa[il]) {
            ++dense_q8;
        }
        if (layers.size() < 120) {
            layers += (layers.empty() ? "" : ",") + std::to_string(il);
        }
    }
    const double pct = plan.empty() ? 0.0 : 100.0 * n / plan.size();
    printf("  %-22s q8=%2u/%-2zu (%5.1f%%)  dense_in_q8=%u  dead=%u  [%s]\n",
           label, n, plan.size(), pct, dense_q8, dead, layers.c_str());
    (void) head;
    (void) tail;
}

// q8 layers that no longer follow a dense layer - the structural invariant of
// neighbour mode.
uint32_t count_misplaced_neighbours(const std::vector<uint8_t> & plan,
                                    const std::vector<bool> & is_swa,
                                    const std::vector<bool> & has_kv,
                                    uint32_t head) {
    uint32_t bad = 0;
    for (uint32_t il = 0; il < plan.size(); ++il) {
        if (!plan[il] || il < head) {
            continue;
        }
        if (!has_kv[il]) {
            ++bad;  // selected a layer that owns no cache: dead slot
            continue;
        }
        // A neighbour slot must follow a dense layer.
        bool hangs = false;
        for (uint32_t d = 1; d <= il; ++d) {
            if (!is_swa[il - d]) {
                hangs = true;
                break;
            }
        }
        if (!hangs) {
            ++bad;
        }
    }
    return bad;
}

void report_model(const std::string & path) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // metadata only: never claim device memory
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        printf("FAIL load %s\n", path.c_str());
        return;
    }
    const int n_layer = llama_model_n_layer(model);
    std::vector<bool> has_kv(n_layer, false);
    std::vector<bool> is_swa(n_layer, false);
    int n_dense = 0;
    int n_kv = 0;
    std::string dense_list;
    for (int il = 0; il < n_layer; ++il) {
        has_kv[il] = model->hparams.has_kv(il);
        is_swa[il] = model->hparams.is_swa(il);
        if (has_kv[il]) {
            ++n_kv;
            if (!is_swa[il]) {
                ++n_dense;
                if (dense_list.size() < 90) {
                    dense_list += (dense_list.empty() ? "" : ",") + std::to_string(il);
                }
            }
        }
    }
    const std::string leaf = path.substr(path.find_last_of("/\\") + 1);
    printf("\n=== %s\n", leaf.c_str());
    printf("  n_layer=%d  kv_layers=%d  dense_with_kv=%d  dense=[%s]\n",
           n_layer, n_kv, n_dense, dense_list.c_str());
    // Per-layer map: which layers exist, which own a cache, which are dense.
    printf("  map: ");
    for (int il = 0; il < n_layer; ++il) {
        printf("%d:%s%s ", il, has_kv[il] ? "kv" : "--",
               !has_kv[il] ? "" : (is_swa[il] ? "swa" : "DEN"));
    }
    printf("\n");
    // Neighbour mode needs full-context layers to hang the q8 slots off. A model
    // with n_swa=0 has none, so every q8 layer would have to be a dense one and
    // the mode cannot express "after dense" at all.
    if (n_dense == 0) {
        printf("  NOTE: no SWA/dense split (n_swa=0) -> dense modes do not apply;"
               " use positional widths\n");
    } else if (n_dense == n_kv) {
        printf("  NOTE: every KV layer is dense -> 'after dense' has no target;"
               " use positional widths\n");
    }

    // Read the widths through the same resolver the loader uses, so the report
    // cannot disagree with the real schedule (a hand-rolled percent here is what
    // hid the shared-KV denominator bug).
    struct ReportWidths {
        uint32_t head, tail, head_pct, tail_pct;
    };
    ReportWidths w{};
    {
        const auto rw = llama_kv_cache_q8_brackets_report_widths(n_layer, n_kv);
        w.head = rw.head;
        w.tail = rw.tail;
        w.head_pct = rw.head_pct;
        w.tail_pct = rw.tail_pct;
    }
    const uint32_t head = w.head;
    const uint32_t tail = w.tail;
    printf("  widths %u%%/%u%% of %d kv layers -> head=%u tail=%u (%.1f%% of real cache)\n",
           w.head_pct, w.tail_pct, n_kv, head, tail,
           n_kv ? 100.0 * (head + tail) / n_kv : 0.0);

    struct Mode {
        const char * label;
        bool dense;
    };
    const Mode modes[] = {
        { "positional", false },
        { "dense",      true  },
    };
    for (const auto & m : modes) {
        try {
            const auto plan = llama_kv_cache_q8_brackets_plan_layers(
                    n_layer, has_kv, is_swa, m.dense, head, tail);
            print_plan(m.label, plan, head, tail, is_swa, has_kv);
            if (m.dense) {
                const uint32_t bad = count_misplaced_neighbours(plan, is_swa, has_kv, head);
                printf("  %-22s misplaced=%u %s\n", "  invariant", bad,
                       bad == 0 ? "ok" : "FAIL");
            }
        } catch (const std::exception & e) {
            printf("  %-22s REFUSED: %s\n", m.label, e.what());
        }
    }
    llama_model_free(model);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [more.gguf ...]\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    for (int i = 1; i < argc; ++i) {
        report_model(argv[i]);
    }
    llama_backend_free();
    return 0;
}
