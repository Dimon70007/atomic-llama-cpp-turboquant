# TASK-066.18 — compile-time ablation levers.
#
# Default ship policy (all backends: Metal, CUDA, …): only 066.18 KEEP levers ON.
#   ON:  FA_TILE, NO_MV_EXT, FLOAT_TILE, ISWA_PADS
#   OFF: MM_MIN, FUSION_CB, MOE_VIEW_ADD, SHARED_KV, SWA_HWM_LOG
# (Combined verdict TASK-066.18 Mode A+B — not "all 10 tip paths".)
#
# TASK-071: SIZE_SWA_UB lever REMOVED. It selected the TASK-066.17 headroom formula
# (PAD128(n_batch)) which was rejected: it dropped -ub from size_swa to green ubatch
# tests and made -b a quality lever. size_swa headroom is now upstream-only (n_ubatch).
#
# Full tip (all 9 remaining ON) for research / suite ON cell:
#   -DTASK066_ABLATION_DISABLE=   (empty)  or  -DTASK066_TIP_KEEP_ALL=ON
#
# A/B OFF / tip−1 (separate BUILD_DIR):
#   -DTASK066_ABLATION_DISABLE="FA_TILE;NO_MV_EXT;…"
#
# Metal shaders get the same macros via XC_FLAGS (FLOAT_TILE).

set(_TASK066_LEVER_IDS
    FA_TILE
    NO_MV_EXT
    FLOAT_TILE
    MM_MIN
    FUSION_CB
    MOE_VIEW_ADD
    ISWA_PADS
    SHARED_KV
    SWA_HWM_LOG
)

# Non-KEEP ids → DISABLE by default (leaves the 5 KEEP ON).
set(_TASK066_DEFAULT_DISABLE "MM_MIN;FUSION_CB;MOE_VIEW_ADD;SHARED_KV;SWA_HWM_LOG")

option(TASK066_TIP_KEEP_ALL "TASK-066.18: enable all tip levers (ignore default DISABLE)" OFF)

set(TASK066_ABLATION_DISABLE "${_TASK066_DEFAULT_DISABLE}" CACHE STRING
    "Semicolon-separated lever ids to disable (default=${_TASK066_DEFAULT_DISABLE}; empty or TIP_KEEP_ALL=ON = all tip ON)")

foreach(_id IN LISTS _TASK066_LEVER_IDS)
    option(TASK066_DISABLE_${_id} "TASK-066.18: disable tip lever ${_id} (master-like)" OFF)
endforeach()

# Resolve effective DISABLE list.
set(_TASK066_EFFECTIVE_DISABLE "${TASK066_ABLATION_DISABLE}")
if(TASK066_TIP_KEEP_ALL)
    set(_TASK066_EFFECTIVE_DISABLE "")
    message(STATUS "TASK-066.18: TASK066_TIP_KEEP_ALL=ON → all tip levers ON")
endif()

# Reset per-lever CACHE, then apply DISABLE (avoids stale ablation bits).
foreach(_id IN LISTS _TASK066_LEVER_IDS)
    set(TASK066_DISABLE_${_id} OFF CACHE BOOL
        "TASK-066.18: disable tip lever ${_id} (master-like)" FORCE)
endforeach()

if(_TASK066_EFFECTIVE_DISABLE STREQUAL "")
    message(STATUS "TASK-066.18: tip KEEP ALL ON (9 levers; SIZE_SWA_UB removed in TASK-071)")
else()
    message(STATUS "TASK-066.18: DISABLE=${_TASK066_EFFECTIVE_DISABLE}")
    foreach(_tok IN LISTS _TASK066_EFFECTIVE_DISABLE)
        string(STRIP "${_tok}" _tok)
        if(_tok STREQUAL "")
            continue()
        endif()
        set(_known FALSE)
        foreach(_id IN LISTS _TASK066_LEVER_IDS)
            if(_tok STREQUAL "${_id}")
                set(_known TRUE)
                set(TASK066_DISABLE_${_id} ON CACHE BOOL
                    "TASK-066.18: disable tip lever ${_id}" FORCE)
                break()
            endif()
        endforeach()
        if(NOT _known)
            string(REPLACE ";" ", " _TASK066_KNOWN_CSV "${_TASK066_LEVER_IDS}")
            message(FATAL_ERROR
                "TASK066_ABLATION_DISABLE: unknown lever id '${_tok}' "
                "(known: ${_TASK066_KNOWN_CSV})")
        endif()
    endforeach()
    # Report which levers remain ON after DISABLE apply.
    set(_TASK066_KEEP_ON "")
    foreach(_id IN LISTS _TASK066_LEVER_IDS)
        if(NOT TASK066_DISABLE_${_id})
            list(APPEND _TASK066_KEEP_ON "${_id}")
        endif()
    endforeach()
    string(REPLACE ";" " " _TASK066_KEEP_ON_MSG "${_TASK066_KEEP_ON}")
    message(STATUS "TASK-066.18: KEEP ON = ${_TASK066_KEEP_ON_MSG}")
endif()

set(TASK066_ABLATION_CXX_DEFINES "")
set(TASK066_ABLATION_METAL_FLAGS "")
foreach(_id IN LISTS _TASK066_LEVER_IDS)
    if(TASK066_DISABLE_${_id})
        list(APPEND TASK066_ABLATION_CXX_DEFINES "TASK066_DISABLE_${_id}=1")
        list(APPEND TASK066_ABLATION_METAL_FLAGS "-DTASK066_DISABLE_${_id}=1")
        message(STATUS "TASK-066.18 ablation: DISABLE ${_id}")
    endif()
endforeach()

foreach(_def IN LISTS TASK066_ABLATION_CXX_DEFINES)
    add_compile_definitions(${_def})
endforeach()

# Exported for ggml-metal CMakeLists (Metal shader compile).
set(TASK066_ABLATION_METAL_FLAGS "${TASK066_ABLATION_METAL_FLAGS}" CACHE INTERNAL
    "Metal -D flags for TASK-066.18 ablation")
