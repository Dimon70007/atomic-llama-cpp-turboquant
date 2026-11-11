# TASK-066.18 — compile-time ablation levers (tip ON by default).
# Disable tip behavior (= restore master-like path at that site) via:
#   -DTASK066_DISABLE_FA_TILE=ON
#   …or batch: -DTASK066_ABLATION_DISABLE="FA_TILE;NO_MV_EXT;…"
#
# Orchestrator builds per cell with a dedicated BUILD_DIR + these flags.
# Metal shaders need the same macros via XC_FLAGS (FLOAT_TILE).
# FA_TILE also gates CUDA ggml-cuda/fattn.cu (TASK-069 B1 kernel pin) via
# add_compile_definitions below — not Metal-only.

set(_TASK066_LEVER_IDS
    FA_TILE
    NO_MV_EXT
    FLOAT_TILE
    MM_MIN
    FUSION_CB
    # MOE_VIEW_ADD retired (TASK-069.3): MOE OFF broke ubatch=1 reference - always view+ADD.
    SIZE_SWA_UB
    ISWA_PADS
    SHARED_KV
    SWA_HWM_LOG
)

set(TASK066_ABLATION_DISABLE "" CACHE STRING
    "Semicolon-separated TASK-066.18 lever short ids to disable (FA_TILE,NO_MV_EXT,...)")

foreach(_id IN LISTS _TASK066_LEVER_IDS)
    option(TASK066_DISABLE_${_id} "TASK-066.18: disable tip lever ${_id} (master-like)" OFF)
endforeach()

# Batch string expands into individual options (fail-closed on unknown ids).
if(TASK066_ABLATION_DISABLE)
    foreach(_tok IN LISTS TASK066_ABLATION_DISABLE)
        string(STRIP "${_tok}" _tok)
        if(_tok STREQUAL "")
            continue()
        endif()
        set(_known FALSE)
        foreach(_id IN LISTS _TASK066_LEVER_IDS)
            if(_tok STREQUAL "${_id}")
                set(_known TRUE)
                set(TASK066_DISABLE_${_id} ON CACHE BOOL "TASK-066.18: disable tip lever ${_id}" FORCE)
                break()
            endif()
        endforeach()
        if(NOT _known)
            message(FATAL_ERROR
                "TASK066_ABLATION_DISABLE: unknown lever id '${_tok}' "
                "(known: ${_TASK066_LEVER_IDS})")
        endif()
    endforeach()
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

# TASK-069 H-SWA-1c (experimental; OFF by default — not product KEEP until measured).
# When ON: headroom = PAD256(n_ubatch); outer_pad stays 256 (FA).
# History: outer-only NO_EFFECT; 1b headroom+outer=block_pad → cross @35 not MATCH.
option(TASK066_ENABLE_H_SWA_1
    "TASK-069 H-SWA-1c: headroom=PAD256(n_ubatch), outer_pad=256" OFF)
if(TASK066_ENABLE_H_SWA_1)
    add_compile_definitions(TASK066_ENABLE_H_SWA_1=1)
    message(STATUS "TASK-069: ENABLE H_SWA_1c (headroom=PAD256(n_ubatch), outer=256)")
endif()

# TASK-069 H-BUF-1 (experimental; OFF by default).
# When ON: size_swa = PAD(raw + outer_pad, outer_pad) — one FA-pad of ring slack
# beyond the usual PAD(raw, outer_pad). Targets ring-full@1536 (ub512 SIZE-off/1c).
# Not a static huge cache (H-REJECT-STATIC); measure before KEEP.
option(TASK066_ENABLE_H_BUF_1
    "TASK-069 H-BUF-1: size_swa ring slack = +outer_pad after PAD(raw)" OFF)
if(TASK066_ENABLE_H_BUF_1)
    add_compile_definitions(TASK066_ENABLE_H_BUF_1=1)
    message(STATUS "TASK-069: ENABLE H_BUF_1 (size_swa += outer_pad slack)")
endif()

# Exported for ggml-metal CMakeLists (Metal shader compile).
set(TASK066_ABLATION_METAL_FLAGS "${TASK066_ABLATION_METAL_FLAGS}" CACHE INTERNAL
    "Metal -D flags for TASK-066.18 ablation")
