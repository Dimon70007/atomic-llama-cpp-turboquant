#!/usr/bin/env bash
# TASK-066 follow-up B4 — E2E MTP vs greedy / verify-width bit-exact gate.
#
# Stand-in localizer (same residual class as product MTP verify width):
#   test-fullgraph-decode-row0-parity — after prefix, decode N=1 vs N=2 row0
#   logits must be bit-exact (temperature=0 class). Expected RED until Track A
#   (+ B1 if shared-KV) land. See TASK-066-cb-fusion-followup.md B4.
#
# True MTP-on vs greedy token identity still needs a draft GGUF; this suite
# gates the compute non-determinism that MTP verify width exposes.
#
# Usage / env: same as run-task066-fullgraph-corners.sh
#   TASK066_MODEL=... TASK066_CTX=... ./tests/run-task066-mtp-e2e-corners.sh [OUT_DIR]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mtp-e2e-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

{
  echo "track=B4"
  echo "contract=E2E temperature=0 identity under verify-shaped batch (N=1 vs N=2 row0)"
  echo "delegate=run-task066-fullgraph-corners.sh"
} | tee "${OUT_DIR}/summary.txt"

set +e
bash "${REPO_ROOT}/tests/run-task066-fullgraph-corners.sh" "${OUT_DIR}/fullgraph"
RC=$?
set -e
echo "fullgraph_exit=${RC}" | tee -a "${OUT_DIR}/summary.txt"

if [[ "${RC}" -ne 0 ]]; then
  echo "VERDICT=FAIL (B4 RED until fullgraph / MTP≡greedy class is green)" | tee -a "${OUT_DIR}/summary.txt"
  exit "${RC}"
fi

echo "VERDICT=PASS" | tee -a "${OUT_DIR}/summary.txt"
echo "OK: TASK-066 mtp-e2e (fullgraph delegate)"
exit 0
