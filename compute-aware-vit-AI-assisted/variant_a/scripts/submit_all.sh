#!/bin/bash
# Launch all Variant A training jobs via nohup (each runs independently in background).
# For sequential execution use run_all.sh instead.
#
# Usage (from repo root):
#   bash scripts/submit_all.sh

SCRIPTS=/home/arooba/compute-aware-vit-variant-a/scripts
mkdir -p "${SCRIPTS}/logs"

echo "Launching Variant A training jobs..."
echo ""

bash "${SCRIPTS}/run_baseline.sh"
bash "${SCRIPTS}/run_static_25.sh"
bash "${SCRIPTS}/run_static_50.sh"
bash "${SCRIPTS}/run_static_75.sh"
bash "${SCRIPTS}/run_fixed_25.sh"
bash "${SCRIPTS}/run_fixed_50.sh"
bash "${SCRIPTS}/run_fixed_75.sh"
bash "${SCRIPTS}/run_controller.sh"

echo ""
echo "All 8 jobs launched. Logs: ${SCRIPTS}/logs/"
echo "Monitor with: ps aux | grep python"
