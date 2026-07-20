#!/bin/bash
# Phase 9 Task Runner — runs all 5 tasks sequentially on single GPU.
# Launched from tmux for long-running safety.
set -euo pipefail

cd /home/hiaskc/Talant/graduation
export PYTHONPATH="scripts:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# NCCL fix
NCCL_LIB=$(conda run -n grad python -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')/lib/libnccl.so.2
export LD_PRELOAD="$NCCL_LIB"

echo "============================================================"
echo "Phase 9 Task Runner — $(date)"
echo "============================================================"
echo "Tasks:"
echo "  1. DiT-S/2 structural distance vs cross-arch spectrum"
echo "  2. fp32 reference precision ablation (SD 1.5 + FLUX)"
echo "  3. Plan B 22→121 pair extension"
echo "  4. Bootstrap CI for distance matrix"
echo "  5. Normalization ablation (z-score/L2/LayerNorm)"
echo "============================================================"

# ---- Task 1: DiT-S/2 structural distance (CPU, ~minutes) ----
echo ""
echo "=== TASK 1: DiT-S/2 Structural Distance ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase9_task1_dit_distance.py
echo "Task 1 done: $(date)"

# ---- Task 5: Normalization ablation (CPU, ~minutes) ----
echo ""
echo "=== TASK 5: Normalization Ablation ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase9_task5_norm_ablation.py
echo "Task 5 done: $(date)"

# ---- Task 4 Step 1: SD 1.5 per-image drift extraction (GPU, ~30 min) ----
echo ""
echo "=== TASK 4a: SD 1.5 Per-Image Drift Extraction ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase9_task4_sd15_perimage.py
echo "Task 4a done: $(date)"

# ---- Task 4 Step 2: Bootstrap CI (CPU, ~hours) ----
echo ""
echo "=== TASK 4b: Bootstrap Distance CI ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase9_task4_bootstrap.py
echo "Task 4b done: $(date)"

# ---- Task 2: fp32 precision ablation (GPU, ~half day) ----
echo ""
echo "=== TASK 2: fp32 Precision Ablation ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase9_task2_fp32_ablation.py
echo "Task 2 done: $(date)"

# ---- Task 3: Plan B 121 pairs (GPU, ~1 day) ----
echo ""
echo "=== TASK 3: Plan B 121-Pair Extension ==="
echo "Started: $(date)"
conda run -n grad python scripts/phase7_planb.py
echo "Task 3 done: $(date)"

echo ""
echo "============================================================"
echo "ALL TASKS COMPLETE — $(date)"
echo "============================================================"
