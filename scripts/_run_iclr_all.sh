#!/bin/bash
cd /home/hiaskc/Talant/graduation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=scripts:$PYTHONPATH

echo "============================================"
echo " ICLR Sequential Runner v2"
echo " Start: $(date)"
echo " GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"
echo "============================================"

cleanup_gpu() {
    echo "  [cleanup] Freeing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache(); print(f'    GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')" 2>/dev/null
    sleep 3
}

# --- Task A: Cross-Prompt Validation ---
echo ""
echo "#############################################"
echo " TASK A: Cross-Prompt (SD 1.5, 25 prompts)"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_cross_prompt.py --prompts 25 --skip-drift 2>&1 | tee outputs/phase8_iclr_cross_prompt/run.log
EXIT_A=$?
echo "Task A exit: $EXIT_A at $(date)"

cleanup_gpu

# --- Task B: Editing Validation ---
echo ""
echo "#############################################"
echo " TASK B: Editing (SD 1.5 + P2P, 25 tasks)"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_editing.py --steps 20 2>&1 | tee outputs/phase8_iclr_editing/run.log
EXIT_B=$?
echo "Task B exit: $EXIT_B at $(date)"

cleanup_gpu

# --- Task C: SDXL Cross-Architecture ---
echo ""
echo "#############################################"
echo " TASK C: SDXL Cross-Architecture"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_sdxl.py --prompts 20 --skip-dose --steps 30 2>&1 | tee outputs/phase8_iclr_sdxl/run.log
EXIT_C=$?
echo "Task C exit: $EXIT_C at $(date)"

echo ""
echo "============================================"
echo " ALL DONE at $(date)"
echo " Exit codes: A=$EXIT_A B=$EXIT_B C=$EXIT_C"
echo "============================================"
