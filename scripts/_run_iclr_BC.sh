#!/bin/bash
cd /home/hiaskc/Talant/graduation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=scripts:$PYTHONPATH

cleanup_gpu() {
    echo "  [cleanup] Freeing GPU..."
    python -c "import torch; torch.cuda.empty_cache(); print(f'    free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')"
    sleep 3
}

echo "============================================"
echo " ICLR B+C Runner"
echo " Start: $(date)"
echo " GPU: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"
echo "============================================"

echo ""
echo "### TASK B: Editing Validation (SD 1.5 + P2P, 20 steps) ###"
echo "Start: $(date)"
python -u scripts/phase8_iclr_editing.py --steps 20 2>&1 | tee outputs/phase8_iclr_editing/run.log
echo "Task B exit: $? at $(date)"
cleanup_gpu

echo ""
echo "### TASK C: SDXL Cross-Architecture (30 steps) ###"
echo "Start: $(date)"
python -u scripts/phase8_iclr_sdxl.py --prompts 20 --skip-dose --steps 30 2>&1 | tee outputs/phase8_iclr_sdxl/run.log
echo "Task C exit: $? at $(date)"

echo ""
echo "ALL DONE at $(date)"
