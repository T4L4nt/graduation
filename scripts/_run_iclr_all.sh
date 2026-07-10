#!/bin/bash
cd /home/hiaskc/Talant/graduation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=scripts:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo " ICLR Sequential Runner"
echo " Start: $(date)"
echo " GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo "============================================"

# --- Task A ---
echo ""
echo "#############################################"
echo " TASK A: Cross-Prompt Validation (SD 1.5)"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_cross_prompt.py --prompts 25 --skip-drift 2>&1 | tee outputs/phase8_iclr_cross_prompt/run.log
EXIT_A=$?
echo "Task A exit code: $EXIT_A at $(date)"
echo "GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

# --- Task B ---
echo ""
echo "#############################################"
echo " TASK B: Editing Validation (SD 1.5 + P2P)"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_editing.py 2>&1 | tee outputs/phase8_iclr_editing/run.log
EXIT_B=$?
echo "Task B exit code: $EXIT_B at $(date)"
echo "GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

# --- Task C ---
echo ""
echo "#############################################"
echo " TASK C: SDXL Cross-Architecture"
echo " Start: $(date)"
echo "#############################################"
python -u scripts/phase8_iclr_sdxl.py --prompts 20 --skip-dose 2>&1 | tee outputs/phase8_iclr_sdxl/run.log
EXIT_C=$?
echo "Task C exit code: $EXIT_C at $(date)"

echo ""
echo "============================================"
echo " ALL DONE at $(date)"
echo " Exit codes: A=$EXIT_A B=$EXIT_B C=$EXIT_C"
echo "============================================"
