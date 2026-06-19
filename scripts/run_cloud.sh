#!/usr/bin/env bash
# Cloud / multi-GPU run for a real result. Edit CONFIG below to choose scale:
#   config/cifar10_small.yaml  -> full CIFAR-10, modest stack, short pretrain
#   config/cifar100_full.yaml  -> full CIFAR-100, deep stack, long pretrain (heavy)
#
# Runs the SIGReg ablation too (with vs without SIGReg) so you can see the
# collapse-prevention story in the probe accuracies.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/cifar10_small.yaml}"
echo "Using config: $CONFIG"

# --- SSL pretrain WITH SIGReg (the main run) ---
echo "=== [1/4] SSL pretrain WITH SIGReg ==="
python3 train_ssl.py --config "$CONFIG"
CP_WITH="runs/$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['run_name'])")/checkpoints/ssl_final.pt"

# --- SSL pretrain WITHOUT SIGReg (ablation: expect collapse -> worse probe) ---
echo ""
echo "=== [2/4] SSL pretrain WITHOUT SIGReg (ablation) ==="
python3 train_ssl.py --config "$CONFIG" --no-sigreg
CP_WITHOUT="runs/$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['run_name'])")_nosigreg/checkpoints/ssl_final.pt"

# --- Linear probe both ---
echo ""
echo "=== [3/4] Linear probe (WITH SIGReg) ==="
python3 linear_probe.py --config "$CONFIG" --checkpoint "$CP_WITH"

echo ""
echo "=== [4/4] Linear probe (WITHOUT SIGReg -- ablation) ==="
python3 linear_probe.py --config "$CONFIG" --checkpoint "$CP_WITHOUT"

echo ""
echo "Done. Compare the two probe accuracies: WITH SIGReg should beat WITHOUT."
