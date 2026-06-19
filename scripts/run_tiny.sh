#!/usr/bin/env bash
# Local correctness / smoke run on CPU or a single weak GPU. Minutes.
# Proves the full pipeline (HOPE + per-layer JEPA + SIGReg) trains end-to-end
# and the linear probe runs. NOT for final accuracy -- use run_cloud.sh for that.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/3] Running test suite (shapes, overfit, sigreg, smoke) ==="
python3 -m tests.test_shapes
python3 -m tests.test_overfit_batch
python3 -m tests.test_sigreg
python3 -m tests.test_smoke

echo ""
echo "=== [2/3] SSL pretraining (tiny config, CIFAR-10, 2000 images, 3 epochs) ==="
python3 train_ssl.py --config config/tiny.yaml

echo ""
echo "=== [3/3] Linear-probe evaluation ==="
python3 linear_probe.py --config config/tiny.yaml \
    --checkpoint runs/tiny/checkpoints/ssl_final.pt

echo ""
echo "Done. For a real run (meaningful accuracy), use scripts/run_cloud.sh."
