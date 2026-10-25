#!/usr/bin/env bash
# One-command retraining on Linux / WSL: links the existing HFValid data,
# activates the training environment, checks the GPU and runs
# scripts/run_retraining.py with any arguments given here.
#
#   bash scripts/train_wsl.sh --also-ap
#
# Defaults match the local setup; override with environment variables:
#   VENV=~/bonevision-femur/femur3d   DATA=~/bonevision-femur/data
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/bonevision-femur/femur3d}"
DATA="${DATA:-$HOME/bonevision-femur/data}"
cd "$REPO"

echo "== Data"
for d in raw occupancy_highres; do
  if [ ! -e "data/$d" ]; then
    [ -d "$DATA/$d" ] || { echo "Missing $DATA/$d (set DATA=...)"; exit 1; }
    ln -s "$DATA/$d" "data/$d"
    echo "linked data/$d -> $DATA/$d"
  fi
done
echo "subjects: $(ls data/raw/Subjects | wc -l)   targets: $(ls data/occupancy_highres/*.npy | wc -l)"

echo "== Environment ($VENV)"
[ -f "$VENV/bin/activate" ] || { echo "No virtualenv at $VENV (set VENV=...)"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -r requirements-train.txt
python - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the GPU")
x = torch.randn(1000, 1000, device="cuda")
torch.cuda.synchronize()
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "|", (x @ x).sum().item() != 0)
EOF

echo "== Pipeline"
mkdir -p logs
python scripts/run_retraining.py "$@" 2>&1 | tee "logs/pipeline_$(date +%Y%m%d_%H%M).log"
