#!/usr/bin/env bash
# Minimal spot TPU VM helper.
#   scripts/tpu.sh up              create spot VM (tries zones in order)
#   scripts/tpu.sh sync            rsync repo to VM and install env
#   scripts/tpu.sh run "<cmd>"     run a command in the repo on the VM
#   scripts/tpu.sh fetch           copy results/ back
#   scripts/tpu.sh down            delete the VM (stops billing)
set -euo pipefail

NAME="${TPU_NAME:-humming-tpu}"
ACCEL="${TPU_ACCEL:-v6e-1}"
VERSION="${TPU_VERSION:-v2-alpha-tpuv6e}"
ZONES="${TPU_ZONES:-us-east4-a us-east1-d us-east5-b}"
ZONE_FILE=".tpu_zone"
REPO_DIR="humming-tpu"

zone() { cat "$ZONE_FILE"; }
ssh_vm() { gcloud compute tpus tpu-vm ssh "$NAME" --zone "$(zone)" --command "$1"; }

case "${1:-}" in
  up)
    for z in $ZONES; do
      echo "trying $z ..."
      if gcloud compute tpus tpu-vm create "$NAME" --zone "$z" \
           --accelerator-type "$ACCEL" --version "$VERSION" --spot --quiet; then
        echo "$z" > "$ZONE_FILE"; echo "up in $z"; exit 0
      fi
    done
    echo "no capacity in: $ZONES" >&2; exit 1 ;;
  sync)
    gcloud compute tpus tpu-vm scp --zone "$(zone)" --recurse \
      pyproject.toml uv.lock humming_tpu bench tests "$NAME:~/$REPO_DIR/"
    ssh_vm "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh;
            cd ~/$REPO_DIR && ~/.local/bin/uv sync -q && ~/.local/bin/uv pip install -q 'jax[tpu]==0.11.2'" ;;
  run)
    ssh_vm "cd ~/$REPO_DIR && ~/.local/bin/uv run --no-sync ${2:?command}" ;;
  fetch)
    mkdir -p results
    gcloud compute tpus tpu-vm scp --zone "$(zone)" --recurse "$NAME:~/$REPO_DIR/results/*" results/ ;;
  down)
    gcloud compute tpus tpu-vm delete "$NAME" --zone "$(zone)" --quiet && rm -f "$ZONE_FILE" ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
