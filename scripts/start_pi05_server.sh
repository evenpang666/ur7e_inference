#!/usr/bin/env bash
set -euo pipefail

# Run this from the official openpi repository on 192.168.124.15.
# This script must be run from Sci-VLA/third_party/openpi.
POLICY_CONFIG="${POLICY_CONFIG:-mani_real_pi05}"
: "${POLICY_DIR:?Set POLICY_DIR to the trained mani_real_pi05 checkpoint directory}"

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${POLICY_DIR}" \
  --port=8000
