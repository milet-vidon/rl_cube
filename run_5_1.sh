#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/activate_5_1.sh"
exec "${PROJECT_ROOT}/revotron_cube_rl/run_linux.sh" "$@"
