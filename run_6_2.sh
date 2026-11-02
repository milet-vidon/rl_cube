#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/activate_6_2.sh"
exec "${PROJECT_ROOT}/revotron_cube_rl/run_linux.sh" "$@"
