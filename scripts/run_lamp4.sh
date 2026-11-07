#!/usr/bin/env bash
# LaMP-4 (news headline generation) — thin wrapper around the generic driver,
# kept so the quick-start command in the README stays valid.
#
# Usage:
#   bash scripts/run_lamp4.sh
#
# Every env var understood by scripts/run_task.sh works here too.
set -euo pipefail
exec bash "$(dirname "$0")/run_task.sh" LaMP_4
