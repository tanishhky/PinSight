#!/usr/bin/env bash
# PinSight scheduled runner.
# Called by launchd with one argument: morning | midday | close
#
# Refuses to run on US-market holidays via a hardcoded calendar; refresh
# yearly. Weekdays are gated by launchd (Mon-Fri only).

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PROJECT}/.venv/bin/python"
SYMBOL="${SYMBOL:-SPY}"

mode="${1:-morning}"

# US market holidays — 2026 + 2027 (NYSE schedule).
# Refresh this list each January.
HOLIDAYS=(
    # 2026
    "2026-01-01"  # New Year's Day
    "2026-01-19"  # MLK Day
    "2026-02-16"  # Presidents Day
    "2026-04-03"  # Good Friday
    "2026-05-25"  # Memorial Day
    "2026-06-19"  # Juneteenth
    "2026-07-03"  # July 4 (observed)
    "2026-09-07"  # Labor Day
    "2026-11-26"  # Thanksgiving
    "2026-12-25"  # Christmas
    # 2027
    "2027-01-01"
    "2027-01-18"
    "2027-02-15"
    "2027-03-26"
    "2027-05-31"
    "2027-06-18"  # Juneteenth observed
    "2027-07-05"  # July 4 observed
    "2027-09-06"
    "2027-11-25"
    "2027-12-24"  # Christmas observed
)

today="$(date +%Y-%m-%d)"
for h in "${HOLIDAYS[@]}"; do
    if [[ "$today" == "$h" ]]; then
        echo "[$(date -Is)] $mode skipped: $today is a US market holiday" >&2
        exit 0
    fi
done

cd "$PROJECT"

case "$mode" in
    morning)
        "$PY" -m pinsight.cli monday-workflow "$SYMBOL"
        ;;
    midday)
        "$PY" -m pinsight.cli fetch-chain "$SYMBOL"
        "$PY" -m pinsight.cli inspect-chain "$SYMBOL" "$(date +%Y-%m-%d)"
        ;;
    close)
        "$PY" -m pinsight.cli monday-workflow "$SYMBOL" --eval
        ;;
    *)
        echo "Usage: $0 {morning|midday|close}" >&2
        exit 2
        ;;
esac
