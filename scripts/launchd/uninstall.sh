#!/usr/bin/env bash
# Stop and remove PinSight launchd jobs.
set -euo pipefail

DST="$HOME/Library/LaunchAgents"
for label in morning midday close; do
    plist="$DST/com.tanishk.pinsight.${label}.plist"
    if [[ -f "$plist" ]]; then
        launchctl unload "$plist" 2>/dev/null || true
        rm -f "$plist"
        echo "Removed $label"
    fi
done
echo "Done."
