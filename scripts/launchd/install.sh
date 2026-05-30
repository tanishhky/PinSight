#!/usr/bin/env bash
# Install PinSight launchd jobs to ~/Library/LaunchAgents and load them.
# Idempotent: re-running will unload then reload.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/Library/LaunchAgents"
mkdir -p "$DST"

for label in morning midday close; do
    plist="com.tanishk.pinsight.${label}.plist"
    src_file="$SRC/$plist"
    dst_file="$DST/$plist"

    if launchctl list | grep -q "com.tanishk.pinsight.${label}"; then
        echo "Unloading existing $label..."
        launchctl unload "$dst_file" 2>/dev/null || true
    fi

    cp "$src_file" "$dst_file"
    launchctl load "$dst_file"
    echo "Loaded $label -> $dst_file"
done

echo
echo "Installed. Verify with:"
echo "  launchctl list | grep pinsight"
