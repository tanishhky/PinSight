# PinSight launchd Automation

Three weekday jobs run automatically on macOS via launchd:

| Job | Time (ET) | Action |
|---|---|---|
| morning | 09:35 Mon–Fri | Pull nearest-expiry chain, persist, inspect |
| midday  | 12:30 Mon–Fri | Pull same expiry again, inspect (flow evolution) |
| close   | 16:10 Mon–Fri | Pull final snapshot, run `eval-flags` against actual close |

The runner script (`pinsight-runner.sh`) skips US-market holidays via a
hardcoded calendar (refresh yearly in January). Weekend filtering is
handled by launchd (Weekday 1–5).

## Install

```
./scripts/launchd/install.sh
```

This copies the three plists to `~/Library/LaunchAgents/` and `launchctl
load`s each. Re-running is idempotent (unload then load).

## Verify

```
launchctl list | grep pinsight
```

Should show three entries.

## Logs

- `logs/launchd-{morning,midday,close}.out` — stdout from each run
- `logs/launchd-{morning,midday,close}.err` — stderr
- `logs/{api,persist,fit,run,error}-YYYY-MM-DD.jsonl` — structured events
  from inside the Python process

## Uninstall

```
./scripts/launchd/uninstall.sh
```

## Adjusting

To change the underlying, export `SYMBOL` in the plist's `EnvironmentVariables` dict:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>SYMBOL</key>
    <string>QQQ</string>
    <key>PATH</key>
    <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
</dict>
```

Then reinstall.

## Caveats

- launchd uses the machine's local time. This Mac is `America/New_York`,
  so times above are ET. If you travel to a different timezone, the
  schedule shifts with you — adjust plists if that's a problem.
- If the laptop is asleep at the scheduled time, launchd fires the missed
  job when the machine wakes (it doesn't skip).
- The runner only checks the holiday list, not early-close days (day
  after Thanksgiving, etc.). On half-days the 16:10 close eval will fire
  ~3 hours after the actual close — still works, just delayed.
