#!/usr/bin/env bash
# Install routined as a recurring job on the host.
#
# - On Linux/macOS with cron available: installs an @reboot entry that
#   keeps the daemon alive plus a watchdog that re-launches it if it
#   crashed. Idempotent — re-running just rewrites the lines we own.
# - On systemd-only hosts, pass --systemd to install a user unit instead.
# - On Windows, see install.ps1 (Task Scheduler).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROUTINED_PYTHON:-python3}"
CONFIG="${ROUTINED_CONFIG:-$HERE/config.yaml}"
LOG="${ROUTINED_LOG:-$HOME/.routined/daemon.log}"
PIDFILE="${ROUTINED_PIDFILE:-$HOME/.routined/routined.pid}"
MARK_BEGIN="# >>> routined managed >>>"
MARK_END="# <<< routined managed <<<"

mode="cron"
for arg in "$@"; do
  case "$arg" in
    --systemd) mode="systemd" ;;
    --uninstall) mode="uninstall" ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
  esac
done

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found at $CONFIG — copy config.example.yaml to config.yaml first." >&2
  exit 1
fi

run_cmd="\"$PYTHON\" \"$HERE/routined.py\" --config \"$CONFIG\""
watchdog="if ! pgrep -f \"routined.py --config $CONFIG\" >/dev/null; then nohup $run_cmd >>\"$LOG\" 2>&1 & echo \$! > \"$PIDFILE\"; fi"

case "$mode" in
  cron)
    tmp="$(mktemp)"
    crontab -l 2>/dev/null | sed "/$MARK_BEGIN/,/$MARK_END/d" > "$tmp"
    {
      echo "$MARK_BEGIN"
      echo "@reboot $watchdog"
      echo "*/5 * * * * $watchdog"
      echo "$MARK_END"
    } >> "$tmp"
    crontab "$tmp"
    rm -f "$tmp"
    echo "installed cron entries. starting now…"
    bash -c "$watchdog"
    echo "logs: $LOG"
    ;;

  systemd)
    unit_dir="$HOME/.config/systemd/user"
    mkdir -p "$unit_dir"
    cat > "$unit_dir/routined.service" <<EOF
[Unit]
Description=routined — agentic routine daemon
After=network-online.target

[Service]
ExecStart=$run_cmd
Restart=always
RestartSec=15
StandardOutput=append:$LOG
StandardError=append:$LOG

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now routined.service
    echo "installed user unit. status: systemctl --user status routined"
    ;;

  uninstall)
    if command -v crontab >/dev/null; then
      tmp="$(mktemp)"
      crontab -l 2>/dev/null | sed "/$MARK_BEGIN/,/$MARK_END/d" > "$tmp" || true
      crontab "$tmp"
      rm -f "$tmp"
    fi
    if command -v systemctl >/dev/null; then
      systemctl --user disable --now routined.service 2>/dev/null || true
      rm -f "$HOME/.config/systemd/user/routined.service"
      systemctl --user daemon-reload 2>/dev/null || true
    fi
    if [[ -f "$PIDFILE" ]]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null || true
      rm -f "$PIDFILE"
    fi
    echo "uninstalled."
    ;;
esac
