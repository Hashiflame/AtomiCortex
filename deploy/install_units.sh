#!/usr/bin/env bash
#
# AtomiCortex Systemd Unit Installation script.
#
# DO NOT RUN FROM THE REPOSITORY DIRECTORY DURING DEPLOYMENT.
# This script should be copied to /usr/local/sbin/atomicortex-install-units
# and owned by root. The deployment pipeline will call it via sudo -n.
#
# REQUIRED sudoers configuration:
# hashiflame ALL=(ALL) NOPASSWD: /usr/local/sbin/atomicortex-install-units
#
# WHAT IT INSTALLS
#   Exactly the units named in deploy/units.enabled — not everything that
#   happens to sit in deploy/. The repository keeps units we deliberately do
#   not run (the 15m bot, the watchdogs, the API, the reconciler); copying
#   them all and then demanding they be active is how this script used to
#   fail on a VM where only two services were ever meant to run.
#
# HOW IT CLASSIFIES
#   By the unit's own Type= directive, never by its name. Type=simple (or a
#   missing Type=, which systemd itself treats as simple) means long-running:
#   restarted, then required to be active. Anything else — oneshot services
#   driven by a timer — is installed and left alone.
#
# WHY IT CREATES DIRECTORIES
#   Under ProtectSystem=strict a path listed in ReadWritePaths= that does not
#   exist makes systemd abort the unit with 226/NAMESPACE *before* exec, with
#   no application output at all. logs/ is gitignored, so a fresh checkout
#   hits exactly that. Paths prefixed with '-' are optional by systemd's own
#   rules and are left alone.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (or with sudo)"
  exit 1
fi

REPO_DIR="/home/hashiflame/AtomiCortex/deploy"
TARGET_DIR="/etc/systemd/system"
MANIFEST="$REPO_DIR/units.enabled"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# First value of a directive inside the [Service] section of a unit file.
# A line-by-line section walk, not grep: the same key can legitimately
# appear in another section or inside a comment.
unit_value() {
    local file="$1" key="$2"
    awk -v key="$key" '
        /^\[/            { section = $0; next }
        section == "[Service]" && index($0, key) == 1 {
            sub("^" key, "", $0); print $0; exit
        }
    ' "$file"
}

# ---------------------------------------------------------------------------
# 1. Read the manifest
# ---------------------------------------------------------------------------

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    echo "       Nothing can be installed without it — refusing to guess."
    exit 1
fi

UNITS=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    if [ -z "$line" ]; then
        continue
    fi
    UNITS+=("$line")
done < "$MANIFEST"

if [ "${#UNITS[@]}" -eq 0 ]; then
    echo "ERROR: manifest lists no units: $MANIFEST"
    exit 1
fi

# A name with no file behind it is a typo, and a typo must not degrade into
# a partial install that then passes its own health-check.
for unit in "${UNITS[@]}"; do
    if [ ! -f "$REPO_DIR/$unit" ]; then
        echo "ERROR: manifest names '$unit' but $REPO_DIR/$unit does not exist"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 2. Copy, recording what was actually installed
# ---------------------------------------------------------------------------

echo "==> Installing ${#UNITS[@]} unit(s) listed in units.enabled..."
SERVICES=()
TIMERS=()
for unit in "${UNITS[@]}"; do
    cp "$REPO_DIR/$unit" "$TARGET_DIR/$unit"
    echo "    installed $unit"
    case "${unit##*.}" in
        service) SERVICES+=("$unit") ;;
        timer)   TIMERS+=("$unit") ;;
        *)
            echo "ERROR: manifest entry '$unit' is neither a service nor a timer"
            exit 1
            ;;
    esac
done

echo "==> Reloading daemon..."
systemctl daemon-reload

# ---------------------------------------------------------------------------
# 3. Create the directories the installed units declare writable
# ---------------------------------------------------------------------------

echo "==> Ensuring ReadWritePaths directories exist..."
for unit in ${SERVICES[@]+"${SERVICES[@]}"}; do
    file="$TARGET_DIR/$unit"
    owner="$(unit_value "$file" 'User=')"
    group="$(unit_value "$file" 'Group=')"
    paths="$(unit_value "$file" 'ReadWritePaths=')"
    if [ -z "$paths" ]; then
        continue
    fi
    # Deliberately unquoted: ReadWritePaths= is a space-separated list.
    for rwp in $paths; do
        if [ "${rwp#-}" != "$rwp" ]; then
            echo "    skipping optional $rwp"
            continue
        fi
        # 775, not 755: install -d applies the mode to an existing directory
        # instead of leaving it alone, and logs/ and data/ on the VM are
        # already drwxrwxr-x. 755 would silently drop group write.
        install -d -o "${owner:-root}" -g "${group:-root}" -m 775 "$rwp"
        echo "    ensured $rwp"
    done
done

# ---------------------------------------------------------------------------
# 4. Classify by Type=
# ---------------------------------------------------------------------------

LONG_RUNNING=()
for unit in ${SERVICES[@]+"${SERVICES[@]}"}; do
    type_value="$(unit_value "$TARGET_DIR/$unit" 'Type=')"
    if [ -z "$type_value" ] || [ "$type_value" = "simple" ]; then
        LONG_RUNNING+=("$unit")
    else
        echo "    $unit is Type=$type_value — installed, not started directly"
    fi
done

# Without this, a broken parser above would empty the list and every check
# below would pass by checking nothing.
if [ "${#SERVICES[@]}" -gt 0 ] && [ "${#LONG_RUNNING[@]}" -eq 0 ]; then
    echo "ERROR: no long-running service among the installed ones —"
    echo "       the Type= classifier or the manifest is wrong"
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Restart — soft: a failure here is reported, not fatal
# ---------------------------------------------------------------------------

echo "==> Restarting long-running services..."
for unit in ${LONG_RUNNING[@]+"${LONG_RUNNING[@]}"}; do
    echo "    restarting $unit"
    if ! systemctl restart "$unit"; then
        echo "WARNING: failed to restart $unit"
    fi
done

# ---------------------------------------------------------------------------
# 6. Enable timers — soft here, verified below
# ---------------------------------------------------------------------------

echo "==> Enabling timers..."
for unit in ${TIMERS[@]+"${TIMERS[@]}"}; do
    echo "    enabling $unit"
    if ! systemctl enable --now "$unit"; then
        echo "WARNING: failed to enable $unit"
    fi
done

# ---------------------------------------------------------------------------
# 7. Health-check — hard, and complete
# ---------------------------------------------------------------------------

echo "==> Performing health-check..."
FAILED=()

for unit in ${LONG_RUNNING[@]+"${LONG_RUNNING[@]}"}; do
    if ! systemctl is-active --quiet "$unit"; then
        FAILED+=("$unit is not active")
    fi
done

for unit in ${TIMERS[@]+"${TIMERS[@]}"}; do
    if ! systemctl is-active --quiet "$unit"; then
        FAILED+=("$unit is not active")
    fi
    if ! systemctl is-enabled --quiet "$unit"; then
        FAILED+=("$unit is not enabled")
    fi
done

# Every failure at once: the old version exited on the first one, so an
# operator learned about one broken unit per deployment.
if [ "${#FAILED[@]}" -ne 0 ]; then
    echo "ERROR: health-check failed:"
    for entry in "${FAILED[@]}"; do
        echo "    - $entry"
    done
    exit 1
fi

echo "==> All units installed and active successfully."
echo "    services: ${#SERVICES[@]} (long-running: ${#LONG_RUNNING[@]}), timers: ${#TIMERS[@]}"
