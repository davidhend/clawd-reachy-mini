#!/usr/bin/env bash
# Bring-up script for the Reachy <-> OpenClaw bridge on a dedicated headless
# Ubuntu host. Idempotent: re-runs are safe.
#
# Assumes the repo is checked out at /opt/reachy-openclaw and the dedicated
# service account is `reachy`. Run as root.

set -euo pipefail

INSTALL_DIR=${INSTALL_DIR:-/opt/reachy-openclaw}
SERVICE_USER=${SERVICE_USER:-reachy}
CONFIG_DIR=/etc/reachy-openclaw
PIPER_VOICE=${PIPER_VOICE:-en_US-lessac-medium}
WHISPER_MODEL=${WHISPER_MODEL:-base}

if [[ $EUID -ne 0 ]]; then
    echo "Run as root (needs to create user, systemd units, udev rules)." >&2
    exit 1
fi

echo "==> Installing system packages (GStreamer + ALSA + Python venv tooling)"
# Ubuntu 24.04 ships GStreamer 1.24+, so no PPA is needed. If you ever rebuild
# this on 22.04, you'll need `add-apt-repository ppa:savoury1/multimedia` first.
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-pip \
    autossh \
    git pkg-config \
    alsa-utils \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libglib2.0-dev libssl-dev \
    libgirepository1.0-dev libcairo2-dev \
    libportaudio2 libnice10 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    gstreamer1.0-tools \
    python3-gi python3-gi-cairo
# Note: despite the name, gstreamer1.0-tools ships no gst-device-monitor
# binary — inspect devices from Python via Gst.DeviceMonitor if needed.

echo "==> Ensuring service user '$SERVICE_USER' exists"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi
# audio for ALSA/PipeWire, dialout for any USB serial the SDK opens,
# video for /dev/video* (camera detection + the reachy-camera service).
usermod -aG audio,dialout,video "$SERVICE_USER"

# Lingering keeps the user's systemd --user instance + PipeWire alive without
# an interactive login. Required for audio under PipeWire.
loginctl enable-linger "$SERVICE_USER"

echo "==> Building gst-plugins-rs webrtcsink (Rust WebRTC plugin)"
# The daemon's media server hard-requires the webrtcsink element, which apt
# does not ship on Linux — it must be built from gst-plugins-rs. Without it
# the media server dies and the SDK gets no IPC/WebRTC endpoint at all.
# The services find the plugin via the GST_PLUGIN_PATH systemd drop-ins
# installed below (a ~/.bashrc export does NOT reach systemd services).
# Docs: https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation
GST_PLUGINS_RS_TAG=${GST_PLUGINS_RS_TAG:-0.14.5}  # 0.14.5 has a webrtcsink deadlock fix
GST_PLUGINS_RS_PREFIX=/opt/gst-plugins-rs
export GST_PLUGIN_PATH="$GST_PLUGINS_RS_PREFIX/lib/x86_64-linux-gnu"
if gst-inspect-1.0 webrtcsink >/dev/null 2>&1; then
    echo "    webrtcsink already available — skipping build"
else
    echo "    NOTE: first build takes 30-60+ min on a small CPU"
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    command -v cargo-cinstall >/dev/null 2>&1 || cargo install cargo-c
    GST_RS_SRC=/opt/src/gst-plugins-rs
    if [[ ! -d "$GST_RS_SRC" ]]; then
        install -d /opt/src
        git clone --depth 1 --branch "$GST_PLUGINS_RS_TAG" \
            https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git "$GST_RS_SRC"
    fi
    (cd "$GST_RS_SRC" && cargo cinstall -p gst-plugin-webrtc \
        --prefix="$GST_PLUGINS_RS_PREFIX" --release)
    # Fail loudly now rather than mysteriously at daemon start.
    gst-inspect-1.0 webrtcsink >/dev/null
fi

echo "==> Provisioning $CONFIG_DIR"
install -d -m 750 -o root -g "$SERVICE_USER" "$CONFIG_DIR"
for f in tunnel.env bridge.env daemon.env; do
    if [[ ! -f "$CONFIG_DIR/$f" ]]; then
        install -m 640 -o root -g "$SERVICE_USER" \
            "$INSTALL_DIR/deploy/env/${f}.example" "$CONFIG_DIR/$f"
        echo "    wrote $CONFIG_DIR/$f (edit before starting services)"
    fi
done

echo "==> Generating SSH tunnel key (if missing)"
if [[ ! -f "$CONFIG_DIR/tunnel_key" ]]; then
    # Generate as root: $CONFIG_DIR is mode 750 root:$SERVICE_USER, so the
    # service user can read/traverse it but not create files in it. Root can,
    # then we hand ownership of the key to the service user below.
    ssh-keygen -t ed25519 -N "" \
        -f "$CONFIG_DIR/tunnel_key" -C "reachy-openclaw-tunnel"
    chown "$SERVICE_USER:$SERVICE_USER" "$CONFIG_DIR/tunnel_key"*
    chmod 600 "$CONFIG_DIR/tunnel_key"
    echo
    echo "    Add this public key to the OpenClaw host user's ~/.ssh/authorized_keys"
    echo "    with the restriction prefix from deploy/scripts/authorized_keys.snippet:"
    echo
    cat "$CONFIG_DIR/tunnel_key.pub"
    echo
fi

echo "==> Installing Python deps under $INSTALL_DIR/.venv"
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    python3 -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install \
    "$INSTALL_DIR" \
    reachy-mini \
    faster-whisper \
    piper-tts
# For the reachy-camera service (YuNet/NanoDet via cv2.dnn). --no-deps keeps
# it from churning numpy & friends in the venv the daemon shares.
"$INSTALL_DIR/.venv/bin/pip" install --no-deps opencv-python-headless
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.venv"

echo "==> Patching reachy_mini camera detection (upstream issue #1248)"
# GStreamer's plain v4l2 provider (no PipeWire on this headless box) exposes
# device.path, not api.v4l2.path, so stock detection logs "No camera found".
# Idempotent; becomes a no-op once the fix from
# https://github.com/pollen-robotics/reachy_mini/issues/1248 ships in the
# installed release. Reapply by re-running this script after reachy-mini
# upgrades until then.
"$INSTALL_DIR/.venv/bin/python" - <<'PYEOF'
import pathlib
import re

import reachy_mini.media.device_detection as dd

p = pathlib.Path(dd.__file__)
src = p.read_text()
if 'props.get("api.v4l2.path")' in src:
    print(f"    already patched: {p}")
elif '"api.v4l2.path" in props' not in src:
    print(f"    pattern not found (upstream fix landed?): {p} — verify camera detection manually")
else:
    pat = re.compile(
        r'^(\s*)if "api\.v4l2\.path" in props:\n\s*device_path = props\["api\.v4l2\.path"\]\n',
        re.M,
    )
    new_src, n = pat.subn(
        lambda m: (
            f'{m.group(1)}device_path = props.get("api.v4l2.path") or props.get("device.path")\n'
            f"{m.group(1)}if device_path:\n"
        ),
        src,
    )
    if n != 1:
        raise SystemExit(f"    expected 1 patch site, found {n} in {p} — refusing to patch")
    p.with_suffix(".py.bak").write_text(src)
    p.write_text(new_src)
    print(f"    patched: {p} (backup: {p.with_suffix('.py.bak')})")
PYEOF

echo "==> Pre-caching Whisper model: $WHISPER_MODEL"
# download_root lives under the root-owned $INSTALL_DIR, so create it as the
# service user first — faster-whisper can't mkdir it itself otherwise.
install -d -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR/models/whisper"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/python" -c "
from faster_whisper import WhisperModel
WhisperModel('$WHISPER_MODEL', download_root='$INSTALL_DIR/models/whisper')
print('whisper $WHISPER_MODEL cached')
"

echo "==> Pre-caching Reachy emotion + dance datasets"
# Reads HF_TOKEN from daemon.env so the user only enters it once.
set -a; source "$CONFIG_DIR/daemon.env"; set +a
install -d -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR/hf-cache"
sudo -u "$SERVICE_USER" HF_HOME="$INSTALL_DIR/hf-cache" HF_TOKEN="$HF_TOKEN" \
    "$INSTALL_DIR/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
for ds in ('pollen-robotics/reachy-mini-emotions-library',
          'pollen-robotics/reachy-mini-dances-library'):
    p = snapshot_download(ds, repo_type='dataset')
    print(f'cached {ds} -> {p}')
"

echo "==> Downloading Piper voice: $PIPER_VOICE"
install -d -m 755 -o "$SERVICE_USER" -g "$SERVICE_USER" "$INSTALL_DIR/voices"
if [[ ! -f "$INSTALL_DIR/voices/${PIPER_VOICE}.onnx" ]]; then
    base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
    sudo -u "$SERVICE_USER" curl -fsSL -o "$INSTALL_DIR/voices/${PIPER_VOICE}.onnx" \
        "${base}${PIPER_VOICE}.onnx"
    sudo -u "$SERVICE_USER" curl -fsSL -o "$INSTALL_DIR/voices/${PIPER_VOICE}.onnx.json" \
        "${base}${PIPER_VOICE}.onnx.json"
fi

echo "==> Installing systemd units + GST_PLUGIN_PATH drop-ins"
install -m 644 "$INSTALL_DIR/deploy/systemd/"*.service /etc/systemd/system/
# The daemon (and legacy bridge) need GST_PLUGIN_PATH to find webrtcsink;
# systemd services don't read shell profiles, hence drop-ins.
for svc in reachy-daemon reachy-openclaw-bridge; do
    install -d "/etc/systemd/system/${svc}.service.d"
    install -m 644 "$INSTALL_DIR/deploy/systemd/${svc}.service.d/gst.conf" \
        "/etc/systemd/system/${svc}.service.d/gst.conf"
done
systemctl daemon-reload

echo "==> Installing udev rule"
install -m 644 "$INSTALL_DIR/deploy/udev/90-reachy-mini.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger

cat <<'EOF'

==> Setup complete. Remaining manual steps:

  1. Edit /etc/reachy-openclaw/{tunnel.env,bridge.env,daemon.env} with real
     values (OPENCLAW_TOKEN, HF_TOKEN, remote host).
  2. Add /etc/reachy-openclaw/tunnel_key.pub to the OpenClaw host's authorized_keys
     with the restriction prefix from deploy/scripts/authorized_keys.snippet.
  3. Plug Reachy in. Confirm udev applied:
         lsusb | grep -E '1a86:55d3|38fb:1001'
     Then find the audio device names and put them in bridge.env:
         arecord -L | grep -i reachy
         aplay   -L | grep -i reachy
  4. Start everything:
         sudo systemctl enable --now reachy-ssh-tunnel reachy-daemon \
             reachy-voice reachy-camera
     (reachy-openclaw-bridge is legacy — superseded by the MCP server on the
      OpenClaw box; leave it disabled.)
  5. Watch logs:
         journalctl -u reachy-voice -u reachy-camera -f
EOF
