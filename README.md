# NCS2 Lab — Inference with Intel Neural Compute Stick 2

A reproducible setup for running computer-vision models on an **Intel Neural Compute Stick 2 (NCS2)** plugged into a **Raspberry Pi**, using **OpenVINO 2022.3**.

> **Tested on:** Debian 13 (trixie) with Raspberry Pi kernel `6.12.75+rpt-rpi-v8`, aarch64, system Python 3.13, Python 3.9 via `uv` for OpenVINO. May 2026.

---

## Why this is tricky

Intel dropped NCS2 support after OpenVINO 2022.3. To use the stick you need that exact version — and that version:

- Only supports **Python 3.7-3.10**, but Debian 13 ships Python 3.13 by default.
- The official `install_openvino_dependencies.sh` does not recognize Debian 13 (`Unsupported OS: debian13`).
- The Python bindings are `.so` files compiled for specific versions (3.7 and 3.9) — they will not load on 3.13 even with `PYTHONPATH` tricks.

The fix is to **install Python 3.9 alongside the system** (using `uv`), create a venv there, and copy the OpenVINO bindings into that venv's `site-packages`. The native C++ libraries are loaded via `LD_LIBRARY_PATH`, which `setupvars.sh` sets up.

---

## Hardware and software

| Component | Version |
|---|---|
| Raspberry Pi | aarch64 (tested on RPi 4) |
| OS | Debian 13 (trixie), RPi kernel `+rpt-rpi-v8` |
| System Python | 3.13 — untouched |
| Venv Python (for OpenVINO) | 3.9, installed via `uv` |
| OpenVINO | 2022.3.2 (official ARM64 build) |
| Stick | Intel NCS2 (Movidius Myriad X, USB ID `03e7:2485`) |
| `uv` | any recent version |

---

## Verify the stick is plugged in

```bash
lsusb | grep 03e7
```

Expected:

```
Bus 001 Device 003: ID 03e7:2485 Intel Movidius MyriadX
```

> The `Device` number changes after the first OpenVINO initialization (e.g. from `003` to `006`). This is normal — the stick boots into a bootloader, OpenVINO uploads the Myriad X firmware on first use, and the USB device re-enumerates.

---

## One-time setup

### 1. Download and extract OpenVINO

```bash
mkdir -p ~/Desktop/ncs2/openvino_2022.3
cd ~/Desktop/ncs2/openvino_2022.3

wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.2/linux/l_openvino_toolkit_debian9_2022.3.2.9279.e2c7e4d7b4d_arm64.tgz \
    -O openvino_2022.3.tgz

tar -xzf openvino_2022.3.tgz --strip-components=1
```

### 2. Install system dependencies

The official installer fails on Debian 13, so install them manually:

```bash
sudo apt install -y \
    cmake \
    libusb-1.0-0-dev \
    libomp-dev \
    python3-dev \
    pkg-config \
    libgtk-3-dev
```

### 3. Install NCS2 udev rules

```bash
cd ~/Desktop/ncs2/openvino_2022.3
sudo ./install_dependencies/install_NCS_udev_rules.sh
```

**Unplug and re-plug the stick** so the new rules apply.

### 4. Install Python 3.9 with `uv` (does not touch system Python)

```bash
uv python install 3.9
```

### 5. Create the venv and copy OpenVINO bindings

```bash
uv venv ~/Desktop/ncs2/env --python 3.9
source ~/Desktop/ncs2/env/bin/activate

SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp -r ~/Desktop/ncs2/openvino_2022.3/python/python3.9/openvino "$SITE/"
cp -r ~/Desktop/ncs2/openvino_2022.3/python/python3.9/ngraph   "$SITE/"
cp    ~/Desktop/ncs2/openvino_2022.3/python/python3.9/_pyngraph.cpython-39-aarch64-linux-gnu.so "$SITE/"

uv pip install -r ~/Desktop/ncs2/openvino_2022.3/python/python3.9/requirements.txt
uv pip install opencv-python-headless
```

### 6. Create the activation script

```bash
cat > ~/Desktop/ncs2/activate.sh << 'EOF'
#!/bin/bash
source ~/Desktop/ncs2/env/bin/activate
source ~/Desktop/ncs2/openvino_2022.3/setupvars.sh 2>/dev/null
echo "✓ NCS2 env ready (Python $(python --version | awk '{print $2}'))"
EOF
chmod +x ~/Desktop/ncs2/activate.sh
```

### 7. Verify

```bash
source ~/Desktop/ncs2/activate.sh

python -c "from openvino.runtime import Core; ie = Core(); print(ie.available_devices)"
```

Expected output:

```
✓ NCS2 env ready (Python 3.9.25)
['CPU', 'MYRIAD']
```

> Ignore the `[setupvars.sh] WARNING: Unsupported Python version` message. It checks the system Python (3.13), not the venv (3.9). The native libraries load fine.

---

## Project layout

```
~/Desktop/ncs2/
├── activate.sh                     # loads venv + native libs
├── env/                            # Python 3.9 venv
├── openvino_2022.3/                # OpenVINO runtime
├── models/                         # IR model files
│   ├── face-detection-retail-0004.xml
│   ├── face-detection-retail-0004.bin
│   └── benchmark.py
├── detect-image/
│   ├── detect_image.py
│   ├── test-images/                # input images
│   └── outputs/                    # annotated outputs
└── README.md
```

---

## Activate the environment

In every new terminal session:

```bash
source ~/Desktop/ncs2/activate.sh
```

---

## Download a model

OpenVINO uses the **IR** format: `.xml` (topology) + `.bin` (weights). The NCS2 only runs **FP16**.

```bash
mkdir -p ~/Desktop/ncs2/models
cd ~/Desktop/ncs2/models

wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-retail-0004/FP16/face-detection-retail-0004.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-retail-0004/FP16/face-detection-retail-0004.bin
```

URL pattern for any other model from the Open Model Zoo:

```
https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/{model_name}/FP16/{model_name}.{xml,bin}
```

---

## Run the benchmark

The script `models/benchmark.py` measures FPS with synthetic input — no camera, no images. Pure NCS2 stress test.

```bash
source ~/Desktop/ncs2/activate.sh
cd ~/Desktop/ncs2/models
python benchmark.py
```

Expected output:

```
Devices: ['CPU', 'MYRIAD']
MYRIAD: 44.9 FPS (22.3 ms/inference)
```

For reference: the same model on the Pi CPU (no stick) yields ~5-10 FPS.

---

## Run face detection on a real image

The script `detect-image/detect_image.py` takes an image path or URL, runs face detection on the NCS2, draws green boxes on detected faces, and saves the output.

**Local image:**

```bash
source ~/Desktop/ncs2/activate.sh
cd ~/Desktop/ncs2/detect-image

python detect_image.py test-images/your_image.png \
    --out outputs/your_image_output.jpg
```

**Image from URL** (GitHub raw works; Wikimedia blocks generic User-Agents):

```bash
python detect_image.py \
    https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg \
    --out outputs/lena_output.jpg
```

**Tune detection sensitivity:**

```bash
python detect_image.py image.jpg --threshold 0.3   # more permissive
python detect_image.py image.jpg --threshold 0.7   # more strict
```

---

## Mental model: where does the model live?

The most confusing part: the model does not "stay" on the Pi — it gets uploaded to the NCS2 and lives there.

```
┌─── Raspberry Pi ────┐                  ┌── NCS2 (USB stick) ──┐
│                     │                  │                      │
│  Disk               │                  │                      │
│  .xml + .bin        │                  │   Onboard memory     │
│      │              │   ② compile      │   FP16 model         │
│      │ ① read_model │  ─────USB──────► │   (lives here)       │
│      ▼              │   (once, ~3s)    │                      │
│  RAM (Python)       │                  │                      │
│  I/O tensors        │   ③ infer        │   Myriad X chip      │
│       ◄──USB──────► │  ◄──USB────────► │   16 SHAVE cores     │
│                     │   (each ~22ms)   │                      │
└─────────────────────┘                  └──────────────────────┘
```

**Three phases:**

1. **`read_model()`** — Pi reads `.xml` + `.bin` from disk into RAM. Local, milliseconds. The stick is not involved.
2. **`compile_model("MYRIAD")`** — OpenVINO converts the model to FP16, uploads it via USB to the stick, the stick stores it in onboard memory. Takes ~3-5 s. **One-time only.**
3. **`infer.infer({0: x})`** — Each inference only ships the input tensor (~1 MB) over USB. The model is already on the stick, so it is not re-uploaded. ~22 ms per call.

**Practical implication:** in any video pipeline, call `compile_model()` **once outside the loop** and reuse the same `infer` request inside. Compiling per frame drops you from 45 FPS to 0.3 FPS.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'openvino._pyopenvino'` | Running system Python 3.13 instead of venv 3.9 | `source ~/Desktop/ncs2/activate.sh` |
| `ImportError: libopenvino.so.2232: cannot open shared object file` | Venv active but `setupvars.sh` not sourced | Run `activate.sh`, not `env/bin/activate` directly |
| `available_devices` shows only `['CPU']` | Stick not initialized | Check `lsusb`, run udev rules, unplug + replug |
| `urllib.error.HTTPError: 403 / 400` on Wikimedia URLs | Wikimedia blocks generic User-Agents | Use GitHub raw URLs or `wget` to a local file |
| `[setupvars.sh] WARNING: Unsupported Python version` | Cosmetic — checks system Python, not venv | Ignore |

---

## Next steps

Models from the Open Model Zoo worth trying (all FP16 for NCS2):

| Model | Use case |
|---|---|
| `face-detection-adas-0001` | Smaller faces, better recall than retail-0004 |
| `person-detection-retail-0013` | Whole-body person detection |
| `pedestrian-and-vehicle-detector-adas-0001` | Pedestrians + vehicles |
| `human-pose-estimation-0001` | Skeleton/keypoint detection |
| `emotion-recognition-retail-0003` | Combines with face-detection: emotion classification |

Most are drop-in replacements via the `--model` flag in `detect_image.py`. Adjust the input shape (some require 256×256 or 672×384 instead of 300×300) and parse the output format from the `.xml` if the layout differs from `[1, 1, N, 7]`.

---

**Setup tested on 2026-05-08 with: Raspberry Pi (aarch64) + Debian 13 + RPi kernel `6.12.75+rpt-rpi-v8` + Python 3.13 (system) / 3.9 (venv) + OpenVINO 2022.3.2 + NCS2.**


## Refrences
https://www.intel.com/content/www/us/en/support/articles/000057005/boards-and-kits.html
https://www.intel.com/content/www/us/en/support/articles/000055220/boards-and-kits.html
https://storage.openvinotoolkit.org/repositories