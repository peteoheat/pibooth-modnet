# pibooth-modnet

AI background replacement and overlays for [pibooth](https://github.com/pibooth/pibooth) 2.1.x, using the
[MODNet](https://github.com/ZHKKKe/MODNet) portrait-matting model on the CPU with ONNX Runtime.

Built for a Raspberry Pi 5 (4GB) on Raspberry Pi OS Trixie with a Camera Module 3 capturing at full resolution
(4608×2592), but it works with any camera pibooth supports.

## How it works

1. **Each capture is processed as soon as it is taken**, in a background thread. The work overlaps the countdown
   for the next shot, so by the time the last photo is taken most of the sequence is already done.
2. **MODNet runs at a small size** (short side 512 px by default, aspect ratio kept, no square stretching).
3. **The matte is upsampled with a fast guided filter**, which snaps its edges to the real edges in the
   full-resolution photo instead of a blurry upscale.
4. **Full-resolution compositing** uses only C-level PIL/OpenCV operations. Backgrounds and overlays are resized once
   per theme and cached.
5. The processed captures replace the originals in pibooth's final picture (and in the wait-screen animation). The
   untouched originals are still saved in pibooth's `raw` folder.

Themes with only an overlay skip the AI step entirely. If the model cannot load, the booth keeps working and simply
leaves backgrounds unchanged.

## Install

On Raspberry Pi OS Trixie, pip refuses to install into the system Python (PEP 668), so choose one of the options below.
In both, OpenCV comes from apt so it matches the system numpy; it is optional, and without it edge refinement is
skipped.

### Option 1: virtual environment (recommended)

```bash
sudo apt install python3-opencv python3-picamera2 python3-venv

# --system-site-packages lets the venv see the apt-installed picamera2 and OpenCV
python3 -m venv --system-site-packages ~/pibooth-venv
source ~/pibooth-venv/bin/activate

pip install pibooth pibooth-modnet     # skip "pibooth" if it is already in the venv
```

Everything that runs pibooth must use this venv: start it with `~/pibooth-venv/bin/pibooth`, and in an autostart
entry or systemd service use that full path instead of `pibooth`. The model download command below needs the venv
too (`~/pibooth-venv/bin/pibooth-modnet --download`, or activate it first).

If pibooth is already installed in an existing venv, activate that one and run only `pip install pibooth-modnet`.

### Option 2: system Python, no venv

```bash
sudo apt install python3-opencv
pip install --break-system-packages pibooth-modnet
```

This is simpler but can clash with apt-managed packages. Use it only if your pibooth is already installed this way.

### Without the apt OpenCV

`pip install "pibooth-modnet[opencv]"` installs `opencv-python-headless` from PyPI instead. Do not install both.

pibooth discovers the plugin automatically; no `plugins =` line in `pibooth.cfg` is needed. Check the start-up log
for `Installed plugins: ... modnet-1.0.0`.

Download the model (about 26 MB) before your first event, so the booth never needs the internet:

```bash
pibooth-modnet --download
```

It is saved to `~/.config/pibooth/models/modnet.onnx` and checked against a SHA-256 hash. If the download fails,
copy `modnet.onnx` from the photobooth-app repository (`src/photobooth/utils/rembg/models/`) to that path.

### Upgrading from the single-file plugin

If you used the old `pibooth-modnet.py` file, remove its path from `[GENERAL] plugins` in `pibooth.cfg` (otherwise
the plugin loads twice), then install from PyPI as above. Your `[MODNET]` settings are kept.

### Known issue: onnxruntime 1.21 on Raspberry Pi

onnxruntime 1.21 prints many `Schema error: Trying to register schema ... already registered` lines when the model
loads. They are harmless, but 1.20.1 is clean; the package therefore excludes 1.21.x. If pip left you on 1.21, run
`pip install onnxruntime==1.20.1`.

## Themes

Each sub-folder of `themes_path` is one theme. Underscores in folder names become spaces on screen.

```
~/.config/pibooth/themes/
├── Beach/
│   ├── background.jpg      # replaces the background (needs AI)
│   └── overlay.png         # drawn on top of each photo (transparent PNG)
├── Neon_Party/
│   └── background.jpg      # background only
└── Wedding_Frame/
    └── overlay.png         # overlay only: no AI, very fast
```

* **Backgrounds** can be any size or aspect ratio; they are centre-cropped to fill the capture.
* **Overlays** are stretched to the capture size, so design them at the capture aspect ratio: **4608×2592 (16:9)**
  for a Camera Module 3 at full resolution. Use a PNG with transparency.
* If `[PICTURE] captures_cropping = True`, pibooth crops captures to fit its layout slots, which can cut off the edges
  of an overlay. Either keep important overlay content near the centre or turn cropping off.
* pibooth's own `[PICTURE] overlays` option still works too; it goes over the *whole* final picture, whereas theme
  overlays go over *each capture*.

### Choosing a theme

| `theme_selection` | Behaviour |
|---|---|
| `manual` | Guests press **← / →** on the wait screen; the theme name shows in a bar at the top. |
| `cycle` | Each new guest gets the next theme. |
| `random` | A random theme per guest. |

On the wait screen, touch and the hardware buttons already mean capture/print in pibooth, so manual selection uses
the keyboard arrow keys (a small USB or Bluetooth keypad works well).

## Configuration

All options live in the `[MODNET]` section and are added to `pibooth.cfg` automatically on first start.

| Option | Default | Notes |
|---|---|---|
| `enabled` | `True` | Also in pibooth's settings menu. |
| `themes_path` | `~/.config/pibooth/themes` | Folder of theme folders. |
| `theme_selection` | `manual` | `manual`, `cycle` or `random`. |
| `include_no_theme` | `False` | Adds a "No theme" choice that keeps the original photo. |
| `show_theme_banner` | `True` | Theme name bar on the wait screen (manual only). |
| `model_path` | `~/.config/pibooth/models/modnet.onnx` | Downloaded automatically if missing. |
| `inference_size` | `512` | Model short side. 512 is what MODNet was trained for; 384 is faster; 640 can help wide group shots. |
| `threads` | `3` | Model CPU threads. 3 leaves one core for a smooth live preview; try 4 if the preview stays smooth. |
| `refine_edges` | `True` | Guided-filter edge refinement (needs OpenCV). |

## Test and benchmark on the Pi

Before an event, run a real capture through a theme:

```bash
pibooth-modnet --test ~/Pictures/pibooth/raw/<date>/pibooth000.jpg ~/.config/pibooth/themes/Beach out.jpg
```

It prints the matte and composite times for three runs and saves the result. On a 2-core x86 test machine a
4608×2592 capture took about 0.9 s in total (0.5 s model, 0.08 s edge refinement, 0.07 s composite). Expect the Pi 5 to
be slower per capture; because each capture is processed during the next countdown, what matters is that one capture
finishes within your `[WINDOW] preview_delay`.

## Tips for clean cut-outs

* Light guests evenly from the front; avoid strong backlight.
* A plain backdrop a metre or more behind guests helps more than any setting. Busy patterns and colours close to skin
  or hair are the hardest cases.
* MODNet is trained for people. Large props held away from the body may be partly removed.

## Credits and licences

* MODNet by Ke et al., Apache-2.0 — https://github.com/ZHKKKe/MODNet
* ONNX model file as packaged by photobooth-app, MIT — https://github.com/photobooth-app/photobooth-app
* Plugin: MIT

## Development and releasing

```bash
git clone https://github.com/peteoheat/pibooth-modnet && cd pibooth-modnet
pip install -e .
```

To publish a release to PyPI:

1. Bump `__version__` in `pibooth_modnet/__init__.py` and add an entry to `CHANGELOG.md`.
2. One-off setup: on pypi.org add a *trusted publisher* for the project (owner `peteoheat`, repository
   `pibooth-modnet`, workflow `publish.yml`, environment `pypi`), and create a `pypi` environment in the GitHub
   repository settings.
3. Create a GitHub release for the new tag. The `Publish to PyPI` workflow builds and uploads it.

To publish by hand instead: `pip install build twine && python -m build && twine upload dist/*`.
