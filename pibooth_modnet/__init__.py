# -*- coding: utf-8 -*-
"""pibooth-modnet: AI background replacement and overlays for pibooth.

Removes the background behind guests with the MODNet portrait-matting model
(ONNX Runtime, CPU), replaces it with a background image and adds an overlay
on top of each capture. Built for pibooth 2.1.x on a Raspberry Pi 5 with
Raspberry Pi OS Trixie and a Camera Module 3 capturing at full resolution.

Speed strategy
--------------
* Each capture is processed in a worker thread as soon as it is taken, so the
  work overlaps the countdown for the next shot instead of piling up in the
  processing screen.
* MODNet runs at a small fixed size (short side 512 by default); the matte is
  then upsampled to full resolution with a fast guided filter that snaps it to
  the real edges in the full-resolution photo.
* Full-resolution compositing uses C-level PIL / OpenCV operations only, and
  resized backgrounds and overlays are cached per theme and capture size.
* Themes with only an overlay skip the AI step entirely.

Install with ``pip install pibooth-modnet``; pibooth discovers it automatically.
See README.md for the full guide.
"""

import os
import os.path as osp
import sys
import time
import random
import hashlib
import threading
import urllib.request
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    cv2.setNumThreads(2)  # Keep OpenCV from fighting ONNX Runtime and the preview for cores
except ImportError:  # Fallback to PIL only (no edge refinement)
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pygame
    import pibooth
    from pibooth.utils import LOGGER
except ImportError:  # Allow the command line tools to run without pibooth
    pibooth = None
    pygame = None
    import logging
    LOGGER = logging.getLogger("pibooth-modnet")


__version__ = "1.0.0"

SECTION = 'MODNET'

# MODNet photographic portrait matting (Apache-2.0), ONNX export with dynamic
# input size, as packaged by photobooth-app. Pinned to a commit and checked by
# SHA-256 so the download can never silently change.
MODEL_URL = ("https://github.com/photobooth-app/photobooth-app/raw/"
             "e2db0d33752efe2204af08bcfd0d20c2c4d985ff/"
             "src/photobooth/utils/rembg/models/modnet.onnx")
MODEL_SHA256 = "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
NO_THEME_NAME = "No theme"


def _plugin_hookimpl(*args, **kwargs):
    """Return pibooth.hookimpl, or a no-op decorator outside pibooth."""
    if pibooth is not None:
        return pibooth.hookimpl(*args, **kwargs)
    if args and callable(args[0]):
        return args[0]
    return lambda func: func


# --------------------------------------------------------------------------
# Model handling
# --------------------------------------------------------------------------

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path):
    """Download the MODNet model if it is missing or corrupted.

    :return: path to a valid model file
    """
    path = osp.abspath(osp.expanduser(path))
    if osp.isfile(path):
        return path  # A user supplied model is trusted as is

    os.makedirs(osp.dirname(path), exist_ok=True)
    LOGGER.info("MODNet: downloading model to %s", path)
    tmp = path + '.part'
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if _sha256(tmp) != MODEL_SHA256:
        os.remove(tmp)
        raise IOError("MODNet: downloaded model failed its checksum, please retry")
    os.replace(tmp, path)
    LOGGER.info("MODNet: model downloaded")
    return path


class MattingEngine(object):

    """Compute a full resolution alpha matte with MODNet."""

    def __init__(self, model_path, ref_size=512, threads=3, refine=True):
        if ort is None:
            raise ImportError("onnxruntime is not installed (pip install onnxruntime)")
        self.ref_size = max(64, int(ref_size) // 32 * 32)
        self.refine = refine and cv2 is not None

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def inference_size(self, width, height):
        """Scale so the short side is ``ref_size``, keep the aspect ratio (no
        square stretching) and round both sides to a multiple of 32."""
        scale = self.ref_size / float(min(width, height))
        return (max(32, int(round(width * scale / 32.0)) * 32),
                max(32, int(round(height * scale / 32.0)) * 32))

    def warmup(self):
        """First ONNX Runtime run allocates buffers, do it before any guest."""
        self.matte(np.zeros((2592, 4608, 3), dtype=np.uint8))

    def _run(self, small_rgb):
        tensor = small_rgb.astype(np.float32)
        tensor *= 1.0 / 127.5
        tensor -= 1.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[np.newaxis])
        return self.session.run(None, {self.input_name: tensor})[0][0, 0]

    def matte(self, rgb):
        """Return a uint8 alpha matte (H, W) for the uint8 RGB array ``rgb``."""
        height, width = rgb.shape[:2]
        size = self.inference_size(width, height)

        if cv2 is not None:
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        else:
            small = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR,
                                                           reducing_gap=2.0))
        low = self._run(small)

        if self.refine:
            return self._guided_upsample(low, small, rgb)

        low8 = (np.clip(low, 0.0, 1.0) * 255.0).astype(np.uint8)
        if cv2 is not None:
            return cv2.resize(low8, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(Image.fromarray(low8).resize((width, height), Image.Resampling.BILINEAR))

    @staticmethod
    def _guided_upsample(low, small_rgb, full_rgb, radius=2, eps=1e-3):
        """Fast guided filter (He & Sun, 2015).

        The linear coefficients are solved at inference resolution, upsampled,
        then applied to the full resolution grey image: the edges of the matte
        follow the real full resolution edges instead of a blurry upscale.
        """
        box = (2 * radius + 1, 2 * radius + 1)
        guide = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) * (1.0 / 255.0)
        mean_i = cv2.blur(guide, box)
        mean_p = cv2.blur(low, box)
        cov_ip = cv2.blur(guide * low, box) - mean_i * mean_p
        var_i = cv2.blur(guide * guide, box) - mean_i * mean_i
        coef_a = cov_ip / (var_i + eps)
        coef_b = mean_p - coef_a * mean_i
        # Work in 0-255 directly on the big image: q = a * grey_u8 + b * 255
        coef_a = cv2.blur(coef_a, box)
        coef_b = cv2.blur(coef_b, box) * 255.0

        height, width = full_rgb.shape[:2]
        full_a = cv2.resize(coef_a, (width, height), interpolation=cv2.INTER_LINEAR)
        full_b = cv2.resize(coef_b, (width, height), interpolation=cv2.INTER_LINEAR)
        grey = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2GRAY)
        result = cv2.multiply(full_a, grey, dtype=cv2.CV_32F)
        del full_a
        cv2.add(result, full_b, dst=result)
        del full_b
        cv2.max(result, 0.0, dst=result)
        return cv2.convertScaleAbs(result)  # Saturates to [0, 255] as uint8


# --------------------------------------------------------------------------
# Themes (background and / or overlay)
# --------------------------------------------------------------------------

class Theme(object):

    """A background image and/or an overlay image applied to each capture."""

    def __init__(self, name, background=None, overlay=None):
        self.name = name
        self.background = background
        self.overlay = overlay
        self._cache = {}
        self._lock = threading.Lock()

    def __repr__(self):
        return "Theme({!r})".format(self.name)

    @property
    def needs_matting(self):
        return self.background is not None

    @property
    def is_empty(self):
        return self.background is None and self.overlay is None

    def get_background(self, size):
        """Background resized to cover ``size`` (centre crop), cached."""
        key = ('bg', size)
        with self._lock:
            if key not in self._cache:
                image = ImageOps.exif_transpose(Image.open(self.background)).convert('RGB')
                self._cache[key] = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            return self._cache[key]

    def get_overlay(self, size):
        """(RGB, alpha) overlay stretched to ``size``, cached.

        Overlays should be designed at the capture aspect ratio (16:9 for a
        Camera Module 3 at full resolution), they are not cropped.
        """
        key = ('ov', size)
        with self._lock:
            if key not in self._cache:
                image = Image.open(self.overlay).convert('RGBA')
                if image.size != size:
                    image = image.resize(size, Image.Resampling.LANCZOS)
                self._cache[key] = (image.convert('RGB'), image.getchannel('A'))
            return self._cache[key]

    def preload(self, size):
        if self.background:
            self.get_background(size)
        if self.overlay:
            self.get_overlay(size)


def _find_image(folder, stem):
    for ext in IMAGE_EXTENSIONS:
        path = osp.join(folder, stem + ext)
        if osp.isfile(path):
            return path
    return None


def load_themes(themes_path, include_no_theme=False):
    """Each sub-folder of ``themes_path`` is a theme containing a
    ``background.*`` image, an ``overlay.png`` image, or both."""
    themes = []
    if osp.isdir(themes_path):
        for name in sorted(os.listdir(themes_path)):
            folder = osp.join(themes_path, name)
            if not osp.isdir(folder):
                continue
            theme = Theme(name.replace('_', ' '), _find_image(folder, 'background'),
                          _find_image(folder, 'overlay'))
            if theme.is_empty:
                LOGGER.warning("MODNet: theme folder '%s' has no background or overlay image", folder)
            else:
                themes.append(theme)
    else:
        LOGGER.warning("MODNet: themes folder '%s' does not exist", themes_path)

    if include_no_theme or not themes:
        themes.insert(0, Theme(NO_THEME_NAME))
    return themes


# --------------------------------------------------------------------------
# Capture processing
# --------------------------------------------------------------------------

class CaptureProcessor(object):

    def __init__(self, engine):
        self.engine = engine

    def process(self, image, theme):
        """Return a new image with ``theme`` applied (``image`` is untouched)."""
        if theme is None or theme.is_empty:
            return image

        start = time.time()
        image = image.convert('RGB')
        size = image.size

        if theme.needs_matting and self.engine is not None:
            rgb = np.asarray(image)
            matte = self.engine.matte(rgb)
            inferred = time.time()
            # composite(fg, bg, mask): PIL C loop, uint8 only, no float copies
            result = Image.composite(image, theme.get_background(size), Image.fromarray(matte))
            LOGGER.info("MODNet: matte %.2fs, composite %.2fs (%dx%d)",
                        inferred - start, time.time() - inferred, size[0], size[1])
        else:
            if theme.needs_matting:
                LOGGER.warning("MODNet: model not available, background left unchanged")
            result = image.copy()

        if theme.overlay:
            overlay_rgb, overlay_alpha = theme.get_overlay(size)
            result.paste(overlay_rgb, (0, 0), overlay_alpha)

        LOGGER.info("MODNet: theme '%s' applied in %.2fs", theme.name, time.time() - start)
        return result


class ModnetController(object):

    """State shared by the pibooth hooks."""

    def __init__(self):
        self.enabled = False
        self.themes = []
        self.index = 0
        self.selection = 'manual'
        self.session_theme = None
        self.show_banner = True
        self.timeout = 60.0
        self.capture_size = None

        self.engine = None
        self.processor = CaptureProcessor(None)
        self.engine_ready = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='modnet')
        self.pending = []   # futures for the captures of the current sequence
        self.results = {}   # id(raw capture image) -> future of processed image
        self._banner = None

    # -- setup ---------------------------------------------------------------

    def load_engine(self, model_path, ref_size, threads, refine):
        try:
            path = ensure_model(model_path)
            started = time.time()
            engine = MattingEngine(path, ref_size, threads, refine)
            engine.warmup()
            self.engine = engine
            self.processor = CaptureProcessor(engine)
            LOGGER.info("MODNet: model ready in %.1fs (inference short side %d, %d threads, "
                        "edge refinement %s)", time.time() - started, engine.ref_size, threads,
                        'on' if engine.refine else 'off')
        except Exception as ex:
            LOGGER.error("MODNet: model could not be loaded, backgrounds will not be replaced: %s", ex)
        finally:
            self.engine_ready.set()

    @property
    def current_theme(self):
        return self.themes[self.index] if self.themes else None

    def preload_current(self):
        theme, size = self.current_theme, self.capture_size
        if theme and size:
            self.executor.submit(self._safe_preload, theme, size)

    @staticmethod
    def _safe_preload(theme, size):
        try:
            theme.preload(size)
        except Exception as ex:
            LOGGER.error("MODNet: theme '%s' images could not be loaded: %s", theme.name, ex)

    # -- capture sequence ----------------------------------------------------

    def start_sequence(self):
        self.reset()
        if not self.enabled or not self.themes:
            self.session_theme = None
            return
        if self.selection == 'random':
            self.index = random.randrange(len(self.themes))
        self.session_theme = self.current_theme
        if self.selection == 'cycle':  # Next guest gets the next theme
            self.index = (self.index + 1) % len(self.themes)
        LOGGER.info("MODNet: theme for this sequence is '%s'", self.session_theme.name)

    def submit_capture(self, camera, capture_data):
        """Start processing a capture still held by the camera, while the
        booth carries on with the next countdown."""
        theme = self.session_theme
        if theme is None or theme.is_empty:
            return
        try:
            stream, effect = capture_data
            # Private copy of the JPEG so the camera can still decode its own one
            copy = (BytesIO(stream.getvalue()), effect)
        except (TypeError, ValueError, AttributeError):
            return  # Not a Picamera2 JPEG stream: processed later in bind_captures()
        self.pending.append(self.executor.submit(self._process_raw, camera, copy, theme))

    def _process_raw(self, camera, capture_data, theme):
        image = camera._post_process_capture(capture_data)  # Same rotation, crop, flip and effect
        return self._process_image(image, theme)

    def _process_image(self, image, theme):
        if theme.needs_matting:
            self.engine_ready.wait(self.timeout)
        return self.processor.process(image, theme)

    def bind_captures(self, captures):
        """Link the raw captures returned by the camera to the processed ones."""
        if self.pending and len(self.pending) == len(captures):
            for capture, future in zip(captures, self.pending):
                self.results[id(capture)] = future
        else:
            if self.pending:
                LOGGER.warning("MODNet: %d early captures for %d raw ones, processing again",
                               len(self.pending), len(captures))
                self.reset()
            theme = self.session_theme
            if theme is not None and not theme.is_empty:
                # Other camera types: process now from the decoded captures
                for capture in captures:
                    self.results[id(capture)] = self.executor.submit(self._process_image,
                                                                     capture, theme)
        self.pending = []

    def replace_images(self, factory):
        """Swap raw captures in a picture factory for their processed version."""
        images = getattr(factory, '_images', None)
        if not images or not self.results:
            return
        replaced = []
        for image in images:
            future = self.results.get(id(image))
            if future is None:
                replaced.append(image)
                continue
            try:
                replaced.append(future.result(timeout=self.timeout))
            except Exception as ex:
                LOGGER.error("MODNet: capture processing failed, using the original: %s", ex)
                replaced.append(image)
        factory._images = tuple(replaced)

    def reset(self):
        for future in self.pending:
            future.cancel()
        self.pending = []
        self.results = {}

    # -- wait screen ---------------------------------------------------------

    def select(self, step):
        if len(self.themes) > 1:
            self.index = (self.index + step) % len(self.themes)
            self._banner = None
            LOGGER.info("MODNet: theme '%s' selected", self.current_theme.name)
            self.preload_current()

    def draw_banner(self, win):
        if not (self.enabled and self.show_banner and self.selection == 'manual'
                and len(self.themes) > 1 and pygame):
            return
        width = win.surface.get_width()
        if self._banner is None or self._banner.get_width() != width:
            height = max(40, win.surface.get_height() // 14)
            font = pygame.font.Font(None, int(height * 0.8))
            label = font.render("<   {}   >".format(self.current_theme.name), True, (255, 255, 255))
            # Opaque: the wait screen is not repainted every frame, so a
            # translucent banner would darken as it is blitted again and again
            banner = pygame.Surface((width, height))
            banner.fill((25, 25, 25))
            banner.blit(label, label.get_rect(center=(width // 2, height // 2)))
            self._banner = banner
        win.surface.blit(self._banner, (0, 0))

    def shutdown(self):
        self.reset()
        self.executor.shutdown(wait=False)


CTRL = ModnetController()


# --------------------------------------------------------------------------
# pibooth hooks
# --------------------------------------------------------------------------

@_plugin_hookimpl
def pibooth_configure(cfg):
    cfg.add_option(SECTION, 'enabled', True,
                   "Replace backgrounds and add overlays on captures",
                   "Enable AI backgrounds", ['True', 'False'])
    cfg.add_option(SECTION, 'themes_path', '~/.config/pibooth/themes',
                   "Folder of themes, one sub-folder per theme holding background.jpg/png and/or overlay.png")
    cfg.add_option(SECTION, 'theme_selection', 'manual',
                   "How the theme is chosen: manual (left/right arrows on the wait screen), "
                   "cycle (next theme for each guest) or random",
                   "Theme selection", ['manual', 'cycle', 'random'])
    cfg.add_option(SECTION, 'include_no_theme', False,
                   "Add a 'No theme' choice that keeps the original photo",
                   "Offer no theme", ['True', 'False'])
    cfg.add_option(SECTION, 'show_theme_banner', True,
                   "Show the selected theme name on the wait screen (manual selection only)")
    cfg.add_option(SECTION, 'model_path', '~/.config/pibooth/models/modnet.onnx',
                   "MODNet ONNX model, downloaded automatically on first start if missing")
    cfg.add_option(SECTION, 'inference_size', 512,
                   "Short side in pixels the model runs at (multiple of 32). 512 is what MODNet "
                   "was trained for; 384 is faster, 640 can help large groups")
    cfg.add_option(SECTION, 'threads', 3,
                   "CPU threads for the model (3 leaves one core free for a smooth preview)")
    cfg.add_option(SECTION, 'refine_edges', True,
                   "Snap the matte to full resolution edges with a guided filter (needs OpenCV)")


@_plugin_hookimpl
def pibooth_startup(cfg, app):
    CTRL.enabled = cfg.getboolean(SECTION, 'enabled')
    CTRL.selection = cfg.get(SECTION, 'theme_selection').strip().lower()
    if CTRL.selection not in ('manual', 'cycle', 'random'):
        LOGGER.warning("MODNet: unknown theme_selection '%s', using manual", CTRL.selection)
        CTRL.selection = 'manual'
    CTRL.show_banner = cfg.getboolean(SECTION, 'show_theme_banner')
    CTRL.themes = load_themes(cfg.getpath(SECTION, 'themes_path'),
                              cfg.getboolean(SECTION, 'include_no_theme'))
    CTRL.index = 0
    LOGGER.info("MODNet: %d theme(s): %s", len(CTRL.themes), ", ".join(t.name for t in CTRL.themes))

    resolution = cfg.gettyped('CAMERA', 'resolution')
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        CTRL.capture_size = (int(resolution[0]), int(resolution[1]))
        for theme in CTRL.themes:  # Resize every theme once, in the background
            CTRL.executor.submit(CTRL._safe_preload, theme, CTRL.capture_size)

    if CTRL.enabled and any(theme.needs_matting for theme in CTRL.themes):
        # Load in the background so pibooth starts without waiting
        threading.Thread(target=CTRL.load_engine, daemon=True, name='modnet-load',
                         args=(cfg.getpath(SECTION, 'model_path'),
                               cfg.getint(SECTION, 'inference_size'),
                               cfg.getint(SECTION, 'threads'),
                               cfg.getboolean(SECTION, 'refine_edges'))).start()
    else:
        CTRL.engine_ready.set()

    # Wrap the camera so raw captures can be matched to processed ones
    camera = app.camera
    original_get_captures = camera.get_captures

    def get_captures():
        captures = original_get_captures()
        CTRL.bind_captures(captures)
        return captures

    camera.get_captures = get_captures


@_plugin_hookimpl(trylast=True)
def state_wait_enter(win):
    CTRL.reset()
    CTRL._banner = None
    CTRL.draw_banner(win)


@_plugin_hookimpl(trylast=True)
def state_wait_do(win, events):
    if CTRL.enabled and CTRL.selection == 'manual':
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                CTRL.select(-1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                CTRL.select(1)
    CTRL.draw_banner(win)


@_plugin_hookimpl
def state_wait_exit(win):
    CTRL.start_sequence()
    if CTRL._banner is not None:
        win.update()  # Repaint the background to remove the theme banner


@_plugin_hookimpl
def state_capture_exit(app):
    captures = getattr(app.camera, '_captures', None)
    if captures:
        CTRL.submit_capture(app.camera, captures[-1])


@_plugin_hookimpl(tryfirst=True)
def pibooth_setup_picture_factory(factory):
    # This is a firstresult hook: plugins such as pibooth-picture-template return
    # a new factory built from factory._images, which stops later plugins from
    # running. Run first so the images are already replaced when they do.
    CTRL.replace_images(factory)
    # Return nothing: the factory is updated in place


@_plugin_hookimpl
def state_failsafe_enter():
    CTRL.reset()


@_plugin_hookimpl
def pibooth_cleanup():
    CTRL.shutdown()


# --------------------------------------------------------------------------
# Command line: download the model and benchmark before an event
#   pibooth-modnet --download
#   pibooth-modnet --test photo.jpg path/to/theme_folder [out.jpg]
# --------------------------------------------------------------------------

def _main(argv):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    model = osp.expanduser('~/.config/pibooth/models/modnet.onnx')

    if len(argv) >= 1 and argv[0] == '--download':
        print(ensure_model(model))
        return 0

    if len(argv) >= 3 and argv[0] == '--test':
        photo, folder = argv[1], argv[2]
        output = argv[3] if len(argv) > 3 else 'modnet-test.jpg'
        theme = Theme(osp.basename(folder.rstrip('/')), _find_image(folder, 'background'),
                      _find_image(folder, 'overlay'))
        image = ImageOps.exif_transpose(Image.open(photo)).convert('RGB')
        engine = MattingEngine(ensure_model(model))
        engine.warmup()
        processor = CaptureProcessor(engine)
        theme.preload(image.size)
        for _ in range(3):
            start = time.time()
            result = processor.process(image, theme)
            print("Total {:.2f}s".format(time.time() - start))
        result.save(output, quality=92)
        print("Saved", output)
        return 0

    print(__doc__.split('Install with')[0])
    print("Usage:\n  pibooth-modnet --download\n"
          "  pibooth-modnet --test photo.jpg themes/My_Theme [out.jpg]")
    return 1


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
