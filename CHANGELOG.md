# Changelog

## 1.0.1

* Fix startup crash on pibooth 3.0 (`get_captures` and `_post_process_capture` were renamed `grab_captures` and `_process_capture`); pibooth 2.x still works.

## 1.0.0

* Initial release: MODNet background replacement and per-theme overlays for pibooth 2.1.x.
* Packaged for PyPI; pibooth discovers the plugin automatically after `pip install pibooth-modnet`.
