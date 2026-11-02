from isaac_compat import close_application
"""Probe USD inside the Isaac application, where extension imports are valid."""
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
from isaaclab.app import AppLauncher
from runtime_platform import platform_kit_args
app = AppLauncher(headless=True, enable_cameras=args.render,
                  kit_args=platform_kit_args('--/app/vulkan=false')).app
try:
    from pxr import Usd
    print('ISAAC_RUNTIME_PROBE=' + json.dumps({'pxr': True, 'rendering_enabled': args.render, 'usd_version': Usd.GetVersion()}), flush=True)
finally:
    import os, threading
    # Kit shutdown can hang after the completed check, as in the training driver.
    timer = threading.Timer(15, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    close_application(app)
    timer.cancel()
