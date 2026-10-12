"""Repository paths shared by the Gradio app and CLI."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "depth_pro.pt"
OUTPUT_DIR = ROOT / "outputs"
EXAMPLE_RGB = ROOT / "PromptDA" / "assets" / "example_images" / "image.jpg"
EXAMPLE_DEPTH = ROOT / "PromptDA" / "assets" / "example_images" / "arkit_depth.png"

BUNDLE_FORMAT_VERSION = "1.0"
BUNDLE_RGB_NAME = "rgb.jpg"
BUNDLE_DEPTH_NAME = "depth_mm.png"
BUNDLE_CAMERA_NAME = "camera.json"
