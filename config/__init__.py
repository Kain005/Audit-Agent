"""Configuration package."""

from .loader import PolicyConfig, get_policy, load_policy_config
from .runtime_paths import configure_pytesseract, get_poppler_path, get_tesseract_path

__all__ = [
	"PolicyConfig",
	"load_policy_config",
	"get_policy",
	"configure_pytesseract",
	"get_poppler_path",
	"get_tesseract_path",
]
