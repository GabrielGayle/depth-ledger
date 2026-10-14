"""Model registry and loading."""
from depth_core.pipeline import (  # noqa: F401
    MODEL_CHOICES,
    MODEL_KEYS,
    PROMPTDA_IDS,
    DEFAULT_MODEL,
    MAX_SIZE,
    LoadedBackend,
    get_backend,
    normalize_model_key,
    is_promptda,
    pick_device,
    CHECKPOINT_PATH,
)
