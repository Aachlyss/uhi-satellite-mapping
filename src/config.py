"""Central configuration for the UHI project (Step 1: Data Preparation)."""
from pathlib import Path

# --- Directory layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input_features"    # optical bands B2, B3, B4, B5
TARGET_DIR = PROJECT_ROOT / "target_labels"    # thermal band B10
PATCH_DIR = PROJECT_ROOT / "dataset" / "patches"  # generated .npz patches
STATS_PATH = PROJECT_ROOT / "dataset" / "stats.json"

# --- Band ordering ----------------------------------------------------------
# Optical input channels, in the order they are stacked for the model.
# B4 = Red, B3 = Green, B2 = Blue, B5 = Near-Infrared  (R, G, B, NIR)
OPTICAL_BAND_SUFFIXES = ["SR_B4", "SR_B3", "SR_B2", "SR_B5"]
THERMAL_BAND_SUFFIX = "ST_B10"

# --- Landsat 8/9 Collection 2 Level-2 scaling factors -----------------------
# Surface Reflectance:  reflectance = DN * 0.0000275 - 0.2   (unitless, ~[0, 1])
SR_SCALE = 0.0000275
SR_OFFSET = -0.2
# Surface Temperature:  kelvin = DN * 0.00341802 + 149.0
ST_SCALE = 0.00341802
ST_OFFSET = 149.0
KELVIN_TO_CELSIUS = 273.15

# --- Patch / sampling settings ----------------------------------------------
PATCH_SIZE = 256
STRIDE = 128               # 256 = non-overlapping tiles; lower for overlap
MIN_VALID_FRACTION = 0.95  # skip patches with too many fill/no-data pixels
FILL_VALUE = 0             # Landsat Collection 2 fill value (raw DN)

# --- Dataloader settings ----------------------------------------------------
VAL_SPLIT = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 4
RANDOM_SEED = 42


def get_device():
    """Pick the best available backend: Apple MPS > CUDA > CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
