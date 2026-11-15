"""
Step 1a: Crop the large single-scene GeoTIFFs into small, GPU-friendly patches.

The 4 optical bands (B2, B3, B4, B5) and the thermal band (B10) all come from the
same Landsat scene, so they share one pixel grid and are perfectly co-registered.
We read them with aligned sliding windows, convert raw DN to physical units using
the official Landsat Collection 2 Level-2 scaling factors, drop patches dominated
by no-data/fill pixels, and save each surviving patch as a single .npz file:

    x : float32 [4, H, W]  surface reflectance in [0, 1]  (R, G, B, NIR)
    y : float32 [H, W]     land surface temperature in degrees Celsius

It also records the global min/max temperature in dataset/stats.json so the target
can be normalised (and later de-normalised back to Celsius) during training.

Run:  python -m src.preprocess
"""
from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from . import config as C


def _find_band(directory, suffix: str):
    """Return the single .TIF file in *directory* whose name contains *suffix*."""
    matches = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() == ".tif" and suffix in p.name
    )
    if not matches:
        raise FileNotFoundError(f"No GeoTIFF containing '{suffix}' found in {directory}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one '{suffix}' file, found {len(matches)}: {matches}")
    return matches[0]


def _scale_reflectance(dn: np.ndarray) -> np.ndarray:
    """Raw DN -> surface reflectance, clipped to the physical [0, 1] range."""
    refl = dn.astype(np.float32) * C.SR_SCALE + C.SR_OFFSET
    return np.clip(refl, 0.0, 1.0)


def _scale_temperature(dn: np.ndarray) -> np.ndarray:
    """Raw DN -> land surface temperature in degrees Celsius."""
    kelvin = dn.astype(np.float32) * C.ST_SCALE + C.ST_OFFSET
    return kelvin - C.KELVIN_TO_CELSIUS


def generate_patches() -> None:
    C.PATCH_DIR.mkdir(parents=True, exist_ok=True)

    optical_paths = [_find_band(C.INPUT_DIR, s) for s in C.OPTICAL_BAND_SUFFIXES]
    thermal_path = _find_band(C.TARGET_DIR, C.THERMAL_BAND_SUFFIX)

    print("Optical bands (R, G, B, NIR):")
    for s, p in zip(C.OPTICAL_BAND_SUFFIXES, optical_paths):
        print(f"  {s}: {p.name}")
    print(f"Thermal band: {thermal_path.name}\n")

    # Open every band once; rasterio reads lazily per window.
    opt_ds = [rasterio.open(p) for p in optical_paths]
    thm_ds = rasterio.open(thermal_path)
    datasets = opt_ds + [thm_ds]

    # Use the common overlapping extent in case bands differ by a pixel.
    width = min(ds.width for ds in datasets)
    height = min(ds.height for ds in datasets)
    print(f"Scene grid: {width} x {height} px, patch={C.PATCH_SIZE}, stride={C.STRIDE}\n")

    ps, stride = C.PATCH_SIZE, C.STRIDE
    saved, skipped = 0, 0
    temp_min, temp_max = np.inf, -np.inf

    rows = range(0, height - ps + 1, stride)
    cols = range(0, width - ps + 1, stride)
    total = len(rows) * len(cols)

    for row in tqdm(rows, desc="Cropping patches"):
        for col in cols:
            win = Window(col, row, ps, ps)

            optical_dn = np.stack([ds.read(1, window=win) for ds in opt_ds], axis=0)
            thermal_dn = thm_ds.read(1, window=win)

            # A pixel is valid only if every band (optical + thermal) has data.
            valid = np.all(optical_dn != C.FILL_VALUE, axis=0) & (thermal_dn != C.FILL_VALUE)
            if valid.mean() < C.MIN_VALID_FRACTION:
                skipped += 1
                continue

            x = _scale_reflectance(optical_dn)          # [4, ps, ps] in [0, 1]
            y = _scale_temperature(thermal_dn)          # [ps, ps] in Celsius

            # Fill the few remaining no-data pixels with the patch mean so they
            # don't inject extreme values into the model.
            if not valid.all():
                y[~valid] = y[valid].mean()
                for ch in range(x.shape[0]):
                    ch_arr = x[ch]
                    ch_arr[~valid] = ch_arr[valid].mean()

            temp_min = min(temp_min, float(y[valid].min()))
            temp_max = max(temp_max, float(y[valid].max()))

            out_path = C.PATCH_DIR / f"patch_{row:05d}_{col:05d}.npz"
            np.savez_compressed(out_path, x=x.astype(np.float32), y=y.astype(np.float32))
            saved += 1

    for ds in datasets:
        ds.close()

    stats = {
        "temp_min": temp_min,
        "temp_max": temp_max,
        "num_patches": saved,
        "patch_size": ps,
    }
    C.STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nScanned {total} windows -> saved {saved}, skipped {skipped} (no-data).")
    print(f"Temperature range: {temp_min:.2f} C .. {temp_max:.2f} C")
    print(f"Patches : {C.PATCH_DIR}")
    print(f"Stats   : {C.STATS_PATH}")


if __name__ == "__main__":
    generate_patches()
