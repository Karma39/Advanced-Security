"""
Data loader for raw DGT matrices.

Instead of extracting RF-DNA patch statistics (505-dim vectors), this module
stops after the DGT step and saves the full normalized 150×150 power matrix.
The preprocessing chain is otherwise identical to dl_classification/:

    IQ → normalize_magnitude → detect_transients → filter_transients
       → apply_lowpass_filter → _fast_dgt → |·|² → _normalize_gabor
       → float32 (150, 150)

Sliding-window augmentation (10 windows/transient) is applied to the
training split only; val/test use one DGT per transient (baseline).
Caches: raw_dgt/processed_dgt.h5  and  raw_dgt/processed_dgt_windowed.h5
"""

import os
import re
import sys
import time
import numpy as np
import h5py

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_HERE = os.path.dirname(os.path.abspath(__file__))

from src.preprocessing import (
    load_iq_data, normalize_magnitude, detect_transients,
    filter_transients, apply_lowpass_filter,
)
# Reuse the fast DGT implementation from dl_classification (not modified)
from dl_classification.fast_fingerprint import _fast_dgt, _normalize_gabor
from dl_classification.data_loader_bridge import _natural_sort_key

RAW_DATA_DIR = os.path.join(_ROOT, "original_dataset")
CACHE_BASE   = os.path.join(_ROOT, "processed_data", "dgt_original.h5")
CACHE_WIN    = os.path.join(_ROOT, "processed_data", "dgt_windowed.h5")

_DGT_WIN = 300   # minimum IQ samples for a complete 150×150 DGT
_M = _KG = 150

_PREP = dict(
    sample_rate=20e6, cutoff_freq=5e6,
    transient_threshold=0.38, min_transient_duration=0.005,
    specific_duration_threshold=0.005, specific_magnitude_threshold=0.3,
    filter_type="chebyshev", filter_order=4, filter_ripple=0.5,
)



class DGTPreprocessor:
    """
    Builds an (N, 150, 150) float32 dataset from raw IQ .bin files.

    windowed=False : one DGT per transient (baseline, 134 samples)
    windowed=True  : sliding-window augmentation, 10 DGTs per transient
    """

    def __init__(self, windowed=False, window_size=_DGT_WIN, window_step=_DGT_WIN,
                 max_windows=10):
        self.windowed = windowed
        self.ws  = window_size
        self.st  = window_step
        self.cap = max_windows
        self.cache = CACHE_WIN if windowed else CACHE_BASE

    def load(self, force=False):
        if not force and os.path.exists(self.cache):
            print(f"[DGTPreprocessor] Loading cache: {self.cache}")
            with h5py.File(self.cache, "r") as f:
                X = f["data"][:].astype(np.float32)
                y = f["labels"][:].astype(np.int64)
            print(f"  {len(X)} samples, shape {X.shape}")
            return X, y
        return self._run()

    def _dgt_segment(self, iq_window):
        """Lowpass filter → DGT → |·|² → normalize → (150, 150) float32."""
        p = _PREP
        filtered = apply_lowpass_filter(
            iq_window, p["cutoff_freq"], p["sample_rate"],
            filter_type=p["filter_type"], order=p["filter_order"],
            ripple=p["filter_ripple"],
        )
        Gmk  = _fast_dgt(filtered, M=_M, KG=_KG, N=1)
        return _normalize_gabor(np.abs(Gmk) ** 2).astype(np.float32)

    def _run(self):
        p = _PREP
        folders = sorted(
            [f for f in os.listdir(RAW_DATA_DIR)
             if os.path.isdir(os.path.join(RAW_DATA_DIR, f))],
            key=_natural_sort_key,
        )
        device_map = {name: idx for idx, name in enumerate(folders)}

        all_X, all_y = [], []
        mode = "windowed" if self.windowed else "baseline"
        t0 = time.time()

        print(f"\n[DGTPreprocessor | {mode}] {len(folders)} devices")
        if self.windowed:
            print(f"  window={self.ws}  step={self.st}  cap={self.cap}")
        print("-" * 60)

        for device_name, device_id in device_map.items():
            folder = os.path.join(RAW_DATA_DIR, device_name)
            bins = sorted(
                [f for f in os.listdir(folder) if f.endswith(".bin")],
                key=_natural_sort_key,
            )
            dev_count = 0

            for fname in bins:
                fpath = os.path.join(folder, fname)
                try:
                    iq = load_iq_data(fpath, 0, -1)
                    norm_mag = normalize_magnitude(iq)
                    starts, ends = detect_transients(
                        norm_mag, p["sample_rate"],
                        p["transient_threshold"], p["min_transient_duration"],
                    )
                    selected = filter_transients(
                        starts, ends, norm_mag, p["sample_rate"],
                        p["specific_duration_threshold"],
                        p["specific_magnitude_threshold"],
                    )

                    for start, end in selected:
                        L = end - start
                        if L < self.ws:
                            continue

                        if self.windowed:
                            max_p = (L - self.ws) // self.st + 1
                            n_win = min(max_p, self.cap) if self.cap else max_p
                            offsets = (
                                np.linspace(0, L - self.ws, n_win, dtype=int)
                                if (self.cap and max_p > self.cap)
                                else np.arange(0, n_win * self.st, self.st)
                            )
                        else:
                            offsets = [0]   # baseline: first window only

                        for offset in offsets:
                            win = iq[start + offset : start + offset + self.ws]
                            if len(win) < self.ws:
                                continue
                            all_X.append(self._dgt_segment(win))
                            all_y.append(device_id)
                            dev_count += 1

                except Exception as e:
                    print(f"  ERROR {fname}: {e}", flush=True)

            print(f"  {device_name}: {dev_count} matrices", flush=True)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        print(f"\n[DGTPreprocessor | {mode}] {len(X)} total  ({time.time()-t0:.0f}s)")

        with h5py.File(self.cache, "w") as f:
            ds = f.create_dataset("data",   data=X)
            ds.attrs["windows_per_transient"] = self.cap if self.windowed else 1
            f.create_dataset("labels", data=y)
        print(f"  Saved → {self.cache}", flush=True)
        return X, y


