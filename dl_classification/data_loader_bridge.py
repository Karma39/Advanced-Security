"""
Bridge between src/ (original RF-DNA pipeline) and the DL classification pipeline.

Raw data: complex64 IQ samples from a BladeRF AX4 at 20 MHz. Each device folder
contains one .bin file (~160 MB for device1-9, ~80 MB for device10-12). The
recordings capture the turn-on transient — a brief burst (a few ms) during which
hardware impairments unique to the device (CFO, DCO, phase offset, amplitude
envelope) are imprinted on the signal.

Preprocessing (src/): load IQ → normalize magnitude → detect transients →
    Chebyshev low-pass filter (5 MHz) → DGT (150×150) → 505-dim feature vector.

Sliding-window augmentation: the original code reads only the first 299 IQ
samples per transient to compute the DGT, leaving most of a 100 000-sample burst
unused. Since CFO and DCO are constant throughout the transient, any 300-sample
window yields a valid fingerprint. This gives up to 10× more training samples
(134 → 1 340) at the cost of intra-transient correlation between windows.
Windows from val/test transients are never mixed into training.
"""

import os
import re
import sys
import time
import numpy as np
import h5py

# Allow imports from the project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dataloader import load_data_and_unique_labels, load_data_from_hdf5
from src.preprocessing import (
    load_iq_data, normalize_magnitude, detect_transients,
    filter_transients, apply_lowpass_filter,
)
from src.features_generation import generate_rf_dna_fingerprint
from dl_classification.fast_fingerprint import fast_rf_dna_fingerprint

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(_ROOT, "original_dataset")
CACHE_PATH   = os.path.join(_ROOT, "processed_data", "fv_original.h5")

# The DGT (M=150, KG=150, N=1) accesses index = n + m*N, max = (MN-1)+(M-1)*N
# = 149 + 149 = 298.  Any input shorter than 299 samples produces incomplete
# DGT coefficients.  We use 300 as the minimum safe window size.
_DGT_MIN_SAMPLES = 300   # = ceil(M*N + (M-1)*N) + 1 with M=150, N=1

# Preprocessing parameters — identical to RF_Fingerprint.py
_LOAD_PARAMS = dict(
    sample_rate              = 20e6,
    cutoff_freq              = 5e6,
    M                        = 150,
    KG                       = 150,
    N                        = 1,
    NP                       = 100,
    NT                       = 15,
    NF                       = 15,
    transient_threshold      = 0.38,
    specific_duration_threshold  = 0.005,
    specific_magnitude_threshold = 0.3,
    min_transient_duration   = 0.005,
    filter_type              = "chebyshev",
    filter_order             = 4,
    filter_ripple            = 0.5,
    mode                     = "diagonal",
)


def load_raw_data():
    """Original src.dataloader pipeline. Returns X (N, 505) and y (N,)."""
    if os.path.exists(CACHE_PATH):
        print(f"[bridge] Loading cached data from {CACHE_PATH}")
        X, y, device_id_mapping, total_devices = load_data_from_hdf5(CACHE_PATH)
    else:
        print(f"[bridge] Cache not found. Running preprocessing on {RAW_DATA_DIR} ...")
        X, y, device_id_mapping, total_devices = load_data_and_unique_labels(
            RAW_DATA_DIR, save_path=CACHE_PATH, **_LOAD_PARAMS,
        )
        print(f"[bridge] Preprocessing complete. Cached to {CACHE_PATH}")

    X = X.astype(np.float32)
    y = y.astype(np.int64)
    print(f"[bridge] Dataset: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
    return X, y, device_id_mapping, total_devices


def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]



def _save_hdf5(path, X, y, device_id_mapping, total_devices):
    """Save dataset in the same HDF5 format used by src.dataloader."""
    with h5py.File(path, "w") as f:
        f.create_dataset("data",   data=X)
        f.create_dataset("labels", data=y)
        f.create_dataset("device_id_mapping",
                         data=np.string_(str(device_id_mapping)))
        f.create_dataset("total_devices",
                         data=np.string_(str(total_devices)))
    print(f"  Cache saved → {path}", flush=True)


class RFPreprocessor:
    """
    Preprocessing pipeline with correct IQ normalization and optional
    sliding-window augmentation.

    windowed=False: one fingerprint per transient (baseline).
    windowed=True:  sliding windows across each transient — see module docstring.
    window_size must be ≥ _DGT_MIN_SAMPLES (300). window_step=300 gives
    non-overlapping windows; smaller values increase sample count and correlation.
    """

    def __init__(
        self,
        params: dict = None,
        cache_path: str = CACHE_PATH,
        data_dir: str = RAW_DATA_DIR,
        # --- sliding-window augmentation parameters ---
        windowed: bool = False,
        window_size: int = _DGT_MIN_SAMPLES,
        window_step: int = _DGT_MIN_SAMPLES,
        max_windows_per_transient: int = 50,
    ):
        if window_size < _DGT_MIN_SAMPLES:
            raise ValueError(
                f"window_size={window_size} is smaller than _DGT_MIN_SAMPLES="
                f"{_DGT_MIN_SAMPLES}.  The DGT with M=150,N=1 accesses indices"
                f" up to {_DGT_MIN_SAMPLES - 2}; shorter windows produce "
                f"incomplete time-frequency coefficients."
            )
        self.p   = {**_LOAD_PARAMS, **(params or {})}
        self.data_dir             = data_dir
        self.windowed             = windowed
        self.window_size          = window_size
        self.window_step          = window_step
        self.max_windows          = max_windows_per_transient
        # Use a separate cache file so baseline and augmented runs coexist
        _cache_dir = os.path.dirname(cache_path)
        self.cache_path = (
            cache_path if not windowed
            else os.path.join(_cache_dir, "fv_windowed.h5")
        )

    # ------------------------------------------------------------------
    def load(self, force_reprocess: bool = False):
        """
        Load or compute the dataset.

        Returns
        -------
        X : np.ndarray, shape (N, 505), float32
        y : np.ndarray, shape (N,),    int64
        device_id_mapping : dict  {device_name: class_index}
        total_devices     : list  [device_name, ...]
        """
        if not force_reprocess and os.path.exists(self.cache_path):
            print(f"[RFPreprocessor] Loading cache: {self.cache_path}")
            X, y, mapping, devices = load_data_from_hdf5(self.cache_path)
            X = X.astype(np.float32)
            y = y.astype(np.int64)
            print(f"[RFPreprocessor] {len(X)} samples | "
                  f"{len(np.unique(y))} classes | "
                  f"windowed={self.windowed}")
            return X, y, mapping, devices

        if self.windowed:
            return self._run_windowed()
        return self._run_baseline()

    # ------------------------------------------------------------------
    def _discover_devices(self):
        """Return sorted list of (device_name, device_id) and total_devices."""
        folders = sorted(
            [f for f in os.listdir(self.data_dir)
             if os.path.isdir(os.path.join(self.data_dir, f))],
            key=_natural_sort_key,
        )
        mapping = {name: idx for idx, name in enumerate(folders)}
        return mapping, list(mapping.keys())

    # ------------------------------------------------------------------
    def _fingerprint_segment(self, iq_segment):
        """
        Apply low-pass filter and compute one 505-dim RF-DNA fingerprint
        from a raw complex IQ segment.  Shared by both baseline and windowed
        paths so the feature computation is identical in both modes.

        Uses fast_rf_dna_fingerprint() (numpy FFT-based DGT, ~50× faster than
        the original pure-Python triple loop) for N=1.  The result is
        numerically identical to generate_rf_dna_fingerprint().
        """
        p = self.p
        filtered = apply_lowpass_filter(
            iq_segment, p["cutoff_freq"], p["sample_rate"],
            filter_type=p["filter_type"],
            order=p["filter_order"],
            ripple=p["filter_ripple"],
        )
        return fast_rf_dna_fingerprint(
            filtered, fs=p["sample_rate"],
            M=p["M"], KG=p["KG"], N=p["N"],
            NP=p["NP"], NT=p["NT"], NF=p["NF"],
            mode=p["mode"],
        )

    # ------------------------------------------------------------------
    def _run_baseline(self):
        """
        Baseline mode: one fingerprint per detected transient.
        Uses normalize_magnitude() before detect_transients() — fixing the
        bug in the original src/dataloader.py where normalize_magnitude() was
        commented out, causing detect_transients() to receive raw complex64
        data and silently compare only real parts via numpy's ComplexWarning.
        """
        p = self.p
        device_id_mapping, total_devices = self._discover_devices()
        all_data, all_labels = [], []
        t0 = time.time()

        print(f"\n[RFPreprocessor | baseline] {len(device_id_mapping)} devices")
        print(f"  data_dir   : {self.data_dir}")
        print(f"  cache      : {self.cache_path}")
        print(f"  threshold  : {p['transient_threshold']}  "
              f"min_dur={p['min_transient_duration']}s  "
              f"mag>{p['specific_magnitude_threshold']}")
        print("-" * 70)

        for device_name, device_id in device_id_mapping.items():
            folder_path = os.path.join(self.data_dir, device_name)
            bin_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".bin")],
                key=_natural_sort_key,
            )
            dev_count = 0
            t_dev = time.time()
            print(f"\n  {device_name} (class {device_id}) — {len(bin_files)} file(s)")

            for fi, fname in enumerate(bin_files):
                filepath = os.path.join(folder_path, fname)
                size_mb  = os.path.getsize(filepath) / 1e6
                t_file   = time.time()
                print(f"    [{fi+1}/{len(bin_files)}] {fname} ({size_mb:.1f} MB)...",
                      flush=True)
                try:
                    iq_data  = load_iq_data(filepath, 0, -1)
                    norm_mag = normalize_magnitude(iq_data)
                    starts, ends = detect_transients(
                        norm_mag, p["sample_rate"],
                        p["transient_threshold"], p["min_transient_duration"],
                    )
                    selected = filter_transients(
                        starts, ends, norm_mag, p["sample_rate"],
                        p["specific_duration_threshold"],
                        p["specific_magnitude_threshold"],
                    )
                    fp_count = 0
                    for start, end in selected:
                        if end - start < 1:
                            continue
                        fp = self._fingerprint_segment(iq_data[start:end])
                        all_data.append(fp)
                        all_labels.append(device_id)
                        fp_count += 1
                    dev_count += fp_count
                    print(f"         transients={len(selected)} | "
                          f"fingerprints={fp_count} | "
                          f"{time.time()-t_file:.1f}s", flush=True)
                except Exception as e:
                    print(f"         ERROR: {e}", flush=True)

            print(f"  → {device_name}: {dev_count} fingerprints  "
                  f"({time.time()-t_dev:.1f}s)", flush=True)

        X = np.array(all_data,   dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        elapsed = time.time() - t0

        print(f"\n{'='*70}")
        print(f"[RFPreprocessor | baseline] {len(X)} fingerprints in "
              f"{elapsed/60:.1f} min  |  X={X.shape}")
        print(f"{'='*70}", flush=True)

        _save_hdf5(self.cache_path, X, y, device_id_mapping, total_devices)
        return X, y, device_id_mapping, total_devices

    # ------------------------------------------------------------------
    def _run_windowed(self):
        p = self.p
        device_id_mapping, total_devices = self._discover_devices()
        all_data, all_labels = [], []
        t0 = time.time()

        ws  = self.window_size
        st  = self.window_step
        cap = self.max_windows

        print(f"\n[RFPreprocessor | windowed] {len(device_id_mapping)} devices")
        print(f"  data_dir        : {self.data_dir}")
        print(f"  cache           : {self.cache_path}")
        print(f"  window_size     : {ws} samples  "
              f"({ws/p['sample_rate']*1e6:.1f} µs at {p['sample_rate']/1e6:.0f} MHz)")
        print(f"  window_step     : {st} samples  "
              f"({st/p['sample_rate']*1e6:.1f} µs)")
        print(f"  max_windows/tr  : {cap if cap else 'unlimited'}")
        print(f"  DGT min samples : {_DGT_MIN_SAMPLES}")
        print("-" * 70)

        for device_name, device_id in device_id_mapping.items():
            folder_path = os.path.join(self.data_dir, device_name)
            bin_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(".bin")],
                key=_natural_sort_key,
            )
            dev_total = 0
            t_dev = time.time()
            print(f"\n  {device_name} (class {device_id}) — {len(bin_files)} file(s)")

            for fi, fname in enumerate(bin_files):
                filepath = os.path.join(folder_path, fname)
                size_mb  = os.path.getsize(filepath) / 1e6
                t_file   = time.time()
                print(f"    [{fi+1}/{len(bin_files)}] {fname} ({size_mb:.1f} MB)...",
                      flush=True)
                try:
                    # -- load and detect transients (same as baseline) --------
                    iq_data  = load_iq_data(filepath, 0, -1)
                    norm_mag = normalize_magnitude(iq_data)
                    starts, ends = detect_transients(
                        norm_mag, p["sample_rate"],
                        p["transient_threshold"], p["min_transient_duration"],
                    )
                    selected = filter_transients(
                        starts, ends, norm_mag, p["sample_rate"],
                        p["specific_duration_threshold"],
                        p["specific_magnitude_threshold"],
                    )

                    file_wins = 0
                    for tr_idx, (start, end) in enumerate(selected):
                        L = end - start
                        if L < ws:
                            # Transient shorter than one window: skip.
                            # This is rare — min_transient_duration=5 ms gives
                            # L≥100 000 at 20 MHz, far above ws=300.
                            print(f"         transient {tr_idx}: too short "
                                  f"({L} samples < ws={ws}), skipped",
                                  flush=True)
                            continue

                        # Maximum possible windows from this transient
                        max_possible = (L - ws) // st + 1
                        n_windows = (
                            min(max_possible, cap) if cap else max_possible
                        )

                        # When capping: distribute windows evenly across the
                        # transient rather than bunching at the start, to
                        # sample the full duration and maximise diversity.
                        if cap and max_possible > cap:
                            # Evenly-spaced start offsets within [0, L-ws]
                            offsets = np.linspace(0, L - ws, n_windows,
                                                  dtype=int)
                        else:
                            offsets = np.arange(0, n_windows * st, st)

                        for offset in offsets:
                            win_iq = iq_data[start + offset :
                                             start + offset + ws]
                            if len(win_iq) < ws:
                                continue   # safety guard for boundary
                            fp = self._fingerprint_segment(win_iq)
                            all_data.append(fp)
                            all_labels.append(device_id)
                            file_wins += 1

                        print(f"         transient {tr_idx}: L={L} samples | "
                              f"max_windows={max_possible} | "
                              f"extracted={n_windows}", flush=True)

                    dev_total += file_wins
                    print(f"         file total: {file_wins} fingerprints | "
                          f"{time.time()-t_file:.1f}s", flush=True)

                except Exception as e:
                    print(f"         ERROR: {e}", flush=True)

            print(f"  → {device_name}: {dev_total} fingerprints  "
                  f"({time.time()-t_dev:.1f}s)", flush=True)

        X = np.array(all_data,   dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        elapsed = time.time() - t0

        unique, counts = np.unique(y, return_counts=True)
        print(f"\n{'='*70}")
        print(f"[RFPreprocessor | windowed] {len(X)} fingerprints in "
              f"{elapsed/60:.1f} min")
        print(f"  X shape : {X.shape}")
        print(f"  Per-class counts: "
              + "  ".join(f"cls{u}={c}" for u, c in zip(unique, counts)))
        print(f"{'='*70}", flush=True)

        _save_hdf5(self.cache_path, X, y, device_id_mapping, total_devices)
        return X, y, device_id_mapping, total_devices
