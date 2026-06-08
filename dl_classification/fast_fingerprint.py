"""
FFT-based drop-in replacement for src/features_generation.generate_rf_dna_fingerprint().

The original DGT is a triple Python loop: O(M × KG × MN) = 150³ ≈ 3.4M ops/fingerprint
(~5 s on M1, ~2 h for the full augmented dataset). For N=1, each time frame is just an
FFT of the windowed signal slice, reducing it to 150 FFT calls (~0.1 s, ~50× faster).

Numerically identical to the original for N=1. Only the DGT step is replaced;
extract_features_from_patches() from src/ is unchanged.
"""

import numpy as np
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features_generation import extract_features_from_patches


def _fast_dgt(signal, M=150, KG=150, N=1):
    """Numerically identical to src/features_generation.dgt() for N=1."""
    if N != 1:
        raise NotImplementedError(
            "fast_dgt only supports N=1 (the value used throughout this project). "
            "For N>1 use src/features_generation.dgt() directly."
        )

    MN = M * N          # = M for N=1
    length = len(signal)

    n_idx  = np.arange(MN)
    std    = MN // 8
    window = np.exp(-0.5 * ((n_idx - MN // 2) / std) ** 2)   # shape (MN,)

    Gmk = np.zeros((M, KG), dtype=np.complex128)

    for m in range(M):
        end = m + MN
        if end > length:
            seg = np.zeros(MN, dtype=np.complex128)
            available = length - m
            if available > 0:
                seg[:available] = signal[m:length]
        else:
            seg = signal[m : m + MN].astype(np.complex128)

        windowed = seg * window
        Gmk[m, :] = np.fft.fft(windowed, n=KG)

    return Gmk


def _normalize_gabor(Gmk):
    """Identical to src/features_generation.normalize_gabor_coefficients()."""
    mn, mx = Gmk.min(), Gmk.max()
    if mx == mn:
        return np.zeros_like(Gmk)
    return (Gmk - mn) / (mx - mn)


def fast_rf_dna_fingerprint(signal, fs=20e6, M=150, KG=150, N=1,
                             NP=100, NT=15, NF=15, mode="diagonal"):
    """
    Drop-in replacement for src/features_generation.generate_rf_dna_fingerprint().

    Uses a numpy-FFT-based DGT (~50× faster) but produces numerically
    identical output for N=1.  All downstream steps (coefficient normalisation,
    patch extraction, statistical feature computation) are unchanged.

    Parameters
    ----------
    signal : np.ndarray  — filtered IQ transient segment
    fs     : float       — sampling frequency (not used in feature computation,
                           kept for API compatibility)
    M, KG  : int         — DGT time frames and frequency bins
    N      : int         — oversampling factor (must be 1)
    NP     : int         — number of patches
    NT, NF : int         — patch height (time) and width (frequency)
    mode   : str         — patch placement mode ('diagonal', 'horizontal', 'vertical')

    Returns
    -------
    fingerprint : np.ndarray, shape (NP*5 + 5,) = (505,) for NP=100
    """
    Gmk = _fast_dgt(signal, M=M, KG=KG, N=N)
    Zxx = _normalize_gabor(np.abs(Gmk) ** 2)
    fingerprint = extract_features_from_patches(Zxx, NP=NP, NT=NT, NF=NF, mode=mode)

    return fingerprint
