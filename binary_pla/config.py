"""Central configuration for all experiment parameters."""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HDF5 caches
FV_ORIGINAL_PATH  = os.path.join(_ROOT, "processed_data", "fv_original.h5")
FV_WINDOWED_PATH  = os.path.join(_ROOT, "processed_data", "fv_windowed.h5")
DGT_ORIGINAL_PATH = os.path.join(_ROOT, "processed_data", "dgt_original.h5")
DGT_WINDOWED_PATH = os.path.join(_ROOT, "processed_data", "dgt_windowed.h5")

N_DEVICES    = 12
N_WIN        = 10   # overlapping windows per transient
DEVICE_NAMES = [f'device{i}' for i in range(1, N_DEVICES + 1)]

TRIALS = {
    "trial_1": {
        "authorized": ["device3", "device2", "device12", "device9"],
        "rogue":      ["device8", "device11", "device5", "device1",
                       "device6", "device10", "device7", "device4"],
    },
    "trial_2": {
        "authorized": ["device10", "device6", "device12", "device3", "device11", "device1"],
        "rogue":      ["device2", "device5", "device8", "device7", "device4", "device9"],
    },
    "trial_3": {
        "authorized": ["device1", "device10", "device9", "device8",
                       "device12", "device11", "device6", "device7"],
        "rogue":      ["device4", "device2", "device5", "device3"],
    },
}

TRIAL        = "trial_1"
TRIALS_TO_RUN = ["trial_1", "trial_2", "trial_3"]

NB01_ID = "nb01_ml_baseline"
NB02_ID = "nb02_dl_honest"
NB04_ID = "nb04_multiclass"

SEEDS        = [42, 7, 13, 99, 2024]
BATCH_SIZE   = 64
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
N_JOBS       = -1   # joblib; set to 1 if Jupyter hangs

SPEC_AUGMENT = {"T_max": 30, "F_max": 30, "noise_std": 0.02, "p": 0.5}

NB01_SVM = {
    "n_features_candidates": [50, 100, 150, 200, 250, 300],
    "k_fixed":     50,
    "n_splits":    5,
    "k_sweep_max": 100,
}

NB02_DL = {
    "mlp_fv":    {"epochs": 50, "patience": 10},
    "gru_dgt":   {"epochs": 50, "patience": 10},
    "bigru_dgt": {"epochs": 50, "patience": 10},
}

NB04_DL = {
    "bigru_dgt": {"epochs": 50, "patience": 10},
    "mlp_fv":    {"epochs": 50, "patience": 10},
}
