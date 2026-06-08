import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from src.dataloader import load_data_and_unique_labels, load_data_from_hdf5
from src.evaluation import evaluate_model_criteria, calculate_closeness_score
from src.features_selections import select_features
import joblib

# Define data paths and processed data file
DATA_DIRECTORY = r'dataset'  # Directory containing the raw data files
PROCESSED_DATA_PATH = r'dataset/processed_fingerprints_data.h5'  # Path to save the processed fingerprints data

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Run RF fingerprinting with trials, models, and feature selection options.')
    parser.add_argument('--trial', type=str, nargs='+', default=None, help='Specify one or more trials (e.g., trial_1 trial_2). Leave empty to run all trials.')
    parser.add_argument('--model', type=str, nargs='+', default=None, help='Specify one or more models (e.g., random_forest svc). Leave empty to run all models.')
    parser.add_argument('--feature', type=str, nargs='+', default=None, help='Specify one or more feature selection methods (e.g., pca anova). Leave empty to run all methods.')
    parser.add_argument('--run_all', action='store_true', help='Run all combinations of trials, models, and feature selection methods.')

    return parser.parse_args()

# Check if processed data exists
if os.path.exists(PROCESSED_DATA_PATH):
    # Load processed data from HDF5 file
    X, y, device_id_mapping, total_devices = load_data_from_hdf5(PROCESSED_DATA_PATH)
else:
    # Load the data and unique labels
    X, y, device_id_mapping, total_devices = load_data_and_unique_labels(
        DATA_DIRECTORY,
        sample_rate=20e6,
        cutoff_freq=5e6,
        M=150,
        KG=150,
        N=1,
        NP=100,
        NT=15,
        NF=15,
        transient_threshold=0.38,
        specific_duration_threshold=0.005,
        specific_magnitude_threshold=0.3,
        min_transient_duration=0.005,
        filter_type='chebyshev',
        filter_order=4,
        filter_ripple=0.5,
        mode='diagonal',
        save_path=PROCESSED_DATA_PATH
    )

# Example trial configuration (authorized and rogue devices)
trials_info = {
    'trial_1': {
        'authorized': ['device3', 'device2', 'device12', 'device9'],
        'rogue': ['device8', 'device11', 'device5', 'device1', 'device6', 'device10', 'device7', 'device4']
    },
    'trial_2': {
        'authorized': ['device10', 'device6', 'device12', 'device3', 'device11', 'device1'],
        'rogue': ['device2', 'device5', 'device8', 'device7', 'device4', 'device9']
    },
    'trial_3': {
        'authorized': ['device1', 'device10', 'device9', 'device8', 'device12', 'device11', 'device6', 'device7'],
        'rogue': ['device4', 'device2', 'device5', 'device3']
    }
}

# Get model based on user input or run all
def get_selected_models(args):
    return args.model if args.model else model_configs.keys()

# Get feature selection methods based on user input or run all
def get_selected_features(args):
    return args.feature if args.feature else feature_selection_methods

# Get trials based on user input or run all
def get_selected_trials(args):
    return args.trial if args.trial else trials_info.keys()

# Function to return a machine learning model based on the specified model type.
# The function uses **kwargs to pass additional hyperparameters to the model.
def get_model(model_type, **kwargs):
    if model_type == 'svc':
        return SVC(**kwargs)
    elif model_type == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_type == 'knn':
        return KNeighborsClassifier(**kwargs)
    elif model_type == 'xgb':
        return XGBClassifier(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
# Function to calculate the Average Detection Rate (ADR) for a given trial.
# ADR is calculated as the average of the True Detection Rates (TDR) for authorized and rogue (malicious) devices.
def calculate_adr_for_trial(results):
    trial_adr = {}
    for device_id, device_results in results.items():
        auth_tvr = device_results['auth_tvr']
        rogue_tvr = device_results['rogue_tvr']
        adr = (auth_tvr + rogue_tvr) / 2
        trial_adr[device_id] = adr
    overall_adr = np.mean(list(trial_adr.values()))
    return trial_adr, overall_adr

def save_results_to_json(trial_name, model_type, method, results, overall_adr, output_dir='results'):
    # Create a folder for each trial and each model type within the main output directory
    trial_dir = os.path.join(output_dir, trial_name, model_type) 
    os.makedirs(trial_dir, exist_ok=True)  # Create trial- and model-specific directory if it doesn't exist
    
    # Define the filename including method
    filename = os.path.join(trial_dir, f'{method}_results.json')  # File name based on the method

    # Prepare the data in the structured format for JSON saving
    results_data = {
        'trial_name': trial_name,
        'model_type': model_type,
        'method': method,
        'overall_adr': overall_adr,
        'devices': {}
    }

    for device_id, result in results.items():
        results_data['devices'][device_id] = {
            'auth_tvr': result['auth_tvr'],
            'auth_fvr': result['auth_fvr'],
            'rogue_tvr': result['rogue_tvr']
        }

    # Save the results in a JSON file
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {filename}")

# Function to train models for each trial, model, and feature selection method.
def train_model_per_trial(device_rf_data, trials_info, model_configs, feature_selection_methods):
    all_results = {}
    all_adr_results = {}

    for trial_name, devices in trials_info.items():
        authorized_devices = devices['authorized']
        rogue_devices = devices['rogue']

        for model_type, model_params in model_configs.items():
            for method in feature_selection_methods:
                results = {}
                trial_weighted_accuracy = []

                for device_id in tqdm(authorized_devices, desc=f"Training {model_type} with {method} for {trial_name}"):
                    X_auth, y_auth, X_rogue = [], [], []
                    sample_device_mapping_auth, sample_device_mapping_rogue = [], []

                    for did, rf_data in device_rf_data.items():
                        for feature_vector in rf_data:
                            if did == device_id:
                                X_auth.append(feature_vector)
                                y_auth.append(1)
                                sample_device_mapping_auth.append(did)
                            elif did in authorized_devices:
                                X_auth.append(feature_vector)
                                y_auth.append(0)
                                sample_device_mapping_auth.append(did)
                            if did in rogue_devices:
                                X_rogue.append(feature_vector)
                                sample_device_mapping_rogue.append(did)

                    X_auth = np.array(X_auth)
                    y_auth = np.array(y_auth)
                    X_rogue = np.array(X_rogue)
                    y_rogue = np.zeros(len(X_rogue), dtype=int)

                    best_score = float('inf')
                    best_model_details = None
                    best_weighted_accuracy = None

                    for n_features_to_select in range(1, X_auth.shape[1] + 1):
                        X_auth_selected, selected_features_indices = select_features(X_auth, y_auth, n_features_to_select, method=method)
                        X_rogue_selected = X_rogue[:, selected_features_indices]

                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        for train_index, test_index in skf.split(X_auth_selected, y_auth):
                            X_train, X_test = X_auth_selected[train_index], X_auth_selected[test_index]
                            y_train, y_test = y_auth[train_index], y_auth[test_index]

                            model = get_model(model_type, **model_params)
                            model.fit(X_train, y_train)

                            y_pred_auth = model.predict(X_test)
                            auth_tvr, auth_fvr, misclassified_auth = evaluate_model_criteria(
                                y_test, y_pred_auth, positive_label=1,
                                sample_device_mapping=[sample_device_mapping_auth[i] for i in test_index]
                            )
                            y_pred_rogue = model.predict(X_rogue_selected)
                            rogue_tvr, rogue_fvr, misclassified_rogue = evaluate_model_criteria(
                                y_rogue, y_pred_rogue, positive_label=0, sample_device_mapping=sample_device_mapping_rogue
                            )

                            score = calculate_closeness_score(auth_tvr, auth_fvr, rogue_tvr, rogue_fvr)
                            weighted_accuracy = (auth_tvr * 0.5 + rogue_tvr * 0.5)

                            if auth_tvr >= 0.95 and auth_fvr <= 0.05 and rogue_tvr >= 0.95 and rogue_fvr <= 0.05:
                                results[device_id] = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_tvr': auth_tvr,
                                    'auth_fvr': auth_fvr,
                                    'rogue_tvr': rogue_tvr,
                                    'rogue_fvr': rogue_fvr,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                trial_weighted_accuracy.append(weighted_accuracy)
                                print(f"Device {device_id} in Trial {trial_name}: Model meets criteria with TDR: {auth_tvr}, FDR: {auth_fvr} for Authorized, and TDR: {rogue_tvr} for Malicious.")
                                save_model(device_id, trial_name, model, selected_features_indices, model_type, method)
                                break
                            elif score < best_score:
                                best_score = score
                                best_model_details = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_tvr': auth_tvr,
                                    'auth_fvr': auth_fvr,
                                    'rogue_tvr': rogue_tvr,
                                    'rogue_fvr': rogue_fvr,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                best_weighted_accuracy = weighted_accuracy

                        if device_id in results:
                            break

                    if device_id not in results and best_model_details is not None:
                        results[device_id] = best_model_details
                        trial_weighted_accuracy.append(best_weighted_accuracy)
                        print(f"Device {device_id} in Trial {trial_name}: No model met full criteria. Using closest model with TDR: {best_model_details['auth_tvr']}, FDR: {best_model_details['auth_fvr']} for Authorized, and TDR : {best_model_details['rogue_tvr']} for Malicious.")
                        # Save the closest model as well
                        save_model(device_id, trial_name, best_model_details['model'], best_model_details['features_indices'], model_type, method)

                # Calculate ADR for the trial and combination
                trial_adr, overall_adr = calculate_adr_for_trial(results)
                all_adr_results[(trial_name, model_type, method)] = {
                    'adr_per_device': trial_adr,
                    'overall_adr': overall_adr
                }

                # Output the overall ADR for the trial and combination
                print(f"Average Detection Rate for Trial {trial_name} with {model_type} and {method}: {overall_adr}")

                # Save the results to JSON
                save_results_to_json(trial_name, model_type, method, results, overall_adr)

                # Store the results for this combination
                all_results[(trial_name, model_type, method)] = {
                    'results': results,
                    'overall_adr': overall_adr
                }

    return all_results, all_adr_results

def save_model(device_id, trial_name, model, selected_features_indices, model_type, method):
    """
    Saves the trained model and selected feature indices to a file using joblib.

    Args:
    - device_id (str): Device ID.
    - trial_name (str): Trial name.
    - model (sklearn.base.BaseEstimator): Trained model.
    - selected_features_indices (np.ndarray): Indices of selected features.
    - model_type (str): Type of model used.
    - method (str): Feature selection method used.
    """
    # Create the directory for the trial and combination if it doesn't exist
    trial_dir = os.path.join('model_saved', trial_name, model_type, method)
    os.makedirs(trial_dir, exist_ok=True)

    model_details = {
        'model': model,
        'features_indices': selected_features_indices
    }
    filename = os.path.join(trial_dir, f'model_{device_id}.joblib')
    joblib.dump(model_details, filename)
    print(f"Model and features indices saved: {filename}")


# Define the configurations for all models that will be used for training and evaluation.
# Each model type has its own set of hyperparameters specific to its algorithm.
model_configs = {
    # Configuration for the Random Forest model
    'random_forest': {
        'n_estimators': 10,         # Number of trees in the forest
        'max_depth': 8,             # Maximum depth of each tree (limits the tree growth)
        'random_state': 42,         # Random seed for reproducibility
        'n_jobs': 1                 # Number of CPU cores to use (1 means use a single core)
    },
    # Configuration for the Support Vector Classifier (SVC)
    'svc': {
        'kernel': 'poly',           # Kernel type to be used in the algorithm (polynomial kernel)
        'C': 1.0,                   # Regularization parameter (controls trade-off between maximizing the margin and minimizing classification errors)
        'probability': True          # Whether to enable probability estimates (useful for cross-validation)
    },
    # Configuration for the K-Nearest Neighbors (KNN) model
    'knn': {
        'n_neighbors': 5            # Number of neighbors to use for classification (the default value is 5)
    },
    # Configuration for the XGBoost model (Extreme Gradient Boosting)
    'xgb': {
        'objective': 'binary:logistic',  # Objective function for binary classification
        'max_depth': 8,                  # Maximum depth of a tree (similar to Random Forest)
        'n_estimators': 10,              # Number of boosting rounds (trees)
        'learning_rate': 0.1,            # Step size shrinkage (learning rate) to prevent overfitting
        'subsample': 0.7,                # Fraction of samples to be randomly chosen for each tree
        'colsample_bytree': 0.7,         # Fraction of features to be used in each boosting round (reduces overfitting)
        'random_state': 42               # Random seed for reproducibility
    },
    # Configuration for the Logistic Regression model
    'logistic_regression': {
        'C': 1.0,                        # Inverse of regularization strength (smaller values specify stronger regularization)
        'random_state': 42,              # Random seed for reproducibility
        'solver': 'liblinear',           # Algorithm to use in the optimization problem
        'max_iter': 500                  # Maximum number of iterations for the solver to converge
    }
}

# Define the list of feature selection methods to be used.
# These methods will be applied to select the most relevant features before training the models.
feature_selection_methods = [
    'pca',           # Principal Component Analysis: reduces dimensionality by selecting key components
    'mutual_info',   # Mutual Information: selects features based on the amount of shared information with the target variable
    'anova',         # ANOVA F-test: selects features based on statistical significance in classification tasks
    'rfe'            # Recursive Feature Elimination: recursively removes features to improve model performance
]

# Main function that orchestrates the workflow
def main():
    args = parse_args()

    # Get the list of selected trials, models, and feature selection methods based on the user's input (command-line arguments)
    selected_trials = get_selected_trials(args)
    selected_models = get_selected_models(args)
    selected_features = get_selected_features(args)

    # If the user specified the --run_all flag, run all combinations of
    # trials, models, and feature selection methods.
    if args.run_all:
        print("Running all combinations of trials, models, and feature selection methods.")
        selected_trials = trials_info.keys()  # Use all trials
        selected_models = model_configs.keys()  # Use all models
        selected_features = feature_selection_methods  # Use all feature selection methods

    # Prepare the device data by creating a mapping of device IDs to the corresponding RF data.
    # X is the data and y contains the labels, which are already preprocessed.
    device_rf_data = {device: X[y == idx] for device, idx in device_id_mapping.items()}

    # Call the function to train models for each trial, model, and feature selection method.
    all_results, all_adr_results = train_model_per_trial(
        device_rf_data,
        {trial: trials_info[trial] for trial in selected_trials},  # Select the relevant trials
        {model: model_configs[model] for model in selected_models},  # Select the relevant models
        selected_features  # Use the selected feature selection methods
    )

if __name__ == "__main__":
    main()
