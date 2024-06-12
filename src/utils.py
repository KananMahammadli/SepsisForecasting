import inspect
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl


def describe_object(obj):
    # Get the class name
    class_name = obj.__class__.__name__

    # Fetch the signature of the __init__ method
    signature = inspect.signature(obj.__class__.__init__)

    # Prepare a dictionary to collect parameter representations
    params_repr = {}

    for name, param in signature.parameters.items():
        # Skip 'self'
        if name == "self":
            continue

        # Get the default value of the parameter (if exists)
        default_value = param.default if param.default != inspect.Parameter.empty else None

        # Get the current value of the parameter in the object
        current_value = getattr(obj, name, None)

        # Check if the current value is different from the default
        if current_value is not None and current_value != default_value:
            params_repr[name] = current_value
        else:
            params_repr[name] = default_value

    # Construct the representation string
    params_str = ", ".join([f"{k}={v}" for k, v in params_repr.items()])

    return f"{class_name}({params_str})"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def split(df, patient_id_col, target_column, random_state, train_ratio, val_ratio, test_ratio):
    # Identify patients with only 0s and those with both 0s and 1s
    patient_groups = df.groupby(patient_id_col)[target_column].agg(["min", "max"])
    patients_with_only_zeros = patient_groups[patient_groups["max"] == 0].index.tolist()
    patients_with_zeros_and_ones = patient_groups[patient_groups["max"] > 0].index.tolist()

    # Function to split each group into train, validation, and test sets
    def stratified_split(patients, train_ratio, val_ratio, test_ratio, random_state):
        train_patients, val_test_patients = train_test_split(
            patients, train_size=train_ratio, random_state=random_state, shuffle=True
        )
        val_patients, test_patients = train_test_split(
            val_test_patients,
            train_size=val_ratio / (val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True,
        )
        return train_patients, val_patients, test_patients

    # Split the patients with only 0s
    train_patients_zeros, val_patients_zeros, test_patients_zeros = stratified_split(
        patients_with_only_zeros, train_ratio, val_ratio, test_ratio, random_state
    )

    # Split the patients with both 0s and 1s
    (train_patients_zeros_and_ones, val_patients_zeros_and_ones, test_patients_zeros_and_ones) = (
        stratified_split(
            patients_with_zeros_and_ones, train_ratio, val_ratio, test_ratio, random_state
        )
    )

    # Combine the patients from both groups for each split
    train_patients = train_patients_zeros + train_patients_zeros_and_ones
    val_patients = val_patients_zeros + val_patients_zeros_and_ones
    test_patients = test_patients_zeros + test_patients_zeros_and_ones

    # Create train, validation, and test datasets
    train_data = df[df[patient_id_col].isin(train_patients)]
    val_data = df[df[patient_id_col].isin(val_patients)]
    test_data = df[df[patient_id_col].isin(test_patients)]

    return train_data, val_data, test_data
