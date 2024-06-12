from loss import CustomBceLossWithWeights
from utils import describe_object

RAW_DATA_PATH = "data/df_raw_combined.csv"
PROJECT = "MedicalDL/SepsisForecasting"

VITAL_SIGNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LAB_FEATURES = [
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "HospAdmTime"]
TARGET_COL = "is_sepsis"
TEMPORARY_TARGET_COL = "SepsisLabel"
PATIENT_ID_COL = "patient_id"

RANDOM_STATE = 42
IS_TRAIN = False
WINDOW_LENGTH = 12
BATCH_SIZE = 256
OUTPUT_LENGTH = 7
N_MIN_ROWS_THRESHOLD = 60
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

age_col = "Age"

METRIC_AND_LOSS_NAMES = [
    "test_loss",
    "test_BinaryF1Score",
    "test_BinaryAccuracy",
    "test_BinaryPrecision",
    "test_BinaryRecall",
    "test_BinarySpecificity",
    "test_BinaryAUROC",
]

STATIC_FEATURES_SIZE = len(DEMOGRAPHICS)

# Define your data and model parameters
OUTPUT_SIZE = OUTPUT_LENGTH
EPOCHS = 45
PATIENCE = 45

CONV_KERNEL_SIZE = 3
PADDING = 1
CONV_OUTPUT_SIZE = 32
HIDDEN_SIZE = WINDOW_LENGTH * CONV_OUTPUT_SIZE

NUM_LAYERS = 2
DROPOUT = 0.1
BIDIRECTIONAL = False

FC1_SIZE = 32
FC2_SIZE = 64
FC3_SIZE = 100
EMBED_SIZE = 128

POS_WEIGHT = 55
SAMPLE_WIGHTS = [20 * (i + 1) for i in range(OUTPUT_LENGTH)]

LOSS_FN = CustomBceLossWithWeights(sample_weights=SAMPLE_WIGHTS, pos_weight=POS_WEIGHT)

OPTIMIZER_NAME = "Adam"
OPTIMIZER_PARAMS = {}
LR_SCHEDULER_NAME = "ExponentialLR"
LR_SCHEDULER_PARAMS = {"gamma": 0.1}
LOG_BEST_CKPT_WEIGHTS = True


PARAMS = {
    "n_min_rows_threshold": N_MIN_ROWS_THRESHOLD,
    "is_train": IS_TRAIN,
    "window_length": WINDOW_LENGTH,
    "output_length": OUTPUT_LENGTH,
    "fc1_size": FC1_SIZE,
    "fc2_size": FC2_SIZE,
    "fc3_size": FC3_SIZE,
    "embed_size": EMBED_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "bidirectional": BIDIRECTIONAL,
    "static_features_size": STATIC_FEATURES_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "random_state": RANDOM_STATE,
    "pos_weight": POS_WEIGHT,
    "sample_weights": SAMPLE_WIGHTS,
    "loss_fn": describe_object(LOSS_FN),
    "optimizer_name": OPTIMIZER_NAME,
    "optimizer_params": OPTIMIZER_PARAMS,
    "lr_scheduler_name": LR_SCHEDULER_NAME,
    "lr_scheduler_params": LR_SCHEDULER_PARAMS,
    "log_best_ckpt_weights": LOG_BEST_CKPT_WEIGHTS,
    "conv_outout_size": CONV_OUTPUT_SIZE,
    "conv_kernel_size": CONV_KERNEL_SIZE,
    "padding": PADDING,
}

DESCRIPTION = f"""
train_data_type_is_train-{IS_TRAIN},
window_length-{WINDOW_LENGTH},
batch_size-{BATCH_SIZE},
n_min_rows_threshold-{N_MIN_ROWS_THRESHOLD},
split_fold_taken-1,
tuning_model_params,
hidden_size-{HIDDEN_SIZE},
num_layers-{NUM_LAYERS},
dropout-{DROPOUT},
loss_fn-{describe_object(LOSS_FN)},
feature_engineering_with_starting_7_lag_rolling_all_diff,
no_filter_age_between_18_100,
fillna(0)_after_fe,
no_oiculus_feature,
embedding_with_linear_layers,
embed_concat_passed_into_lstm_as_input
conv_output_passed_as_initial_hidden_state,
fc1_size-{FC1_SIZE},
fc2_size-{FC2_SIZE},
fc3_size-{FC3_SIZE},
embed_size-{EMBED_SIZE},
conv_outout_size-{CONV_OUTPUT_SIZE},
conv_kernel_size-{CONV_KERNEL_SIZE},
"""
