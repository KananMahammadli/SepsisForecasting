import pandas as pd
from utils import set_seed, split
from etl_pipeline import preprocess_data, prepare_data
from data_loader import EHRSepsisDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import LSTMModel

from config import (
    BATCH_SIZE,
    BIDIRECTIONAL,
    CONV_KERNEL_SIZE,
    CONV_OUTPUT_SIZE,
    DESCRIPTION,
    DROPOUT,
    EMBED_SIZE,
    EPOCHS,
    FC1_SIZE,
    FC2_SIZE,
    FC3_SIZE,
    HIDDEN_SIZE,
    LOSS_FN,
    LR_SCHEDULER_NAME,
    LR_SCHEDULER_PARAMS,
    NUM_LAYERS,
    OPTIMIZER_NAME,
    OPTIMIZER_PARAMS,
    OUTPUT_SIZE,
    PADDING,
    PARAMS,
    PATIENCE,
    PROJECT,
    RAW_DATA_PATH,
    PATIENT_ID_COL,
    STATIC_FEATURES_SIZE,
    TEMPORARY_TARGET_COL,
    N_MIN_ROWS_THRESHOLD,
    DEMOGRAPHICS,
    VITAL_SIGNS,
    LAB_FEATURES,
    OUTPUT_LENGTH,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    TARGET_COL,
    IS_TRAIN,
    WINDOW_LENGTH,
)


def prepare_datamodule():
    df_raw = pd.read_csv(RAW_DATA_PATH)

    df_preprocessed = preprocess_data(
        df=df_raw,
        patient_id_col=PATIENT_ID_COL,
        temporary_target_col=TEMPORARY_TARGET_COL,
        n_min_rows_threshold=N_MIN_ROWS_THRESHOLD,
        vital_signs=VITAL_SIGNS,
        lab_features=LAB_FEATURES,
        output_length=OUTPUT_LENGTH,
    )

    train_data, val_data, test_data = split(
        df=df_preprocessed,
        patient_id_col=PATIENT_ID_COL,
        target_column=TEMPORARY_TARGET_COL,
        random_state=RANDOM_STATE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    static_features_cols = DEMOGRAPHICS
    dynamic_features_cols = list(
        set(df_preprocessed.columns).difference(
            set(static_features_cols + [PATIENT_ID_COL, TARGET_COL, TEMPORARY_TARGET_COL])
        )
    )

    input_size = len(dynamic_features_cols)

    df_train = prepare_data(
        df=train_data,
        patient_id_col=PATIENT_ID_COL,
        temporary_target_col=TEMPORARY_TARGET_COL,
        target_col=TARGET_COL,
        features=dynamic_features_cols + static_features_cols,
        is_train=IS_TRAIN,
    )
    df_val = prepare_data(
        df=val_data,
        patient_id_col=PATIENT_ID_COL,
        temporary_target_col=TEMPORARY_TARGET_COL,
        target_col=TARGET_COL,
        features=dynamic_features_cols + static_features_cols,
        is_train=False,
    )

    df_test = prepare_data(
        df=test_data,
        patient_id_col=PATIENT_ID_COL,
        temporary_target_col=TEMPORARY_TARGET_COL,
        target_col=TARGET_COL,
        features=dynamic_features_cols + static_features_cols,
        is_train=False,
    )

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    data_module = EHRSepsisDataModule(
        df=df,
        train_patients=df_train[PATIENT_ID_COL].unique().tolist(),
        val_patients=df_val[PATIENT_ID_COL].unique().tolist(),
        test_patients=df_test[PATIENT_ID_COL].unique().tolist(),
        window_length=WINDOW_LENGTH,
        dynamic_features_cols=dynamic_features_cols,
        static_features_cols=static_features_cols,
        output_length=OUTPUT_LENGTH,
        batch_size=BATCH_SIZE,
        target_col=TARGET_COL,
        patient_id_col=PATIENT_ID_COL,
    )

    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    return (
        data_module,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        input_size,
        static_features_cols,
        dynamic_features_cols,
    )


def run_model():
    (
        data_module,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        input_size,
        static_features_cols,
        dynamic_features_cols,
    ) = prepare_datamodule()

    PARAMS["input_size"] = input_size
    PARAMS["static_features"] = ", ".join(static_features_cols)
    PARAMS["dynamic_features"] = ", ".join(dynamic_features_cols)

    neptune_logger = pl.loggers.neptune.NeptuneLogger(
        project=PROJECT,
        description=DESCRIPTION,
        tags=["lstm", "sepsis", "prediction"],
        capture_hardware_metrics=True,
        mode="async",
    )

    # Early Stopping and Model Checkpoint Callbacks
    early_stop_callback = EarlyStopping(
        monitor="validation_BinaryF1Score", patience=PATIENCE, verbose=True, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_BinaryF1Score", mode="max", save_top_k=1, verbose=True
    )

    model = LSTMModel(
        input_size=input_size,
        fc1_size=FC1_SIZE,
        fc2_size=FC2_SIZE,
        fc3_size=FC3_SIZE,
        embed_size=EMBED_SIZE,
        conv_output_size=CONV_OUTPUT_SIZE,
        conv_kernel_size=CONV_KERNEL_SIZE,
        padding=PADDING,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        output_size=OUTPUT_SIZE,
        static_features_size=STATIC_FEATURES_SIZE,
        loss_fn=LOSS_FN,
        optimizer_name=OPTIMIZER_NAME,
        optimizer_params=OPTIMIZER_PARAMS,
        lr_scheduler_name=LR_SCHEDULER_NAME,
        lr_scheduler_params=LR_SCHEDULER_PARAMS,
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=neptune_logger,
        deterministic=False,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = LSTMModel.load_from_checkpoint(
        best_model_path,
        input_size=input_size,
        fc1_size=FC1_SIZE,
        fc2_size=FC2_SIZE,
        fc3_size=FC3_SIZE,
        embed_size=EMBED_SIZE,
        conv_output_size=CONV_OUTPUT_SIZE,
        conv_kernel_size=CONV_KERNEL_SIZE,
        padding=PADDING,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        output_size=OUTPUT_SIZE,
        static_features_size=STATIC_FEATURES_SIZE,
        loss_fn=LOSS_FN,
        optimizer_name=OPTIMIZER_NAME,
        optimizer_params=OPTIMIZER_PARAMS,
        lr_scheduler_name=LR_SCHEDULER_NAME,
        lr_scheduler_params=LR_SCHEDULER_PARAMS,
    )

    # Test the model
    trainer.test(best_model, dataloaders=data_module.test_dataloader())

    neptune_logger.experiment["params"] = PARAMS
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    neptune_logger.experiment.stop()


if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    run_model()
