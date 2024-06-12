import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class EHRSepsisDataset(Dataset):
    def __init__(
        self,
        data,
        patient_ids,
        target_column,
        patient_id_col,
        window_length,
        dynamic_features_cols,
        static_features_cols,
        output_length,
    ):
        self.data = data
        self.patient_ids = patient_ids
        self.target_column = target_column
        self.patient_id_col = patient_id_col
        self.window_length = window_length
        self.dynamic_features_cols = dynamic_features_cols
        self.static_features_cols = static_features_cols
        self.output_length = output_length
        self.sequences = self._prepare_sequences()

    def _prepare_sequences_for_patient(self, patient_id):
        patient_data = self.data[self.data[self.patient_id_col] == patient_id].reset_index(
            drop=True
        )
        dynamic_data = patient_data.loc[:, self.dynamic_features_cols]
        static_data = patient_data.loc[0, self.static_features_cols].values
        sequences = []

        for i in range(len(patient_data) - self.window_length - self.output_length + 1):
            X = dynamic_data.loc[i : i + self.window_length - 1, :].values
            y = patient_data.loc[
                i + self.window_length : i + self.window_length - 1 + self.output_length,
                self.target_column,
            ].values

            assert (
                X.shape[0] == self.window_length
            ), f"X shape: {X.shape}, window length: {self.window_length}"
            assert X.shape[1] == len(
                self.dynamic_features_cols
            ), f"X shape: {X.shape}, dynamic_features_cols: {self.dynamic_features_cols}"
            assert (
                y.shape[0] == self.output_length
            ), f"y shape: {y.shape}, output length: {self.output_length}"
            sequences.append((X, static_data, y))

        return sequences

    def _prepare_sequences(self):
        with Pool(cpu_count()) as pool:
            result = pool.map(self._prepare_sequences_for_patient, self.patient_ids)

        # Flatten the list of lists
        sequences = [item for sublist in result for item in sublist]
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, static_data, y = self.sequences[idx]
        X = X.astype(np.float32)
        static_data = static_data.astype(np.float32)
        y = y.astype(np.float32)
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(static_data, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class EHRSepsisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        train_patients,
        val_patients,
        test_patients,
        window_length,
        dynamic_features_cols,
        static_features_cols,
        target_col,
        patient_id_col,
        output_length,
        batch_size,
    ):
        super().__init__()
        self.df = df
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients
        self.window_length = window_length
        self.dynamic_features_cols = dynamic_features_cols
        self.static_features_cols = static_features_cols
        self.target_col = target_col
        self.patient_id_col = patient_id_col
        self.output_length = output_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EHRSepsisDataset(
            data=self.df,
            patient_ids=self.train_patients,
            target_column=self.target_col,
            patient_id_col=self.patient_id_col,
            window_length=self.window_length,
            dynamic_features_cols=self.dynamic_features_cols,
            static_features_cols=self.static_features_cols,
            output_length=self.output_length,
        )

        self.val_dataset = EHRSepsisDataset(
            data=self.df,
            patient_ids=self.val_patients,
            target_column=self.target_col,
            patient_id_col=self.patient_id_col,
            window_length=self.window_length,
            dynamic_features_cols=self.dynamic_features_cols,
            static_features_cols=self.static_features_cols,
            output_length=self.output_length,
        )

        self.test_dataset = EHRSepsisDataset(
            data=self.df,
            patient_ids=self.test_patients,
            target_column=self.target_col,
            patient_id_col=self.patient_id_col,
            window_length=self.window_length,
            dynamic_features_cols=self.dynamic_features_cols,
            static_features_cols=self.static_features_cols,
            output_length=self.output_length,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
