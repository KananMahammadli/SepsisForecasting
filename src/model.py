import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        output_size,
        static_features_size,
        loss_fn,
        optimizer_name,
        optimizer_params,
        lr_scheduler_name,
        lr_scheduler_params,
        fc1_size,
        fc2_size,
        fc3_size,
        embed_size,
        conv_output_size,
        conv_kernel_size,
        padding,
    ):
        self.num_layers = num_layers
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(input_size, fc2_size)
        self.fc3 = nn.Linear(input_size, fc3_size)

        self.fc4 = nn.Linear(fc1_size + fc2_size + fc3_size, embed_size)

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_output_size,
            kernel_size=conv_kernel_size,
            padding=padding,
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size + static_features_size, output_size)
        self.loss_fn = loss_fn
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_params = lr_scheduler_params

        # loss function is inherited from nn.Module, and already saved during checkpointing
        # so it is advised to ignore it in save_hyperparameters
        self.save_hyperparameters(ignore=["loss_fn"])

        # Metrics
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(task="binary"),
                torchmetrics.Accuracy(task="binary"),
                torchmetrics.Precision(task="binary"),
                torchmetrics.Recall(task="binary"),
                torchmetrics.Specificity(task="binary"),
                torchmetrics.AUROC(task="binary"),
            ]
        )

        self.training_metrics = metrics.clone(prefix="training_")
        self.validation_metrics = metrics.clone(prefix="validation_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x, demographics):
        embed1 = F.relu(self.fc1(x))
        embed2 = F.relu(self.fc2(x))
        embed3 = F.relu(self.fc3(x))
        embed_concat = torch.cat((embed1, embed2, embed3), dim=2)
        embed = self.fc4(embed_concat)

        # Permute input to match Conv1D expected input shape (batch_size, channels, length)
        x_permuted = x.permute(0, 2, 1)  # (batch_size, num_features, window_length)
        conv_out = self.conv(x_permuted)
        # Permute the output back to (window_length, batch_size, num_filters)
        conv_out_permuted_back = conv_out.permute(
            2, 0, 1
        )  # (window_length, batch_size, num_filters)

        h_0 = conv_out_permuted_back.contiguous().view(x.shape[0], -1).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        lstm_out, _ = self.lstm(embed, (h_0, c_0))
        last_hidden_state = lstm_out[:, -1, :]

        concatenated_features = torch.cat((last_hidden_state, demographics), dim=1)
        output = self.fc(concatenated_features)
        return output

    def training_step(self, batch, batch_idx):
        x, static_data, y = batch
        y_hat = self(x, static_data)
        loss = self.loss_fn(y_hat, y.float())

        # Metrics
        metrics = self.training_metrics(y_hat, y)

        # Log loss and metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, static_data, y = batch
        y_hat = self(x, static_data)
        val_loss = self.loss_fn(y_hat, y.float())

        # Metrics
        metrics = self.validation_metrics(y_hat, y)

        # Log loss and metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, static_data, y = batch
        y_hat = self(x, static_data)
        test_loss = self.loss_fn(y_hat, y.float())

        # Metrics
        metrics = self.test_metrics(y_hat, y)

        # Log loss and metrics
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        # Define optimizer and scheduler (if any)
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), **self.optimizer_params
        )
        scheduler = (
            getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                optimizer, **self.lr_scheduler_params
            )
            if self.lr_scheduler_name
            else None
        )

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        return ([optimizer], [scheduler]) if scheduler else optimizer

    def on_after_backward(self):
        # Log gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment["gradients/" + name].log(param.grad.norm().item())
