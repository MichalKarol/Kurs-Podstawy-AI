import os
import hydra
from omegaconf import OmegaConf
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import torch
from torch import optim, nn
import lightning as L
import torchmetrics as TM
from lightning.pytorch.loggers import WandbLogger
import wandb


class HouseDataModule:
    class HouseDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            assert len(self.x) == len(
                self.y
            ), "Number of values and labels are not equal"

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def __init__(self, data_path, val_split, test_split, seed):
        dataset = pd.read_csv(data_path)
        dataset.drop_duplicates(inplace=True)
        dataset.dropna(inplace=True)
        dataset = dataset.drop("Address", axis=1)
        y = dataset["Price"].to_numpy().astype(float).reshape(-1, 1)
        X = dataset.drop("Price", axis=1)
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_split, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=(val_split / (1 - test_split)), random_state=seed
        )

        y_scaler = RobustScaler()
        self.y_train = y_scaler.fit_transform(y_train)
        self.y_val = y_scaler.transform(y_val)
        self.y_test = y_scaler.transform(y_test)

        ct = ColumnTransformer(
            [
                (
                    "numerical",
                    RobustScaler(),
                    [
                        "Beds",
                        "Living Space",
                        "Zip Code Population",
                        "Zip Code Density",
                        "Median Household Income",
                        "Latitude",
                        "Longitude",
                    ],
                ),
                ("categories", OneHotEncoder(), ["State"]),
            ]
        )

        self.x_train = ct.fit_transform(X_train).toarray()
        self.x_val = ct.transform(X_val).toarray()
        self.x_test = ct.fit_transform(X_test).toarray()
        print("Example", self.x_test[0], self.y_test[0])

    @property
    def train(self):
        return HouseDataModule.HouseDataset(self.x_train, self.y_train)

    @property
    def val(self):
        return HouseDataModule.HouseDataset(self.x_val, self.y_val)

    @property
    def test(self):
        return HouseDataModule.HouseDataset(self.x_test, self.y_test)


# Definiujemy nasz moduł z kodem do uczenia modelu
class LightningRegression(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(36, 64, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(64, 64, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(64, 32, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(32, 1, dtype=torch.float64),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self.model(x)
        self.log("val_r2", TM.functional.r2_score(y_pred, y))
        self.log("val_mse", TM.functional.mean_squared_error(y_pred, y), prog_bar=True)
        self.log("val_mae", TM.functional.mean_absolute_error(y_pred, y))

    def test_step(self, batch):
        x, y = batch
        y_pred = self.model(x)
        self.log("test_r2", TM.functional.r2_score(y_pred, y))
        self.log("test_mse", TM.functional.mean_squared_error(y_pred, y), prog_bar=True)
        self.log("test_mae", TM.functional.mean_absolute_error(y_pred, y))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    config = OmegaConf.load("params.yaml")
    datamodule = HouseDataModule(
        config.dataset_path,
        config.val_split,
        config.test_split,
        config.seed,
    )
    train_dataloader = DataLoader(
        datamodule.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() - 1,
    )
    val_dataloader = DataLoader(
        datamodule.val, batch_size=config.batch_size, num_workers=os.cpu_count() - 1
    )
    test_dataloader = DataLoader(
        datamodule.test, batch_size=config.batch_size, num_workers=os.cpu_count() - 1
    )

    regression = LightningRegression()

    with wandb.init(project="Kurs AI") as run:
        wandb_logger = WandbLogger(log_model="all", name="Kurs AI", save_dir="logs")
        wandb_logger.experiment.config["max_epochs"] = config.max_epochs
        wandb_logger.experiment.config["batch_size"] = config.batch_size
        trainer = L.Trainer(max_epochs=config.max_epochs, logger=wandb_logger)
        trainer.fit(
            model=regression,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        trainer.test(regression, test_dataloader)
        run.finish()


if __name__ == "__main__":
    main()
