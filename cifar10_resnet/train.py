import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import wandb
from cifar10_resnet import modlit
from cifar10_resnet.data import CIFAR10DataModule
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse


def main():
    # Add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.resnet20)')
    m = parser.parse_args().model

    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)
    dm.prepare_data()
    dm.setup(stage="fit")

    iteration = 64000

    net = modlit.CIFARResnet(m)

    # print(net)
    # exit(0)
    wandb_logger = WandbLogger(log_model=False, name=f'{m}', project='cifar10_resnet')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    cp_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation loss",
        mode="min",
        dirpath="./model/",
        filename=f"model_{m}"
    )
    trainer = L.Trainer(
        max_steps=iteration,
        accelerator='cuda', logger=wandb_logger,
        callbacks=[lr_monitor, cp_callback], devices=1)
    trainer.fit(net, dm)


if __name__ == "__main__":
    main()
