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

    # Data preprocessing
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])
    # train = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    # train_data_mean = train.data.mean(axis=(0, 1, 2)) / 127
    # train_data_std = train.data.std(axis=(0, 1, 2)) / 127
    #
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(train_data_mean, train_data_std)
    # ])
    #
    # train = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
    # train, valid = random_split(train, [int(len(train) * 0.9), len(train) - int(len(train) * 0.9)])
    # train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=8)
    # valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=False, num_workers=8)

    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)
    dm.prepare_data()
    dm.setup(stage="fit")

    iteration = 64000
    total_epochs = int(iteration / len(dm.train_dataloader()))

    net = modlit.CIFARResnet(m)
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
        max_epochs=total_epochs,
        accelerator='cuda', logger=wandb_logger,
        callbacks=[lr_monitor, cp_callback], devices=1)
    trainer.fit(net, dm)
    x = torch.randn(1, 3, 32, 32)
    net.to_onnx(f"./model/model_{m}.onnx", x)


if __name__ == "__main__":
    main()
