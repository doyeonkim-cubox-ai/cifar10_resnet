import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import lightning as L
from cifar10_resnet.modlit import CIFARResnet
from cifar10_resnet.data import CIFAR10DataModule
import argparse


def main():
    # Add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.resnet20)')
    m = parser.parse_args().model

    # Data preprocessing
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomCrop((32, 32))
    #     ])
    # test = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=8)

    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)
    dm.prepare_data()
    dm.setup(stage="test")

    checkpoint = f"./model/model_{m}.ckpt"
    net = CIFARResnet.load_from_checkpoint(checkpoint, m=m)
    trainer = L.Trainer(accelerator='cuda', devices=1)
    trainer.test(net, dm)


if __name__ == "__main__":
    main()
