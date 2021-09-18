import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
from model import SimsiamModel
from classifier import Classifier
import lightly.data as data
import time 
import argparse
from pathlib import Path
from dataloader import NoisyCIFAR10

parser = argparse.ArgumentParser(description='SimSiam Training')
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--max_epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--num_ftrs', default=512, type=int, metavar='N',
                    help='number of filters')
parser.add_argument('--backbone_model', default='resnet18', type=str,
                    help='backbone model architechture')
parser.add_argument('--noise_rate', default=0.0, type=float, help='the noise level')
parser.add_argument('--seed', default=0, type=int, help='seeding value')
parser.add_argument('--input_size', default=32,type=int,help='the size of your input')
parser.add_argument('--lr', default=0.05, type=float, help = 'initial learning rate for training')
parser.add_argument('--train_mode', default="simsiam", type=str, help = 'simsiam and simsiam_classifier')
args = parser.parse_args()


print("CODE STARTED AT: ", time.time())
# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

#seeding
pl.seed_everything(args.seed)

#loads cifar 10 from torchvision
# load cifar10 from torchvision
if args.train_mode == "simsiam":
    #base = torchvision.datasets.CIFAR10(root=args.data,train=True,download=False)
    #dataset_train_simsiam = data.LightlyDataset.from_torch_dataset(base)


    dataset_train_simsiam = NoisyCIFAR10(root=args.data, 
                        train=True, 
                        download=True, 
                        noise_rate=args.noise_rate
                        )
    dataset_train_simsiam = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_simsiam) 


    #Augmentation using the collate func
    
    collate_fn = lightly.data.ImageCollateFunction(
        input_size=32,
        # require invariance to flips and rotations
        hf_prob=0.5,
        vf_prob=0.5,
        rr_prob=0.5,
        # satellite images are all taken from the same height
        # so we use only slight random cropping
        min_scale=0.5,
        # use a weak color jitter for invariance w.r.t small color changes
        cj_prob=0.2,
        cj_bright=0.1,
        cj_contrast=0.1,
        cj_hue=0.1,
        cj_sat=0.1,
    )


    # The dataloader for this simsiam mode
    dataloader_train_simsiam = torch.utils.data.DataLoader(
        dataset_train_simsiam,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers
    )

if args.train_mode == "simsiam_classifier":
    # Augmentations typically used to train on cifar-10
    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])


    dataset_train_classifier = NoisyCIFAR10(root=args.data, 
                                       train=True, 
                                       download=True, 
                                       noise_type=args.noise_type, 
                                       noise_rate=args.noise_rate, 
                                       transform=train_classifier_transforms)

    dataset_test = NoisyCIFAR10(root=args.data, 
                                        train=False, 
                                        download=True,
                                        noise_rate=0.0,
                                        transform=test_transforms)

    dataloader_train_classifier = torch.utils.data.DataLoader(
                                            dataset_train_classifier,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=args.num_workers)

    dataloader_test = torch.utils.data.DataLoader(
                                            dataset_test,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=args.num_workers)


    #the dataloaders for the downstream task
    
    dataloader_train_classifier = torch.utils.data.DataLoader(
        dataset_train_classifier,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )



if args.train_mode == "simsiam":
    print(type(args.batch_size))
    print(type(args.input_size))
    print(type(args.lr))
    print(type(args.num_ftrs))
    print(type(args.max_epochs))
    model = SimsiamModel( args.batch_size, args.input_size, args.lr, args.num_ftrs, args.max_epochs, args.backbone_model)

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus,progress_bar_refresh_rate=100)
    trainer.fit(
        model,
        dataloader_train_simsiam)
    trainer.save_checkpoint("simsiam_ssl.ckpt")
    

if args.train_mode == "simsiam_classifier":
    # load SIMSIAM model
    model = SimsiamModel()
    assert args.checkpoint != None
    model = model.load_from_checkpoint(args.checkpoint)

    model.eval()
    classifier = Classifier(model.resnet_simsiam)
    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
    trainer.fit(
        classifier,
        dataloader_train_classifier,
        dataloader_test
    )
    trainer.save_checkpoint("example_classifier.ckpt")