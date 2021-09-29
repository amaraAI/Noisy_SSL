import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
from model import SimsiamModel, BYOLModel, MocoModel
from classifier import Classifier
import lightly.data as data
import time 
import argparse
from pathlib import Path
from dataloader import NoisyCIFAR10
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='MOCO Training')
#shared parameters
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--backbone_model', default='resnet18', type=str,
                    help='backbone model architechture')
parser.add_argument('--noise_rate', default=0.0, type=float, help='the noise level')
parser.add_argument('--noise_type',default='sym',type=str, help='the type fo noise')
parser.add_argument('--seed', default=0, type=int, help='seeding value')
parser.add_argument('--input_size', default=32,type=int,help='the size of your input')
parser.add_argument('--train_mode', default="moco", type=str, help = 'moco and moco_classifier')

# parameters related to SSL training 
parser.add_argument('--max_epochs_ssl', default=1200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size_ssl', default=512, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--num_ftrs_ssl', default=512, type=int, metavar='N',
                    help='number of filters')
parser.add_argument('--lr_ssl', default=0.06, type=float, help = 'initial learning rate for training ssl')
parser.add_argument('--temperature', default=0.5, type = float)
parser.add_argument('--memory_bank_size', default =512 , type =int)
parser.add_argument('--dir_path' , type =str)

#paramters related to classifier training
parser.add_argument('--noisydata', default=True,type=bool,help='if True the data is from a pkl file')
parser.add_argument('--checkpoint', default="checkpoint_ssl.ckpt",type=str)
parser.add_argument('--max_epochs_clf', default=300, type=int, metavar='N',
                    help='number of total epochs to run for the classsifier')
parser.add_argument('--batch_size_clf', default=256, type=int, metavar='N',
                    help='mini-batch size for the clf training')
parser.add_argument('--num_ftrs_clf', default=512, type=int, metavar='N',
                    help='size of fc layer ')           
parser.add_argument('--lr_clf', default=0.1, type=float, help = 'initial learning rate for training the classifier ')
parser.add_argument('--savename_clf', default="checkpoint_clf.ckpt",type=str)
args = parser.parse_args()


print("CODE STARTED AT: ", time.time())
# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

#seeding
pl.seed_everything(args.seed)

#loads cifar 10 from torchvision
# load cifar10 from torchvision
if args.train_mode == "moco":
    base = torchvision.datasets.CIFAR10(root=args.data,train=True,download=False)
    dataset_train = data.LightlyDataset.from_torch_dataset(base)

    # MoCo v2 uses SimCLR augmentations, additionally, disable blur
    collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
        )

    # The dataloader for this byol ssl mode
    dataloader_train_moco = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_ssl,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers
    )




if args.train_mode == "moco_classifier":
    # Augmentations typically used to train on cifar-10
    

    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    

    # No additional augmentations for the test set
    # '~projects/def-jjclark/shared_data/CIFAR10_noisy_checkpoints/cifar10_noise_sym_0.1.pkl' 
    if args.noisydata == True:
        path_ = '/home/iamara/projects/def-jjclark/shared_data/CIFAR10_noisy_checkpoints/cifar10_noise_'+args.noise_type+'_'+str(args.noise_rate)+'.pkl'
        dataset_train_classifier_ = NoisyCIFAR10.load_(path_)
        dataset_train_classifier = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_classifier_) 

    else: 
        dataset_train_classifier = NoisyCIFAR10(root=args.data, 
                                        train=True, 
                                        download=True,
                                        noise_type=args.noise_type, 
                                        noise_rate=args.noise_rate,
                                        transform=train_classifier_transforms)
    


    base_test = torchvision.datasets.CIFAR10(root=args.data,train=False,download=False,transform=test_transforms)
    dataset_test = data.LightlyDataset.from_torch_dataset(base_test)
        
        

    dataloader_train_classifier = torch.utils.data.DataLoader(
                                            dataset_train_classifier,
                                            batch_size=args.batch_size_clf,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=args.num_workers)

    dataloader_test = torch.utils.data.DataLoader(
                                            dataset_test,
                                            batch_size=args.batch_size_clf,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=args.num_workers)



if args.train_mode == "moco":

    model = MocoModel(batch_size=args.batch_size_ssl,input_size=args.input_size,lr=args.lr_ssl,num_ftrs=args.num_ftrs_ssl,max_epochs=args.max_epochs_ssl, temperature=args.temperature,memory_bank_size=args.memory_bank_size, momentum=0.9, weight_decay=5e-4)
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=args.dir_path, filename='moco_{epoch}', save_last=True, save_top_k=-1) 
    trainer = pl.Trainer(max_epochs=args.max_epochs_ssl, gpus=gpus, progress_bar_refresh_rate=10,default_root_dir='Saved_models/',callbacks=[checkpoint_callback])
    trainer.fit(model,dataloader_train_moco)    

if args.train_mode == "moco_classifier":
    model = MocoModel()
    assert args.checkpoint != None
    model = model.load_from_checkpoint(args.checkpoint)
    model.eval()
    classifier = Classifier(model.resnet_moco, lr=args.lr_clf, max_epochs=args.max_epochs_clf,numftrs_clf=args.num_ftrs_clf,num_classes=10)
    
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=args.dir_path, filename='classifier_moco_noisy_before_projection_{epoch}', save_last=True, save_top_k=-1)
    trainer = pl.Trainer(max_epochs=args.max_epochs_clf, gpus=gpus, 
                         progress_bar_refresh_rate=10,
                         default_root_dir='Saved_models/',
                         callbacks=[checkpoint_callback])
    trainer.fit(
        classifier,
        dataloader_train_classifier,
        dataloader_test
    )
    #trainer.save_checkpoint(args.savename_clf)