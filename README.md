# Effects of Noisy labels on SSL models

## Description
This project tries to investigate the resiliency of certain SSL models to label noise on CIFAR-10. 
Specifically, we train SIMSIAM and BYOL using the lightly library on CIFAR-10 (unlabled). Then, we used the learnt representations onto a downstream task of linear classification. The training of this linear classifier is performed using noisy CIFAR-10 data with different levels and types (symmetric or assymetric) of noise. Finally, to see the behaviour of these models, we evaluate on the noise free CIFAR10-dataset. 

### Training SIMSIAM (SSL)
```
## using resnet-18
python3 train_noisy_cifar10.py --data "data/" \
--train_mode simsiam \
--noise_rate 0.0 \
--seed 0 \
--input_size 32 \
--max_epochs_ssl 800 \
--batch_size_ssl 512 \
--num_ftrs_ssl 512 \
--lr_ssl 0.06 
```

### Training BYOL (SSL) 

```
## using resnet-18
python3 train_noisy_cifar10_byol.py --data "data/" \
--train_mode byol_ssl \
--noise_rate 0.0 \
--seed 0 \
--input_size 32 \
--max_epochs_ssl 200 \
--batch_size_ssl 512 \
--num_ftrs_ssl 512 \
--lr_ssl 0.06 
```


### Training downstream task: Linear classifier
```
## Using SIMSIAM 
python3 train_noisy_cifar10.py --data "data/" \
--train_mode simsiam_classifier \
--noise_rate 0.1 \
--noise_type sym \
--seed 0 \
--input_size 32 \
--max_epochs_ssl 800 \
--batch_size_ssl 512 \
--num_ftrs_ssl 512 \
--lr_ssl 0.06 \
--checkpoint simsiam_ssl.ckpt \
--lr_clf 30.0 \
--batch_size_clf 512 \
--max_epochs_clf 100 \
--num_ftrs_clf 512 \


## USing BYOL 
python3 train_noisy_cifar10_byol.py --data "data/" \
--train_mode byol_classifier \
--noise_rate 0.1 \
--noise_type sym \
--seed 0 \
--input_size 32 \
--max_epochs_ssl 800 \
--batch_size_ssl 512 \
--num_ftrs_ssl 512 \
--lr_ssl 0.06 \
--checkpoint byol_ssl.ckpt \
--lr_clf 30.0 \
--batch_size_clf 512 \
--max_epochs_clf 100 \
--num_ftrs_clf 512 \

```

