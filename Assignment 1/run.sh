#! /bin/bash

BASEDIR=./


printf "\nRunning experiment 1\n"
#python train.py --optimizer Adam --lr 1e-4 --conv_filters 8 16 32 --dense_units 64 --pooling_type Avg --epochs 60


printf "\nRunning experiment 2\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 8 16 32 --dense_units 64 --pooling_type Avg --epochs 60

printf "\nRunning experiment 2\n"
#python train.py --optimizer Adam --lr 1e-4 --conv_filters 8 16 32 --dense_units 64 --pooling_type Avg --batch_norm --augmentation --epochs 60

printf "\nRunning experiment 3\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 512 --pooling_type Max --augmentation --epochs 50 --early_stop 10

#python train.py --optimizer Adam --lr 1e-3 --kernel_sizes 5 3 3 --conv_filters 32 64 128 --dense_units 512 --pooling_type Max --augmentation --epochs 50 --early_stop 10

#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 512 --pooling_type Max --augmentation --batch_norm --epochs 50 --early_stop 10

python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 512 --pooling_type Max --augmentation --batch_norm --dropout 0.2 --epochs 50 --early_stop 10

#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 512 --pooling_type Max --augmentation --epochs 50 --early_stop 10

#python train.py --optimizer Adam --lr 1e-3 --conv_filters 8 16 32 --dense_units 64 --pooling_type Avg --batch_norm --dropout 0.25 --augmentation --epochs 30

printf "\nRunning experiment 4\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 256 --pooling_type Avg --batch_norm --dropout 0.2 --epochs 30

printf "\nRunning experiment 5\n"
#python train.py --optimizer Adam --lr 1e-4 --conv_filters 32 64 128 --dense_units 256 --pooling_type Avg --batch_norm --dropout 0.2 --epochs 30

printf "\nRunning experiment 6\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 256 --pooling_type Avg --batch_norm --epochs 30 --weight_decay 0.001

printf "\nRunning experiment 7\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 256 --pooling_type Avg --batch_norm --epochs 30 --weight_decay 0.001 --augmentation

printf "\nRunning experiment 8\n"
#python train.py --optimizer Adam --lr 1e-3 --conv_filters 32 64 128 --dense_units 256 --pooling_type Avg --batch_norm --dropout 0.2 --epochs 30 --augmentation