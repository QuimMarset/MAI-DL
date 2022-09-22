#! /bin/bash

BASEDIR=./

# Check difference between Max and Average Pooling (both with size 2 and stride 2)

printf "\nRunning experiment 1\n"
python train.py --optimizer Adam --lr 1e-3 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 2\n"
python train.py --optimizer Adam --lr 1e-3 --conv_filters 16 32 64 --dense_units 128 --pooling_type Avg --epochs 80 --early_stopping 10

# Check difference in batch size

printf "\nRunning experiment 3\n"
python train.py --optimizer Adam --lr 1e-3 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 4\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 64 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 5\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

# Check difference in learning rate

printf "\nRunning experiment 6\n"
python train.py --optimizer Adam --lr 1e-4 --batch_size 128 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 7\n"
python train.py --optimizer Adam --lr 1e-3 --decay_rate 0.95 --decay_epochs 2 --batch_size 128 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 8\n"
python train.py --optimizer Adam --lr 1e-3 --decay_rate 0.95 --decay_epochs 1.3 --batch_size 128 --conv_filters 16 32 64 --dense_units 128 --pooling_type Max --epochs 80 --early_stopping 10

# Check difference when changing the number of convolution blocks

printf "\nRunning experiment 9\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 16 32 64 128 256 --dense_units 128 256 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 10\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 16 32 64 128 256 --dense_units 128 256 --pooling_type Max --epochs 80 --early_stopping 10

printf "\nRunning experiment 11\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 32 64 128 256 --dense_units 256 512 --pooling_type Max --epochs 80 --early_stopping 10

# Check difference when adding data augmentation

printf "\nRunning experiment 12\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 16 32 64 128 256 --dense_units 128 256 --pooling_type Max --epochs 80 --early_stopping 10 --augmentation

printf "\nRunning experiment 13\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 16 32 64 128 256 --dense_units 128 256 --pooling_type Max --epochs 80 --early_stopping 10 --augmentation

printf "\nRunning experiment 14\n"
python train.py --optimizer Adam --lr 1e-3 --batch_size 128 --conv_filters 32 64 128 256 --dense_units 256 512 --pooling_type Max --epochs 80 --early_stopping 10 --augmentation

python train.py --optimizer Adam --lr 1e-3 --batch_size 32 --conv_filters 16 32 64 96 --dense_units 128 --pooling_type Avg --epochs 90 --early_stopping --min_delta 1e-4 --patience 45  --activation relu --augmentation --lr_decay --decay_epochs 1.5 --decay_rate 0.98