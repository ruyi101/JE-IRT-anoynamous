#!/bin/sh

seed=916
encoder="sentence-transformer"

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 12 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.75 \



python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 16 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.75 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 32 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.75 \

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 64 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.75 \

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 72 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 96 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 128 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 160 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \



python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 192 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.4 \
  --gamma 0.7 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 224 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 256 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 512 \
  --batch_size 512 \
  --lr 5e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.65 \


python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 768 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \

python train_hyper_select.py \
  --encoder $encoder \
  --seed $seed \
  --train_csv "../../train.csv" \
  --val_csv "../../val.csv" \
  --test_csv "../../test.csv" \
  --embed_dim 1024 \
  --batch_size 512 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --epochs 100 \
  --scheduler "step" \
  --output_path "./checkpoints/" \
  --dropout 0.5 \
  --gamma 0.7 \