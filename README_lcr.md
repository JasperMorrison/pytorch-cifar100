CUDA_VISIBLE_DEVICES=1 python train.py -net resnet18 -gpu -data data/train.csv -test data/val.csv -classes 2 -warm 3 -b 64 -lr 0.01 -focal_loss -pretrained

dataset tree:
data
├── train
│   ├── cat_1.jpg
│   └── dog_1.jpg
├── train.csv
├── val
│   ├── cat_1.jpg
│   └── dog_1.jpg
└── val.csv

csv file format:(one-shot)
image,cat,dog
cat_1.jpg,1,0
dog_1.jpg,0,1
