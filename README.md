# ProSub

Official repository for ECCV2024 paper [ProSub: Probabilistic Open-Set Semi-Supervised Learning with Subspace-Based Out-of-Distribution Detection](https://arxiv.org/abs/2407.11735)

## Requirements

Python requirements are specified in requirements.txt.

**Optional**: [Dockerfile](Dockerfile) defines a working docker environment for running this code.

## Preparation

Make sure the prosub directory is in your Python path when running this code:
```bash
REPODIR=/path/to/this/repo
export PYTHONPATH=$PYTHONPATH:$REPODIR
```

Set bash variables specifying where to store data and training results:
```bash
DATADIR="/directory/for/storing/data"
TRAINDIR="/directory/for/storing/checkpoints/and/results"
```

Create the data directory if it does not already exist:
```bash
mkdir -p $DATADIR
```

**Optional**: set logging level = 1 to disable info messages
```bash
export TF_CPP_MIN_LOG_LEVEL=1
```
(0 prints all messages, 1 disables info messages, 2 disables info & warning messages, 3 disables all messages)


## Datasets

For CIFAR-10, CIFAR-100, and TinyImageNet, we use tfrecord files. For ImageNet30 and ImageNet100 we read the data from image files. This section describes how to download and prepare the datasets.

### ImageNet30
Download the tar files following the links named "ImageNet-30-train" and "ImageNet-30-test" in the [CSI repo](ttps://github.com/alinlab/CSI).

Extract the files:
```bash
mkdir $DATADIR/imagenet30
tar -xvf one_class_train.tar -C $DATADIR/imagenet30
tar -xvf one_class_test.tar -C $DATADIR/imagenet30
```

### ImageNet100
Download the zip-file from the [Kaggle page](https://www.kaggle.com/datasets/ambityga/imagenet100).

Extract the files and rename the directories:
```bash
unzip imagenet100.zip -d $DATADIR/imagenet100
cd $DATADIR/imagenet100
mkdir train
mv train.X1/* train.X2/* train.X3/* train.X4/* train/
mv val.X val
rm -r train.X*
```

The following script downloads the data and prepares the tfrecord files for CIFAR-10, CIFAR-100, and TinyImageNet. It also checks if the ImageNet datasets are correctly installed.
```bash
python3 $REPODIR/scripts/create_datasets.py \
--datadir=$DATADIR \
--repodir=$REPODIR
```

### Labeled subsets

To create the labeled subsets, run e.g.
```bash
python3 $REPODIR/scripts/create_split.py \
--seed=1 \
--size=5000 \
$DATADIR/SSL2/tinyimagenet-id \
$DATADIR/tinyimagenet-id-train.tfrecord
```
which creates a labeled subset from the tinyimagenet-id-train records with 5,000 samples using random seed 1. It generates the the file `tinyimagenet-id.1@5000-label.tfrecord` in the directory `$DATADIR/SSL2`.

The corresponding command for CIFAR is e.g.
```bash
python3 $REPODIR/scripts/create_split.py \
--seed=1 \
--size=4000 \
$DATADIR/SSL2/cifar10 \
$DATADIR/cifar10-train.tfrecord
```

We do not need to run `scripts/create_split.py` for ImageNet30 and ImageNet100. The labeled subsets for these datasets are defined in the text files `$REPODIR/data-files/imagenet100-id.0@5000-label.txt` and `$REPODIR/data-files/imagenet30-id.0@2600-label.txt`.

## Training

Here are examples of how to run ProSub for the different datasets with the configurations used for the results in the paper.

**CIFAR-10**
```bash
python3 $REPODIR/prosub_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=50000 \
--pretrainsteps=$((2**19)) \
--dataset=cifar10 \
--datasetood=cifar100 \
--datasetunseen=cifar100 \
--nlabeled=4000 \
--ws=10.0 \
--arch=WRN-28-2 \
--seed=1
```
**CIFAR-100**
```bash
DECAYFACTOR=$(bc <<< "scale=4; 5/8")
python3 $REPODIR/prosub_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=50000 \
--pretrainsteps=$((2**19)) \
--dataset=cifar100 \
--datasetood=cifar10 \
--datasetunseen=cifar10 \
--nlabeled=2500 \
--decayfactor=$DECAYFACTOR \
--ws=15.0 \
--wd=0.001 \
--arch=WRN-28-8 \
--seed=1
```

**TinyImageNet**
```bash
DECAYFACTOR=$(bc <<< "scale=4; 5/8")
python3 $REPODIR/prosub_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=50000 \
--pretrainsteps=$((2**19)) \
--dataset=tinyimagenet-id \
--datasetood=tinyimagenet-ood \
--datasetunseen=tinyimagenet-ood \
--nlabeled=5000 \
--ws=50.0 \
--wd=0.001 \
--arch=WRN-28-4 \
--seed=1
```

**ImageNet30**
```bash
python3 $REPODIR/prosub_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=30000 \
--pretrainsteps=100000 \
--dataset=imagenet30-id \
--datasetood=imagenet30-ood \
--datasetunseen=imagenet30-ood \
--nlabeled=2600 \
--ws=20.0 \
--pi=0.66 \
--arch=ResNet18 \
--seed=0
```

**ImageNet100**
```bash
python3 $REPODIR/prosub_ossl.py \
--datadir=$DATADIR \
--traindir=$TRAINDIR \
--trainsteps=30000 \
--pretrainsteps=100000 \
--dataset=imagenet100-id \
--datasetood=imagenet100-ood \
--datasetunseen=imagenet100-ood \
--nlabeled=5000 \
--ws=40.0 \
--arch=ResNet18 \
--seed=0
```

**Note:**
* The argument `--datasetunseen` does not affect training. It can be used to make evaluations on a third dataset, unseen during training.
* The argument `-seed` only affects the selection of labeled data. It does not seed other random sources during training. Needs to be set to 0 for ImageNet runs because we only use single predefined labeled subsets for these runs.

## Results

Results are stored in summary files in the training directory. View results with tensorboard using
```bash
tensorboard --logdir $TRAINDIR
```
