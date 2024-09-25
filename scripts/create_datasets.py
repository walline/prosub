import random
import os
import shutil
import tarfile
import tempfile
import io
import zipfile
from urllib import request

import numpy as np
import scipy.io
import tensorflow as tf
from absl import app, flags
from tqdm import trange, tqdm
from PIL import Image

FLAGS = flags.FLAGS

URLS = {
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz',
    'tinyimagenet': 'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
}


def _encode_png(images):
    raw = []
    for x in trange(images.shape[0], desc="PNG Encoding", leave=False):
        raw.append(tf.image.encode_png(images[x]))
    return raw

def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_cifar100():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar100'], f.name)
        tar = tarfile.open(fileobj=f)
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/train.mat'))
        train_set = {'images': data_dict['data'],
                     'labels': data_dict['fine_labels'].flatten()}
        data_dict = scipy.io.loadmat(tar.extractfile('cifar-100-matlab/test.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['fine_labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _load_tinyimagenet():

    def _get_tin_classes():
        classes = {}
        fn = os.path.join(FLAGS.repodir, "data-files", "tin-classes.txt")
        with open(fn, "r") as f:
            for line in f:
                columns = line.strip().split('\t')
                classes[columns[0]] = int(columns[1])
        return classes
            
    tin_classes = _get_tin_classes()

    response = request.urlopen(URLS["tinyimagenet"])
    zip_file_bytes = response.read()
    zip_file_buffer = io.BytesIO(zip_file_bytes)

    with zipfile.ZipFile(zip_file_buffer, 'r') as zip_ref:

        images, labels = [], []

        all_files = zip_ref.namelist()
        class_names = list(tin_classes.keys())

        for i in trange(len(class_names), desc="Encoding TIN training set"):

            class_ = class_names[i]

            train_files = [x for x in all_files
                           if class_ in x
                           and "train" in x
                           and x.endswith(".JPEG")]

            for img_file in train_files:
                with zip_ref.open(img_file) as f:
                    png_data = _jpg_to_png(f.read())
                    images.append(tf.convert_to_tensor(png_data, tf.string))

            label = tin_classes[class_]
            labels.extend([label]*len(train_files))

        # shuffle training data
        ziplist = list(zip(images, labels))
        random.shuffle(ziplist)
        images, labels = zip(*ziplist)

        # read test labels
        val_dict = {}
        with zip_ref.open("tiny-imagenet-200/val/val_annotations.txt") as f:
            for line in f:
                line = line.decode('utf-8')
                columns = line.strip().split('\t')
                val_dict[columns[0]] = columns[1]

        val_files = [x for x in all_files
                     if "/val/images/" in x
                     and x.endswith(".JPEG")]

        val_images, val_labels = [], []

        for img_file in tqdm(val_files, desc="Encoding TIN test set"):
            with zip_ref.open(img_file) as f:
                png_data = _jpg_to_png(f.read())
                val_images.append(tf.convert_to_tensor(png_data, tf.string))
                class_ = val_dict[img_file.split("/")[-1]]
                label = tin_classes[class_]
                val_labels.append(label)

    def _filter_data(image_list, label_list, min_, max_):
        ziplist = [(x, y) for x, y in zip(image_list, label_list)
                   if min_ <= y < max_]
        images_filtered, labels_filtered = zip(*ziplist)        
        return dict(images=images_filtered, labels=labels_filtered)

    return {"id-train": _filter_data(images, labels, 0, 100),
            "id-test": _filter_data(val_images, val_labels, 0, 100),
            "ood-train": _filter_data(images, labels, 100, 200),
            "ood-test": _filter_data(val_images, val_labels, 100, 200),
            "id70-train": _filter_data(images, labels, 0, 70),
            "id70-test": _filter_data(val_images, val_labels, 0, 70),
            "ood70-train": _filter_data(images, labels, 70, 140),
            "ood70-test": _filter_data(val_images, val_labels, 70, 140),
            "unseen60-test": _filter_data(val_images, val_labels, 140, 200),
            }

def _load_imagenet(n_classes):

    nclasses2nlabels = {30: 2600,
                        100: 5000,}

    n_labels = nclasses2nlabels[n_classes]
    
    fns = [f"imagenet{n_classes}-id-test.txt",
           f"imagenet{n_classes}-ood-test.txt",
           f"imagenet{n_classes}-id-train.txt",
           f"imagenet{n_classes}-ood-train.txt"]
    for fn in fns:
        shutil.copy(os.path.join(FLAGS.repodir, "data-files", fn),
                    os.path.join(FLAGS.datadir, fn))

    labeled_dir = os.path.join(FLAGS.datadir, "SSL2")
    os.makedirs(labeled_dir, exist_ok=True)

    labels_fn = f"imagenet{n_classes}-id.0@{n_labels}-label.txt"
    shutil.copy(os.path.join(FLAGS.repodir, "data-files", labels_fn),
                os.path.join(FLAGS.datadir, "SSL2", labels_fn))

    fns.append(f"SSL2/{labels_fn}")

    for fn in fns:
        if not _check_files(os.path.join(FLAGS.datadir, fn)):
            raise FileNotFoundError(f"Did not find all files listed in {fn}")

    print(f"All files associated with ImageNet{n_classes} are successfully located")
    
    return {}

def _check_files(paths_file):
    status = True
    with open(paths_file, "r") as f:
        for line in f:
            img_path = line.split(" ")[0]
            full_path = os.path.join(FLAGS.datadir, img_path)
            if not os.path.isfile(full_path):
                status = False
    return status

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _jpg_to_png(jpg_bytes):
    img = Image.open(io.BytesIO(jpg_bytes))
    img = img.convert("RGB")
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_data = png_buffer.getvalue()
    return png_data

def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(FLAGS.datadir, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x].numpy()),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(FLAGS.datadir, '%s-%s.tfrecord' % (name, subset))
        if not tf.io.gfile.exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.io.gfile.makedirs(os.path.join(FLAGS.datadir, folder))
    for filename, contents in files.items():
        with tf.io.gfile.GFile(os.path.join(FLAGS.datadir, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.io.gfile.exists(os.path.join(FLAGS.datadir, name, folder))


CONFIGS = dict(
    cifar10=dict(loader=_load_cifar10, checksums=dict(train=None, test=None)),
    cifar100=dict(loader=_load_cifar100, checksums=dict(train=None, test=None)),
    tinyimagenet=dict(loader=_load_tinyimagenet,
                      checksums={"id-train": None, "id-test": None,
                                 "ood-train": None, "ood-test": None,
                                 "id70-train": None, "id70-test": None,
                                 "ood70-train": None, "ood70-test": None,
                                 "unseen60-test": None}),
    imagenet30=dict(loader=lambda: _load_imagenet(30),
                    is_installed=lambda: False,
                    saver=lambda *args: None),
    imagenet100=dict(loader=lambda: _load_imagenet(100),
                     is_installed=lambda: False,
                     saver=lambda *args: None),
)


def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.io.gfile.makedirs(FLAGS.datadir)
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(FLAGS.datadir, '%s-%s.txt' % (name, sub_name))
                with tf.io.gfile.GFile(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(FLAGS.datadir, file_and_data.filename)
                    with tf.io.gfile.GFile(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    flags.DEFINE_string("datadir", "./data", "Directory for data")
    flags.DEFINE_string("repodir", ".", "Directory for ProSub repo")
    app.run(main)
