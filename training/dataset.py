# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os
import shutil
import zipfile
from pathlib import Path

import dnnlib
import numpy as np
import PIL.Image
import torch

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._weight = None
        self.num_classes = 3 # simple, normal, complex

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def get_raw_weight(self):
        if self._weight is None:
            _, self._weight = self._load_raw_metadata() if self._use_labels else None
            if self._weight is None:
                self._weight = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._weight, np.ndarray)
            assert self._weight.shape[0] == self._raw_shape[0]
            assert self._weight.dtype in [np.float32]
        return self._weight

    def _get_raw_labels(self):
        if self._raw_labels is None:
            # assume load_raw_metadata is in correct form
            self._raw_labels, _ = self._load_raw_metadata() if self._use_labels else (None, None)
            if self._raw_labels is None:
                # Initialize with a default value if no labels are found
                self._raw_labels = np.zeros((self._raw_shape[0], 0), dtype=np.float32)
            else:
                # Ensure the data structure is correct
                assert isinstance(self._raw_labels, list)
                # Convert list of lists into a structured numpy array if necessary
                dtype = [('multilabel_vector', np.float32, (len(self._raw_labels[0][0]),)), ('multiclass_label', np.int32)]
                self._raw_labels = np.array(self._raw_labels, dtype=dtype)
        return self._raw_labels


    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError
    
    def _load_raw_weight(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        label_vector, class_label = self.get_label(idx)
        onehot_multiclass = np.zeros((self.num_classes,))
        onehot_multiclass[class_label] = 1
        return image.copy(), (label_vector.copy(), onehot_multiclass)

    def get_label(self, idx):
        label_entry = self._get_raw_labels()[self._raw_idx[idx]]
        multilabel_vector, multiclass_label = label_entry
        return multilabel_vector, multiclass_label
    
    def get_weight(self, idx):
        weight = self.get_raw_weight()[self._raw_idx[idx]]
        return weight.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_metadata(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            labels = data['labels']
            weights = data['weights']
        if labels is None:
            return None
        if weights is None:
            return None
        parsed_labels = []
        parsed_weights = []
        #below is to handle multi label and multi classification
        for img_path, label in labels.items():
            multilabel_vector, multiclass_label = label[0], label[1]
            # Store multilabel as numpy array and multiclass as integer
            parsed_labels.append([np.array(multilabel_vector, dtype=np.float32), multiclass_label])
            parsed_weights.append(weights[img_path])

        parsed_weights = np.array(parsed_weights, dtype=np.float32)
        return parsed_labels, parsed_weights
        # labels = dict(labels)
        # weights = dict(weights)
        # labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        # labels = np.array(labels)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        # weights = [weights[fname.replace('\\', '/')] for fname in self._image_fnames]
        # weights = np.array(weights)
        # weights = weights.astype({1: np.float32}[weights.ndim])
        # return labels, weights

#----------------------------------------------------------------------------
#Testing changes
    
# Mock the methods from your dataset class that depend on actual file IO
def mock_open_file(self, fname):
    if 'json' in fname:
        return open('dataset.json', 'r')  # Assume dataset.json is in the current directory for simplicity
    raise FileNotFoundError

def mock_load_raw_image(self, idx):
    # Return a dummy image array (3-channel RGB, 64x64)
    return np.zeros((3, 64, 64), dtype=np.uint8)

# Replace file I/O methods with mocks
ImageFolderDataset._open_file = mock_open_file
ImageFolderDataset._load_raw_image = mock_load_raw_image

# Assuming ImageFolderDataset is already defined and imported
# if __name__ == "__main__":
#     data = {
#         "labels": {
#             "images/img1.png": [[1, 0, 0, 1], 0],
#             "images/img2.png": [[0, 1, 1, 0], 1],
#             "images/img3.png": [[1, 1, 0, 1], 2],
#             "images/img4.png": [[0, 0, 1, 1], 0],
#             "images/img5.png": [[1, 0, 1, 0], 1]
#         },
#         "weights": {
#             "images/img1.png": 0.5,
#             "images/img2.png": 0.7,
#             "images/img3.png": 0.9,
#             "images/img4.png": 0.4,
#             "images/img5.png": 0.6
#         }
#     }

#     with open('dataset.json', 'w') as f:
#         json.dump(data, f, indent=4)

#     # Setup test environment
#     dataset_path = Path('test_dataset')
#     dataset_path.mkdir(exist_ok=True)
#     (dataset_path / 'img1.png').touch()
#     (dataset_path / 'img2.png').touch()
#     (dataset_path / 'img3.png').touch()
#     (dataset_path / 'img4.png').touch()

#     # Initialize the dataset
#     dataset = ImageFolderDataset(path=str(dataset_path), use_labels=True)

#     # Test the label loading and parsing
#     for idx in range(len(dataset)):
#         image, (multilabel_vector, multiclass_label) = dataset[idx]
#         print(f"Image {idx + 1}: Multilabel: {multilabel_vector}, Multiclass: {multiclass_label}")

#     # Clean up
#     shutil.rmtree(dataset_path)