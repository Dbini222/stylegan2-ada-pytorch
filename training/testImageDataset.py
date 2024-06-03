import unittest

import numpy as np
from dataset import ImageFolderDataset
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms


class TestImageDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "path/to/your/dataset"
        self.dataset = ImageFolderDataset(path=self.dataset_path, use_labels=True)

    def test_label_loading(self):
        # Assume that dataset.json is set up correctly
        labels, weights = self.dataset._load_raw_metadata()
        self.assertEqual(len(labels), len(weights))  # Ensure lengths match
        self.assertIsInstance(labels[0][0], np.ndarray)  # Multilabel is ndarray
        self.assertIsInstance(labels[0][1], int)  # Multiclass is int

    def test_image_loading(self):
        image, label = self.dataset[0]  # Load first sample
        self.assertEqual(image.shape, (3, 64, 64))  # Check the shape for a 64x64 RGB image

    def test_weighted_sampling(self):
        sampler = WeightedRandomSampler(self.dataset.get_raw_weight(), num_samples=100, replacement=True)
        indices = list(iter(sampler))
        # Check if indices with higher weights appear more frequently
        self.assertTrue(indices.count(0) > indices.count(1))  # Assuming weight[0] > weight[1]

if __name__ == '__main__':
    unittest.main()
