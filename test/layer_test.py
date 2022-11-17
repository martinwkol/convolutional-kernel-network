import unittest
import numpy as np

import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from src.kernel import dot_product_kernel
from src.layer import layer

class LayerTest(unittest.TestCase):
    def test_extract_patches_without_zero_padding(self):
        l = layer(
            input_size=(3, 3), num_channels=2, patch_size=(3, 3), 
            pooling_factor=0.5, dp_kernel=dot_product_kernel, filter=[],
            use_zero_padding=False
        )
        input = np.array([
            [11, 12], [13, 14], [15, 16], 
            [21, 22], [23, 24], [25, 26], 
            [31, 32], [33, 34], [35, 36]
        ]).transpose()
        patched = l.extract_patches(input)

        self.assertEqual(patched.shape, (2 * l.patch_size[0] * l.patch_size[1], 1))
        expectedPatched = np.array([
            [
                11, 12, 13, 14, 15, 16,
                21, 22, 23, 24, 25, 26,
                31, 32, 33, 34, 35, 36,
            ]
        ]).transpose()
        self.assertTrue((patched == expectedPatched).all())

    def test_extract_patches_zero_padding(self):
        l = layer(
            input_size=(3, 3), num_channels=2, patch_size=(3, 3), 
            pooling_factor=0.5, dp_kernel=dot_product_kernel, filter=[],
            use_zero_padding=True
        )
        input = np.array([
            [11, 12], [13, 14], [15, 16], 
            [21, 22], [23, 24], [25, 26], 
            [31, 32], [33, 34], [35, 36]
        ]).transpose()
        patched = l.extract_patches(input)

        self.assertEqual(patched.shape, (2 * l.patch_size[0] * l.patch_size[1], l.input_size[0] * l.input_size[1]))
        expectedPatched = np.array([
            [
                0, 0,   0, 0,   0, 0,
                0, 0,   11, 12, 13, 14,
                0, 0,   21, 22, 23, 24,
            ], [
                0, 0,   0, 0,   0, 0,
                11, 12, 13, 14, 15, 16,
                21, 22, 23, 24, 25, 26,
            ], [
                0, 0,   0, 0,   0, 0,
                13, 14, 15, 16, 0, 0,
                23, 24, 25, 26, 0, 0,
            ], [
                0, 0,   11, 12, 13, 14,
                0, 0,   21, 22, 23, 24,
                0, 0,   31, 32, 33, 34,
            ], [
                11, 12, 13, 14, 15, 16,
                21, 22, 23, 24, 25, 26,
                31, 32, 33, 34, 35, 36,
            ], [
                13, 14, 15, 16, 0, 0,
                23, 24, 25, 26, 0, 0,
                33, 34, 35, 36, 0, 0,
            ], [
                0, 0,   21, 22, 23, 24,
                0, 0,   31, 32, 33, 34,
                0, 0,   0, 0,   0, 0,
            ], [
                21, 22, 23, 24, 25, 26,
                31, 32, 33, 34, 35, 36,
                0, 0,   0, 0,   0, 0,
            ], [
                23, 24, 25, 26, 0, 0,
                33, 34, 35, 36, 0, 0,
                0, 0,   0, 0,   0, 0,
            ]
        ]).transpose()

        self.assertTrue((patched == expectedPatched).all())

if __name__ == '__main__':
    unittest.main()