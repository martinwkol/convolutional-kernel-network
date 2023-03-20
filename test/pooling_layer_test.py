import unittest
import numpy as np

import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
sys.path.append("src/")

from src.pooling_layer import PoolingLayer

class LayerTest(unittest.TestCase):
    def test_avg_pooling(self):
        pl = PoolingLayer(
            input_size=(5, 5), in_channels=1, pooling_size=(2, 2),
        )

        U = np.array([[
            1., 2., 3., 4., 5.,
            2., 3., 4., 5., 6.,
            3., 4., 5., 6., 7.,
            4., 5., 6., 7., 8.,
            5., 6., 7., 8., 9.
        ]])

        pooled = pl._avg_pooling(U)

        expected_pooled = np.array([[
            2., 4.,
            4., 6.
        ]])

        self.assertTrue((pooled == expected_pooled).all())


    def test_avg_pooling_t(self):
        pl = PoolingLayer(
            input_size=(5, 5), in_channels=1, pooling_size=(2, 2),
        )

        U = np.array([[
            2., 4.,
            4., 6.
        ]])

        upscaled = pl._avg_pooling_t(U)

        expected_upscaled = np.array([[
            0.5, 0.5,   1., 1.,     0.,
            0.5, 0.5,   1., 1.,     0.,
            1., 1.,     1.5, 1.5,   0.,
            1., 1.,     1.5, 1.5,   0.,
            0., 0.,     0., 0.,     0.
        ]])

        self.assertTrue((upscaled == expected_upscaled).all())


if __name__ == '__main__':
    unittest.main()