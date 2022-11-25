import unittest
import numpy as np

import sys
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from src.internal_pooling_layer import IntPoolingLayer

class LayerTest(unittest.TestCase):
    @staticmethod
    def random_filter_matrix(shape):
        filter_mx = np.random.rand(shape[0], shape[1])
        norms = np.linalg.norm(filter_mx, axis = 0)
        return filter_mx * (1 / norms)

    def setUp(self):
        self.filter_mx_1x1x1 = LayerTest.random_filter_matrix((1*1*1, 2))
        self.filter_mx_3x3x1 = LayerTest.random_filter_matrix((3*3*1, 2))
        self.filter_mx_3x3x2 = LayerTest.random_filter_matrix((3*3*2, 2))


    def test_avg_pooling(self):
        pl = IntPoolingLayer(
            input_size=(5, 5), pooling_size=(2, 2),
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
        pl = IntPoolingLayer(
            input_size=(5, 5), pooling_size=(2, 2),
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