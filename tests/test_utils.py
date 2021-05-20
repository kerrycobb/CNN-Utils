import pytest
from cnn_utils import padMatrices, padArrays, sortMatrices, getSimilarityRanks, similarityRank
import numpy as np

def test_padMatrices():
    mat1 = np.empty(2, dtype=object)
    mat1[0] = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]).transpose()
    mat1[1] = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]]).transpose()
    padded = padMatrices(mat1)
    mat2 = np.array([
            [[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]], 
            [[1, 1, 1, 1],[1, 1, 1, 1],[0, 0, 0, 0],[0, 0, 0, 0]]])
    assert np.array_equal(padded, mat2)

def test_padArrays():
    unpadded = np.array([
        np.array([1, 1, 1, 1, 1]),
        np.array([1, 1, 1]),    
        np.array([1])], dtype=object)
    padded = padArrays(unpadded, value=-1)
    testArr = np.array([[1, 1, 1, 1, 1], [1, 1, 1, -1, -1], [1, -1, -1, -1, -1]])
    assert(np.array_equal(padded, testArr))

def test_sortMatrices():
    mat1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]).transpose()
    mat1 = np.array([mat1])
    sortMatrices(mat1)
    mat2 = np.array([[
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
    ]])
    assert np.array_equal(mat1, mat2)

def test_similarityRank():
    mat1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]).transpose()
    mat1 = np.array([mat1])
    assert np.array_equal(getSimilarityRanks(mat1)[0], np.array([2, 1, 3, 0, 4]))
    assert np.array_equal(similarityRank(mat1[0]), np.array([2, 1, 3, 0, 4]))