import numpy as np
from sklearn.neighbors import NearestNeighbors

def similarityRank(matrix):
    # TODO: Would be interesting to implement this using Travelling Salesman 
    # approximation with Hamming distances and see if it improves results even more
    """
    Returns array of matrix indices sorted by similarity.

    Adopted from Flagel et al. 2018 https://doi.org/10.1101/336073
    """
    mat = matrix.transpose()
    nbrs = NearestNeighbors(n_neighbors=mat.shape[0], metric="manhattan").fit(mat)
    distances, ixs = nbrs.kneighbors(mat)
    smallest = np.argmin(distances.sum(axis=1))
    rank = ixs[smallest]
    return rank

def getSimilarityRanks(matrices):
    """
    Returns array of similarity rank arrays for each matrix in input array.
    """
    ranks = np.empty(matrices.shape[0], dtype=object)#TODO: Need to be dtype=object, should it be?
    for i, mat in enumerate(matrices):
        ranks[i] = similarityRank(mat)
    return ranks

def reorderMatrices(ixs, matrices):
    """
    Reorder columns of matrices with array of indexes  
    """
    if not isinstance(matrices, list):
        matrices = [matrices]
    for i in range(matrices[0].shape[0]):
        for j in matrices:
            j[i] = j[i][:,ixs[i]]

def sortMatrices(matrices):
    """
    Sort each matrix in the input arrays
    """
    if not isinstance(matrices, list): 
        matrices = [matrices]
    for i in range(matrices[0].shape[0]):
        rank = similarityRank(matrices[0][i])
        for j in matrices:
            j[i] = j[i][:, rank]

def padMatrices(matrices):
    """
    Pad array of matrices with zeros
    """
    maxLen = max([i.shape[0] for i in matrices]) 
    paddedMatrixArray = np.zeros((matrices.shape[0], maxLen, 
            matrices[0].shape[1]), dtype=np.int8)
    for i in range(matrices.shape[0]):
        for j in range(matrices[i].shape[0]):
            paddedMatrixArray[i][j] = matrices[i][j]
    return paddedMatrixArray

def padArrays(arrays):
    """
    Pad array of arrays with zeros
    """
    maxLen = max([i.shape[0] for i in arrays]) 
    # paddedArrays = np.zeros((arrays.shape[0], maxLen), dtype=np.int8)
    paddedArrays = np.full((arrays.shape[0], maxLen), -1)
    for i in range(arrays.shape[0]):
        paddedArrays[i][0:arrays[i].shape[0]] = arrays[i]
    return paddedArrays


