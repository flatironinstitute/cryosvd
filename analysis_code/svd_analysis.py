import torch
import numpy as np
from typing import Tuple


### Proportional captured variance Analysis ###
def captured_variance(V, U, S):
    US = U @ torch.diag(S)
    return torch.norm(torch.adjoint(V) @ US) / torch.norm(torch.adjoint(U) @ US)


def proportional_captured_variance(V1, V2, S1, S2):
    return 0.5 * (captured_variance(V1, V2, S2) + captured_variance(V2, V1, S1))


def sort_matrix(
    dist_matrix: torch.Tensor, labels: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Sort the distance matrix making the first row to be the method that is the closest to most of the other methods.
    """
    dist_matrix = dist_matrix.clone()
    labels = labels.copy()

    # Sort by sum of rows
    row_sum = torch.sum(dist_matrix, dim=0)
    sort_idx = torch.argsort(row_sum, descending=True)
    dist_matrix = dist_matrix[:, sort_idx][sort_idx]
    labels = labels[sort_idx.numpy()]

    # Sort the first row
    sort_idx = torch.argsort(dist_matrix[:, 0], descending=True)
    dist_matrix = dist_matrix[:, sort_idx][sort_idx]
    labels = labels[sort_idx.numpy()]

    return dist_matrix, labels


def compute_distance_matrix(methods_data: dict) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Compute the distance matrix between the methods using the relative capture variance.

    Parameters
    ----------
    methods_data : dict
        A dictionary containing the SVD of the mean-removed volumes obtained by each method.

    Returns
    -------
    Tuple[torch.Tensor, np.ndarray]
        A tuple containing the distance matrix and the labels of rows and columns of the distance matrix.
    """

    n_subs = len(list(methods_data.keys()))
    labels = list(methods_data.keys())
    dtype = methods_data[labels[0]]["eigenvectors"].dtype

    dist_matrix = torch.ones((n_subs, n_subs), dtype=dtype)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels[i:]):
            dist_matrix[i, j + i] = proportional_captured_variance(
                methods_data[label1]["eigenvectors"],
                methods_data[label2]["eigenvectors"],
                methods_data[label1]["singular_values"],
                methods_data[label2]["singular_values"],
            )
            dist_matrix[j + i, i] = dist_matrix[i, j + i]

    labels = np.array(labels)
    if sort_matrix == True:

        dist_matrix, labels = sort_matrix(dist_matrix, labels)

    return dist_matrix, labels


### Common Embedding Analysis ###


def compute_common_embedding(methods_data: dict, sort_matrix = True) -> dict:
    """
    Compute the common embedding of the methods using the SVD of the concatenated eigenvectors.

    Parameters
    ----------

    methods_data : dict
        A dictionary containing the SVD of the mean-removed volumes obtained by each method.

    Returns
    -------
    dict
        A dictionary containing the coordinates of the volumes of each method in the common embedding, the singular values of the concatenated eigenvectors, and the common eigenvectors.
    """

    labels = list(methods_data.keys())
    n_subs = len(labels)
    shape_per_sub = methods_data[labels[0]]["eigenvectors"].T.shape
    dtype = methods_data[labels[0]]["eigenvectors"].dtype
    eigenvectors = torch.zeros(
        (n_subs * shape_per_sub[0], shape_per_sub[1]), dtype=dtype
    )

    # Concatenate and weight the eigenvectors by the singular values
    for i, label in enumerate(labels):
        eigenvectors[i * shape_per_sub[0] : (i + 1) * shape_per_sub[0], :] = (
            methods_data[label]["eigenvectors"].T
        ) * methods_data[label]["singular_values"][:, None]

    U, S, V = torch.linalg.svd(eigenvectors, full_matrices=False)

    Z_common = (U @ torch.diag(S)).reshape(n_subs, shape_per_sub[0], -1)
    embeddings = {}

    for i, label in enumerate(labels):
        Z_i = methods_data[label]["u_matrices"]
        Z_i_common = torch.einsum("ij, jk -> ik", Z_i, Z_common[i])
        embeddings[labels[i]] = Z_i_common

    results = {
        "common_embedding": embeddings,
        "singular_values": S,
        "common_eigenvectors": V,
    }

    return results
