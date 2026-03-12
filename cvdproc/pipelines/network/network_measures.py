import numpy as np
import pandas as pd
import bct


def load_connectivity_matrix(
    csv_file,
    header=None,
    index_col=None,
    force_square=True,
    force_symmetric=True,
    zero_diagonal=True,
    remove_negative=True,
    dtype=np.float64,
):
    """
    Load a connectivity matrix from a CSV file and clean it for graph analysis.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    header : int or None, optional
        Row number to use as the column names. Use None if the CSV has no header.
    index_col : int or None, optional
        Column to use as row labels. Use None if the CSV has no index column.
    force_square : bool, optional
        If True, raise an error when the matrix is not square.
    force_symmetric : bool, optional
        If True, replace the matrix with (A + A.T) / 2.
    zero_diagonal : bool, optional
        If True, set diagonal values to zero.
    remove_negative : bool, optional
        If True, set negative values to zero.
    dtype : numpy dtype, optional
        Numeric dtype for the output matrix.

    Returns
    -------
    mat : numpy.ndarray
        Cleaned connectivity matrix with shape (N, N).
    """
    df = pd.read_csv(csv_file, header=header, index_col=index_col)

    try:
        mat = df.to_numpy(dtype=dtype)
    except ValueError as e:
        raise ValueError(f"Failed to convert CSV contents to numeric matrix: {e}")

    if mat.ndim != 2:
        raise ValueError(f"Matrix must be 2D, but got ndim={mat.ndim}")

    n_rows, n_cols = mat.shape
    if force_square and n_rows != n_cols:
        raise ValueError(
            f"Connectivity matrix must be square, but got shape {mat.shape}"
        )

    if not np.isfinite(mat).all():
        raise ValueError("Matrix contains NaN or infinite values.")

    if remove_negative:
        mat[mat < 0] = 0.0

    if zero_diagonal:
        np.fill_diagonal(mat, 0.0)

    if force_symmetric:
        mat = (mat + mat.T) / 2.0

    return mat


def describe_connectivity_matrix(mat):
    """
    Print basic information about a connectivity matrix.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    print(f"Shape: {A.shape}")
    print(f"Dtype: {A.dtype}")
    print(f"Min: {A.min()}")
    print(f"Max: {A.max()}")
    print(f"Mean: {A.mean()}")
    print(f"Nonzero edges: {np.count_nonzero(A)}")
    print(f"Symmetric: {np.allclose(A, A.T, atol=1e-8)}")
    print(f"Diagonal all zero: {np.allclose(np.diag(A), 0)}")


def compute_global_efficiency(mat):
    """
    Compute global efficiency for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    global_eff : float
        Global efficiency of the network.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    global_eff = float(bct.efficiency_wei(A, local=False))
    return global_eff


def compute_local_efficiency(mat, return_nodal=False):
    """
    Compute local efficiency for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.
    return_nodal : bool, optional
        If True, return nodal local efficiency values.
        If False, return the mean local efficiency across nodes.

    Returns
    -------
    mean_local_eff : float
        Mean local efficiency across nodes, if return_nodal=False.
    nodal_local_eff : numpy.ndarray
        Nodal local efficiency values, if return_nodal=True.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    nodal_local_eff = np.asarray(bct.efficiency_wei(A, local=True), dtype=np.float64)

    if return_nodal:
        return nodal_local_eff

    mean_local_eff = float(np.mean(nodal_local_eff))
    return mean_local_eff


def compute_nodal_strength(mat):
    """
    Compute nodal strength for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    nodal_strength : numpy.ndarray
        Strength of each node.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    nodal_strength = np.sum(A, axis=1)
    return nodal_strength


def compute_mean_strength(mat):
    """
    Compute mean nodal strength for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    mean_strength : float
        Mean strength across nodes.
    """
    nodal_strength = compute_nodal_strength(mat)
    mean_strength = float(np.mean(nodal_strength))
    return mean_strength


def compute_nodal_clustering_coefficient(mat):
    """
    Compute nodal clustering coefficient for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    nodal_clustering : numpy.ndarray
        Clustering coefficient of each node.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    nodal_clustering = np.asarray(bct.clustering_coef_wu(A), dtype=np.float64)
    return nodal_clustering


def compute_mean_clustering_coefficient(mat):
    """
    Compute mean clustering coefficient for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    mean_clustering : float
        Mean clustering coefficient across nodes.
    """
    nodal_clustering = compute_nodal_clustering_coefficient(mat)
    mean_clustering = float(np.mean(nodal_clustering))
    return mean_clustering


def compute_length_matrix(mat):
    """
    Convert a weighted adjacency matrix to a length matrix.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    length_mat : numpy.ndarray
        Length matrix where larger weights correspond to shorter lengths.
    """
    A = np.asarray(mat, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Input matrix must be square, but got shape {A.shape}")

    length_mat = np.asarray(bct.weight_conversion(A, "lengths"), dtype=np.float64)
    return length_mat


def compute_distance_matrix(mat):
    """
    Compute the weighted shortest-path distance matrix.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    dist_mat : numpy.ndarray
        Shortest-path distance matrix.
    """
    length_mat = compute_length_matrix(mat)
    dist_mat, _ = bct.distance_wei(length_mat)
    dist_mat = np.asarray(dist_mat, dtype=np.float64)
    return dist_mat


def compute_characteristic_path_length(mat, include_diagonal=False, include_infinite=False):
    """
    Compute characteristic path length for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.
    include_diagonal : bool, optional
        If True, include diagonal elements in averaging.
        Usually False.
    include_infinite : bool, optional
        If True, include infinite distances in averaging.
        Usually False.

    Returns
    -------
    char_path_length : float
        Characteristic path length.
    """
    dist_mat = compute_distance_matrix(mat)

    mask = np.ones(dist_mat.shape, dtype=bool)

    if not include_diagonal:
        np.fill_diagonal(mask, False)

    if not include_infinite:
        mask &= np.isfinite(dist_mat)

    valid_distances = dist_mat[mask]

    if valid_distances.size == 0:
        raise ValueError("No valid distances available to compute characteristic path length.")

    char_path_length = float(np.mean(valid_distances))
    return char_path_length


def compute_nodal_betweenness_centrality(mat):
    """
    Compute nodal betweenness centrality for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    nodal_betweenness : numpy.ndarray
        Betweenness centrality of each node.
    """
    length_mat = compute_length_matrix(mat)
    nodal_betweenness = np.asarray(bct.betweenness_wei(length_mat), dtype=np.float64)
    return nodal_betweenness


def compute_mean_betweenness_centrality(mat):
    """
    Compute mean nodal betweenness centrality for a weighted undirected network.

    Parameters
    ----------
    mat : array-like
        Weighted adjacency matrix.

    Returns
    -------
    mean_betweenness : float
        Mean betweenness centrality across nodes.
    """
    nodal_betweenness = compute_nodal_betweenness_centrality(mat)
    mean_betweenness = float(np.mean(nodal_betweenness))
    return mean_betweenness


if __name__ == "__main__":
    csv_file = "/mnt/f/BIDS/WCH_SVD_3T_BIDS/derivatives/dwi_pipeline/sub-SSI0008/ses-baseline/connectome/sub-SSI0008_ses-baseline_acq-DTIb1000_space-preprocdwi_seg-aparc_connectivity.csv"

    mat = load_connectivity_matrix(
        csv_file,
        header=None,
        index_col=None,
        force_square=True,
        force_symmetric=True,
        zero_diagonal=True,
        remove_negative=True,
    )

    describe_connectivity_matrix(mat)

    global_eff = compute_global_efficiency(mat)
    mean_local_eff = compute_local_efficiency(mat, return_nodal=False)
    nodal_local_eff = compute_local_efficiency(mat, return_nodal=True)

    mean_strength = compute_mean_strength(mat)
    nodal_strength = compute_nodal_strength(mat)

    mean_clustering = compute_mean_clustering_coefficient(mat)
    nodal_clustering = compute_nodal_clustering_coefficient(mat)

    char_path_length = compute_characteristic_path_length(mat)
    mean_betweenness = compute_mean_betweenness_centrality(mat)
    nodal_betweenness = compute_nodal_betweenness_centrality(mat)

    print("Global efficiency:", global_eff)
    print("Mean local efficiency:", mean_local_eff)
    print("Nodal local efficiency:", nodal_local_eff)

    print("Mean strength:", mean_strength)
    print("Nodal strength:", nodal_strength)

    print("Mean clustering coefficient:", mean_clustering)
    print("Nodal clustering coefficient:", nodal_clustering)

    print("Characteristic path length:", char_path_length)
    print("Mean betweenness centrality:", mean_betweenness)
    print("Nodal betweenness centrality:", nodal_betweenness)