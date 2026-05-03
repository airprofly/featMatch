def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """
    import numpy as np

    # Compute ||a||^2 for all vectors in features1: shape (n, 1)
    a_sq = np.sum(features1 ** 2, axis=1, keepdims=True)

    # Compute ||b||^2 for all vectors in features2: shape (1, m)
    b_sq = np.sum(features2 ** 2, axis=1, keepdims=True).T

    # Compute a·b for all pairs: shape (n, m)
    ab = np.dot(features1, features2.T)

    # Compute ||a - b||^2 = ||a||^2 - 2*a·b + ||b||^2
    dists_sq = a_sq + b_sq - 2 * ab

    # Ensure non-negative values (handle numerical errors)
    dists_sq = np.maximum(dists_sq, 0)

    # Compute the Euclidean distances
    dists = np.sqrt(dists_sq)

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    import numpy as np
    
    # Compute pairwise distances between all features: shape (n, m)
    dists = compute_feature_distances(features1, features2)
    
    # For each feature in features1, find the two nearest features in features2
    # argsort returns indices in ascending order of distance
    sorted_indices = np.argsort(dists, axis=1)
    
    # Get the indices of the nearest and second-nearest neighbors
    nearest_idx = sorted_indices[:, 0]
    second_nearest_idx = sorted_indices[:, 1]
    
    # Get the corresponding distances
    nearest_dists = dists[np.arange(dists.shape[0]), nearest_idx]
    second_nearest_dists = dists[np.arange(dists.shape[0]), second_nearest_idx]
    
    # Apply ratio test: keep matches where nearest_dist / second_nearest_dist < threshold
    ratio_threshold = 0.7
    ratio = nearest_dists / (second_nearest_dists + 1e-8)
    
    # Find matches that pass the ratio test
    valid_matches = ratio < ratio_threshold
    match_indices_1 = np.where(valid_matches)[0]
    match_indices_2 = nearest_idx[valid_matches]
    
    # Build matches array: shape (k, 2)
    matches = np.column_stack((match_indices_1, match_indices_2))
    
    # Compute confidences: 1 - ratio gives higher confidence for better matches
    confidences = 1.0 - ratio[valid_matches]
    
    return matches, confidences
