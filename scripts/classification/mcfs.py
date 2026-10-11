"""
scripts/classification/mcfs.py

Supervised Multi-Cluster Feature Selection (MCFS).
Implements the feature selection methodology cited and described in:
- Primary Paper: Lashkari et al. (2026), Section 3.6 & 3.6.3
- Method Reference [19]: Deng Cai, Chiyuan Zhang, Xiaofei He (2010),
  "Unsupervised Feature Selection for Multi-Cluster Data", IEEE TPAMI, 32(9): 1568-1581.

Algorithm Steps:
1. Construct a k-nearest-neighbor affinity graph W from training data X_train.
   In the supervised setting (Lashkari et al., Section 3.6), class labels are utilized
   to steer the selection: affinity edges between samples of different classes are zeroed out (W_ij = 0 if y_i != y_j).
2. Compute the degree matrix D and graph Laplacian L = D - W.
3. Solve the generalized eigenvalue problem L y = lambda D y for the top non-trivial eigenvectors Y.
4. Fit sparse L1-regularized regressions (Lasso) of normalized features onto each eigenvector target.
5. Compute spectral feature scores MCFS(j) = max_r |a_{r, j}|.
6. Rank features in descending order of score and select the top-k feature indices.
"""

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Lasso, Ridge


class MCFS(BaseEstimator, TransformerMixin):
    """
    Supervised Multi-Cluster Feature Selection (MCFS) Transformer.

    Parameters
    ----------
    n_selected_features : int, default=10
        Number of top features k to select.
    n_neighbors : int, default=5
        Number of nearest neighbors for affinity graph construction.
    weighting : str, default='heat_kernel'
        Affinity edge weighting scheme: 'heat_kernel' (RBF) or 'binary' (0-1).
    heat_kernel_gamma : float or None, default=None
        Gamma parameter for heat kernel: W_ij = exp(-gamma * ||x_i - x_j||^2).
        If None, automatically set to 1.0 / (2.0 * sigma^2), where sigma is the median
        k-NN pairwise distance on the training set.
    alpha : float, default=0.01
        L1 regularization parameter for Lasso regression.
    n_clusters : int or None, default=None
        Number of spectral embedding eigenvectors K to solve for.
        If None, defaults to the number of unique classes C in y (or 2 for binary).
    supervised : bool, default=True
        Whether to constrain the affinity graph by class labels (W_ij = 0 if y_i != y_j).
    random_state : int, default=42
        Random state for deterministic Lasso solver.

    Attributes
    ----------
    selected_indices_ : np.ndarray of shape (n_selected_features,)
        0-based column indices of selected features, sorted by rank.
    feature_scores_ : np.ndarray of shape (n_features,)
        Spectral importance score computed for each input feature.
    ranking_ : np.ndarray of shape (n_features,)
        Indices of all features sorted descending by score.
    eigenvalues_ : np.ndarray of shape (n_clusters,)
        Smallest non-trivial generalized eigenvalues of the graph Laplacian.
    eigenvectors_ : np.ndarray of shape (n_samples, n_clusters)
        Corresponding D-orthonormal eigenvectors.
    coefficients_ : np.ndarray of shape (n_clusters, n_features)
        Lasso regression coefficients across all eigenvector targets.
    alpha_used_ : np.ndarray of shape (n_clusters,)
        Effective regularization parameter used for each eigenvector target.
    lasso_fallback_used_ : list of str of length n_clusters
        Indicates fallback method ('none', 'retry_alpha_scaled', 'ridge') per target.
    n_features_in_ : int
        Number of input features seen during fit.
    """

    def __init__(
        self,
        n_selected_features: int = 10,
        n_neighbors: int = 5,
        weighting: str = "heat_kernel",
        heat_kernel_gamma: float | None = None,
        alpha: float = 0.01,
        n_clusters: int | None = None,
        supervised: bool = True,
        random_state: int = 42,
    ):
        self.n_selected_features = n_selected_features
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.heat_kernel_gamma = heat_kernel_gamma
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.supervised = supervised
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MCFS":
        """
        Fit the MCFS feature selector using training data and labels strictly.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : MCFS
            Fitted transformer instance.
        """
        # --- 1. Input Validation ---
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Sample count mismatch: X has {len(X_arr)} samples, y has {len(y_arr)}")

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if n_samples < 2:
            raise ValueError(f"MCFS requires at least 2 samples to construct graph, got {n_samples}")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input X contains non-finite values (NaN or Inf)")

        if self.n_selected_features < 1:
            raise ValueError(f"n_selected_features must be >= 1, got {self.n_selected_features}")
        if self.n_selected_features > n_features:
            raise ValueError(
                f"n_selected_features ({self.n_selected_features}) cannot exceed feature count ({n_features})"
            )

        unique_classes = np.unique(y_arr)
        n_classes = len(unique_classes)
        if self.supervised and n_classes < 2:
            raise ValueError(f"Supervised MCFS requires at least 2 distinct classes, got {n_classes}")

        # Determine number of clusters / eigenvectors K
        if self.n_clusters is not None:
            K = int(self.n_clusters)
        else:
            K = max(2, n_classes)

        if K >= n_samples:
            K = max(1, n_samples - 1)

        # --- 2. Affinity Graph Construction W ---
        W = np.zeros((n_samples, n_samples), dtype=np.float64)

        if self.supervised:
            # Supervised affinity graph: connect samples to nearest neighbors of the SAME class
            all_valid_dists = []
            class_neighbors_info = {}
            for c in unique_classes:
                c_mask = (y_arr == c)
                c_indices = np.where(c_mask)[0]
                n_c = len(c_indices)
                k_c = min(self.n_neighbors, n_c - 1)
                if k_c > 0:
                    nn_c = NearestNeighbors(n_neighbors=k_c + 1, metric="euclidean")
                    nn_c.fit(X_arr[c_indices])
                    dists_c, nbrs_c = nn_c.kneighbors(X_arr[c_indices])
                    class_neighbors_info[c] = (c_indices, dists_c, nbrs_c)
                    all_valid_dists.extend(dists_c[:, 1:].ravel())
                else:
                    class_neighbors_info[c] = (c_indices, None, None)

            if self.weighting == "heat_kernel":
                if self.heat_kernel_gamma is not None:
                    gamma = float(self.heat_kernel_gamma)
                else:
                    valid_dists_arr = np.array(all_valid_dists)
                    pos_dists = valid_dists_arr[valid_dists_arr > 1e-12]
                    sigma = float(np.median(pos_dists)) if len(pos_dists) > 0 else 1.0
                    gamma = 1.0 / (2.0 * sigma * sigma)

            for c, (c_indices, dists_c, nbrs_c) in class_neighbors_info.items():
                if dists_c is None:
                    continue
                for i_local in range(len(c_indices)):
                    i_global = c_indices[i_local]
                    for j_local, dist in zip(nbrs_c[i_local, 1:], dists_c[i_local, 1:]):
                        j_global = c_indices[j_local]
                        weight = float(np.exp(-gamma * (dist ** 2))) if self.weighting == "heat_kernel" else 1.0
                        W[i_global, j_global] = weight
        else:
            # Unsupervised affinity graph: global k-NN
            k_neighbors = min(self.n_neighbors, n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
            nn.fit(X_arr)
            dists, indices = nn.kneighbors(X_arr)

            if self.weighting == "heat_kernel":
                if self.heat_kernel_gamma is not None:
                    gamma = float(self.heat_kernel_gamma)
                else:
                    valid_dists = dists[:, 1:].ravel()
                    pos_dists = valid_dists[valid_dists > 1e-12]
                    sigma = float(np.median(pos_dists)) if len(pos_dists) > 0 else 1.0
                    gamma = 1.0 / (2.0 * sigma * sigma)

                for i in range(n_samples):
                    for j_idx, dist in zip(indices[i, 1:], dists[i, 1:]):
                        W[i, j_idx] = float(np.exp(-gamma * (dist ** 2)))
            elif self.weighting == "binary":
                for i in range(n_samples):
                    for j_idx in indices[i, 1:]:
                        W[i, j_idx] = 1.0
            else:
                raise ValueError(f"Unsupported weighting scheme: '{self.weighting}'. Must be 'heat_kernel' or 'binary'")

        # Symmetrize affinity matrix: W = (W + W^T) / 2
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)

        # Degree matrix D
        degrees = np.sum(W, axis=1)

        # Degree safeguard for isolated vertices:
        # Avoid division by near-zero which blows up D^{-1/2} to 10^5.
        isolated_mask = (degrees < 1e-12)
        degrees_safe = np.where(isolated_mask, 1.0, degrees)
        D_inv_sqrt = 1.0 / np.sqrt(degrees_safe)
        D_inv_sqrt[isolated_mask] = 0.0

        # Normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
        L_sym = np.eye(n_samples, dtype=np.float64) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])
        L_sym = 0.5 * (L_sym + L_sym.T)  # Force exact numerical symmetry

        # --- 3. Generalized Eigenproblem L y = lambda D y ---
        # Solved via standard symmetric eigenproblem on L_sym: L_sym z = lambda z, where y = D^{-1/2} z
        eigenvalues, z_vecs = linalg.eigh(L_sym)

        # Convert z_vecs back to generalized eigenvectors: y_r = D^{-1/2} z_r
        y_vecs = D_inv_sqrt[:, None] * z_vecs

        # Identify trivial all-ones direction v0 = D^{1/2} 1 / ||D^{1/2} 1||
        sqrt_deg = np.sqrt(degrees_safe)
        sqrt_deg[isolated_mask] = 0.0
        deg_norm = np.linalg.norm(sqrt_deg)
        v0 = sqrt_deg / deg_norm if deg_norm > 1e-12 else np.ones(n_samples) / np.sqrt(n_samples)

        selected_eigenvalues = []
        selected_y_targets = []

        if self.supervised and n_classes >= 2:
            # In supervised mode, the null space (lambda = 0) has dimension >= C.
            # Construct deterministic class-discriminant basis in the null space orthogonal to v0:
            U = np.zeros((n_samples, n_classes), dtype=np.float64)
            for c_idx, c in enumerate(unique_classes):
                c_mask = (y_arr == c) & (~isolated_mask)
                v_c = np.zeros(n_samples, dtype=np.float64)
                v_c[c_mask] = np.sqrt(degrees_safe[c_mask])
                norm_c = np.linalg.norm(v_c)
                if norm_c > 1e-12:
                    U[:, c_idx] = v_c / norm_c

            # Orthogonalize against v0 via QR decomposition on [v0, U[:, :-1]]
            M = np.column_stack([v0, U[:, :-1]])
            Q, _ = np.linalg.qr(M)
            # Columns 1 to n_classes - 1 of Q are orthogonal to v0 and span the class discriminant subspace
            for d_idx in range(1, n_classes):
                z_disc = Q[:, d_idx]
                y_disc = D_inv_sqrt * z_disc
                norm_f = np.sqrt(np.sum(y_disc ** 2 * degrees_safe))
                if norm_f > 1e-12:
                    y_disc /= norm_f
                    # Deterministic sign convention (coordinate with max absolute magnitude is positive)
                    max_abs_idx = int(np.argmax(np.abs(y_disc)))
                    if y_disc[max_abs_idx] < 0:
                        y_disc = -y_disc
                    selected_y_targets.append(y_disc)
                    selected_eigenvalues.append(0.0)
                    if len(selected_y_targets) == K:
                        break

        # If more targets needed (K > C - 1, or unsupervised), select from eigenvectors with lambda > 1e-8
        if len(selected_y_targets) < K:
            for i in range(len(eigenvalues)):
                if eigenvalues[i] > 1e-8:
                    y_cand = y_vecs[:, i].copy()
                    # Orthogonalize against v0 under D metric
                    proj = np.sum(y_cand * sqrt_deg) / deg_norm
                    y_cand = y_cand - proj * (1.0 / np.maximum(sqrt_deg, 1e-12))
                    norm_f = np.sqrt(np.sum(y_cand ** 2 * degrees_safe))
                    if norm_f > 1e-12:
                        y_cand /= norm_f
                        max_abs_idx = int(np.argmax(np.abs(y_cand)))
                        if y_cand[max_abs_idx] < 0:
                            y_cand = -y_cand
                        selected_y_targets.append(y_cand)
                        selected_eigenvalues.append(float(eigenvalues[i]))
                        if len(selected_y_targets) == K:
                            break

        # Degenerate defensive fallback if graph has fewer positive eigenvalues
        if len(selected_y_targets) < K:
            for i in range(len(eigenvalues)):
                if len(selected_y_targets) == K:
                    break
                y_cand = y_vecs[:, i].copy()
                norm_f = np.sqrt(np.sum(y_cand ** 2 * degrees_safe))
                if norm_f > 1e-12:
                    y_cand /= norm_f
                selected_y_targets.append(y_cand)
                selected_eigenvalues.append(float(eigenvalues[i]))

        Y_targets = np.column_stack(selected_y_targets[:K])
        self.eigenvalues_ = np.array(selected_eigenvalues[:K], dtype=np.float64)
        self.eigenvectors_ = Y_targets

        # --- 4. Sparse L1 Regression (Lasso) ---
        # Normalize features X for fair L1 penalty comparison across features
        x_mean = np.mean(X_arr, axis=0)
        x_std = np.std(X_arr, axis=0)
        x_std_safe = np.where(x_std < 1e-12, 1.0, x_std)
        X_scaled = (X_arr - x_mean) / x_std_safe

        coefficients = np.zeros((K, n_features), dtype=np.float64)
        alpha_used = np.zeros(K, dtype=np.float64)
        lasso_fallback_used = []

        for r in range(K):
            target_y = Y_targets[:, r]
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=True,
                max_iter=10000,
                random_state=self.random_state
            )
            lasso.fit(X_scaled, target_y)
            coefs = np.asarray(lasso.coef_, dtype=np.float64)
            current_alpha = float(self.alpha)
            current_fallback = "none"

            # Defensive fallback if alpha is too large and zeroed out all coefficients
            if np.all(np.abs(coefs) < 1e-12):
                lasso_retry = Lasso(
                    alpha=self.alpha * 0.01,
                    fit_intercept=True,
                    max_iter=10000,
                    random_state=self.random_state
                )
                lasso_retry.fit(X_scaled, target_y)
                coefs = np.asarray(lasso_retry.coef_, dtype=np.float64)
                current_alpha = float(self.alpha * 0.01)
                current_fallback = "retry_alpha_scaled"

                if np.all(np.abs(coefs) < 1e-12):
                    # Ultimate fallback to Ridge regression
                    ridge = Ridge(alpha=1e-3, fit_intercept=True, random_state=self.random_state)
                    ridge.fit(X_scaled, target_y)
                    coefs = np.asarray(ridge.coef_, dtype=np.float64)
                    current_alpha = 1e-3
                    current_fallback = "ridge"

            coefficients[r, :] = coefs
            alpha_used[r] = current_alpha
            lasso_fallback_used.append(current_fallback)

        self.coefficients_ = coefficients
        self.alpha_used_ = alpha_used
        self.lasso_fallback_used_ = lasso_fallback_used

        # --- 5. MCFS Feature Score Calculation ---
        # Cai et al. (2010), Eq. 11: MCFS(j) = max_r |a_{r, j}|
        scores = np.max(np.abs(coefficients), axis=0)
        self.feature_scores_ = scores

        # --- 6. Deterministic Feature Ranking & Selection ---
        # Use mergesort for guaranteed stable sorting on ties
        ranking = np.argsort(-scores, kind="mergesort")
        self.ranking_ = ranking
        self.selected_indices_ = ranking[: self.n_selected_features]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix by retaining only the selected top-k features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to transform.

        Returns
        -------
        X_selected : np.ndarray of shape (n_samples, n_selected_features)
            Transformed feature matrix containing top-k features.
        """
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("MCFS instance is not fitted yet. Call 'fit' before 'transform'.")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: fitted on {self.n_features_in_} features, got {X_arr.shape[1]}"
            )

        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training class labels.

        Returns
        -------
        X_selected : np.ndarray of shape (n_samples, n_selected_features)
            Transformed feature matrix.
        """
        return self.fit(X, y).transform(X)
