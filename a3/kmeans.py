import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
from typing import Optional, Union


class KMeans:
    '''
    Sample Class for KMeans Clustering
    DO NOT MODIFY any part of this code unless you are requested to do so.
    '''

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 30,
        tol: float = 1e-4,
        init: str = 'random',
        n_init: int = 300,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.best_centroids: Optional[np.ndarray] = None
        self.best_objective: float = np.inf
        self.verbose = verbose

        if self.seed is not None:
            np.random.seed(self.seed)

    def fit(self, X: np.ndarray, algorithm: str = 'lloyd') -> None:
        for init in tqdm(range(self.n_init)):
            self.centroids = self._initialize_centroids(X)

            if init == 0:
                self.best_centroids = self.centroids.copy()
                self.best_objective = np.inf

            if algorithm == 'lloyd':
                self._lloyd(X)
            elif algorithm == 'elkan':
                self._elkan(X)
            elif algorithm == 'hamerly':
                self._hamerly(X)
            else:
                raise ValueError("Unknown algorithm type")

        self.centroids = self.best_centroids

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.init == 'kmeans++':
            return self._kmeans_pp(X)
        return self._random_init(X)

    def _random_init(self, X: np.ndarray) -> np.ndarray:
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        centroids = [X[np.random.choice(X.shape[0])]]

        for _ in range(self.n_clusters - 1):
            # Compute the distance of each point to the nearest centroid
            dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids])
                                for x in X])

            # Choose the next centroid based on the probability proportional to the squared distance
            probs = dist_sq / dist_sq.sum()
            centroids.append(X[np.random.choice(X.shape[0], p=probs)])

        return np.array(centroids)

    def _lloyd(self, X: np.ndarray) -> None:
        for iteration in range(self.max_iter):
            # Assignment step
            clusters = self._assign_clusters(X)

            # Update centroids
            new_centroids = np.array([
                X[clusters == i].mean(axis=0) if np.any(clusters == i)
                else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            # Compute the new objective value
            new_objective = self._compute_objective(X, clusters, new_centroids)

            # Update the best centroids if the objective improves
            if new_objective < self.best_objective:
                self.best_centroids = new_centroids.copy()
                self.best_objective = new_objective

            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) <= self.tol:
                break

            # Always update centroids
            self.centroids = new_centroids
            if iteration % 10 == 0 and self.verbose:
                print(f"Iteration {iteration}: objective = {new_objective}")

    def _elkan(self, X: np.ndarray) -> None:
        centroid_dists = np.zeros((self.n_clusters, self.n_clusters))
        distances = cdist(X, self.centroids, 'euclidean')
        clusters = np.argmin(distances, axis=1)
        upper_bounds = np.min(distances, axis=1)
        lower_bounds = distances
        p = np.zeros(self.n_clusters)

        for iteration in range(self.max_iter):
            # Step 1: Compute distances between centroids
            centroid_dists = cdist(self.centroids, self.centroids, 'euclidean')
            np.fill_diagonal(centroid_dists, np.inf)
            s = np.min(centroid_dists, axis=1) / 2

            # Step 2: Update bounds
            upper_bounds += p[clusters]
            lower_bounds = np.maximum(lower_bounds - p, 0)

            # Step 3: Assignment step
            mask1 = upper_bounds > s[clusters]
            X_masked = X[mask1]
            distances = np.linalg.norm(
                X_masked[:, np.newaxis] - self.centroids, axis=2)
            # The commented code below is the algorithm in the original paper but it is hard to vectorize
            # distances = np.full((X_masked.shape[0], self.n_clusters), np.inf)
            # for i in range(self.n_clusters):
            #     mask2 = upper_bounds[mask1] > np.maximum(
            #         lower_bounds[mask1][:, i], centroid_dists[clusters[mask1], i])
            #     distances[mask2, i] = np.linalg.norm(
            #         X_masked[mask2] - self.centroids[i], axis=1)
            upper_bounds[mask1] = np.min(distances, axis=1)
            lower_bounds[mask1] = distances
            clusters[mask1] = np.argmin(distances, axis=1)

            # Step 4: Update centroids
            new_centroids = np.array([
                X[clusters == i].mean(axis=0) if np.any(clusters == i)
                else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            # Calculate the distance that each centroid moved
            p = np.linalg.norm(new_centroids - self.centroids, axis=1)

            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) <= self.tol:
                break

            # Compute the new objective value
            new_objective = self._compute_objective(X, clusters, new_centroids)

            # Update the best centroids if the objective improves
            if new_objective < self.best_objective:
                self.best_centroids = new_centroids.copy()
                self.best_objective = new_objective

            # Always update centroids
            self.centroids = new_centroids
            if iteration % 10 == 0 and self.verbose:
                print(f"Iteration {iteration}: objective = {new_objective}")

    def _hamerly(self, X: np.ndarray) -> None:
        centroid_dists = np.zeros((self.n_clusters, self.n_clusters))
        distances = cdist(X, self.centroids, 'euclidean')
        clusters = np.argmin(distances, axis=1)
        upper_bounds = np.min(distances, axis=1)
        lower_bounds = np.partition(distances, 1, axis=1)[:, 1]
        p = np.zeros(self.n_clusters)

        for iteration in range(self.max_iter):
            # Step 1: Compute distances between centroids
            centroid_dists = cdist(self.centroids, self.centroids, 'euclidean')
            np.fill_diagonal(centroid_dists, np.inf)
            s = np.min(centroid_dists, axis=1) / 2

            # Step 2: Update bounds
            sorted_indices = np.argpartition(p, 2)
            r = sorted_indices[0]
            r_prime = sorted_indices[1]
            upper_bounds += p[clusters]
            mask = r == clusters
            lower_bounds[mask] -= p[r_prime]
            lower_bounds[~mask] -= p[r]

            # Step 3: Assignment step
            mask = upper_bounds > np.maximum(s[clusters], lower_bounds)
            distances = np.linalg.norm(
                X[mask][:, np.newaxis] - self.centroids, axis=2)
            upper_bounds[mask] = np.min(distances, axis=1)
            lower_bounds[mask] = np.partition(distances, 2, axis=1)[:, 1]
            clusters[mask] = np.argmin(distances, axis=1)

            # Step 4: Update centroids
            new_centroids = np.array([
                X[clusters == i].mean(axis=0) if np.any(clusters == i)
                else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            # Calculate the distance that each centroid moved
            p = np.linalg.norm(new_centroids - self.centroids, axis=1)

            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) <= self.tol:
                break

            # Compute the new objective value
            new_objective = self._compute_objective(X, clusters, new_centroids)

            # Update the best centroids if the objective improves
            if new_objective < self.best_objective:
                self.best_centroids = new_centroids.copy()
                self.best_objective = new_objective

            # Always update centroids
            self.centroids = new_centroids
            if iteration % 10 == 0 and self.verbose:
                print(f"Iteration {iteration}: objective = {new_objective}")

    def _assign_clusters(self, X: np.ndarray, point_index: Optional[int] = None, use_bounds: bool = False,
                         upper_bounds: Optional[np.ndarray] = None,
                         lower_bounds: Optional[np.ndarray] = None) -> Union[np.ndarray, int]:
        if point_index is None:
            if use_bounds and upper_bounds is not None and lower_bounds is not None:
                distances = np.full((X.shape[0], self.n_clusters), np.inf)
                for i in range(self.n_clusters):
                    mask = upper_bounds < lower_bounds[:, i]
                    distances[mask, i] = np.linalg.norm(
                        X[mask] - self.centroids[i], axis=1)
                return np.argmin(distances, axis=1)
            else:
                distances = cdist(X, self.centroids, 'euclidean')
                return np.argmin(distances, axis=1)
        else:
            distances = cdist(X[point_index:point_index + 1],
                              self.centroids, 'euclidean')
            return np.argmin(distances)

    def _compute_objective(self, X: np.ndarray, clusters: np.ndarray, centroids: np.ndarray) -> float:
        objective = 0.0
        for i in range(self.n_clusters):
            mask = clusters == i
            size = np.sum(mask)
            objective += size * \
                np.sum(np.linalg.norm(X[mask] - centroids[i], axis=1) ** 2)
        return objective

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)
