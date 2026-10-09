# K-Means Clustering

import torch


def k_means_clustering(
    points, k, initial_centroids, max_iterations
) -> list[tuple[float, ...]]:
    """
    Perform k-means clustering on `points` into `k` clusters.
    points: tensor of shape (n_points, n_features)
    initial_centroids: tensor of shape (k, n_features)
    max_iterations: maximum number of iterations
    Returns a list of k centroids as tuples, rounded to 4 decimals.
    """
    # Convert to tensors
    points_t = torch.as_tensor(points, dtype=torch.float)
    centroids = torch.as_tensor(initial_centroids, dtype=torch.float)
    n_points, n_features = points_t.shape

    for i in range(max_iterations):
        dist = torch.cdist(points_t, centroids, p=2)
        _, cid = torch.min(dist, dim=1)
        new_centroids = torch.zeros_like(centroids)
        count = torch.bincount(cid, minlength=k)

        new_centroids.index_add_(0, cid, points_t)

        # for j in range(n_points):
        #     new_centroids[cid[j]] += points_t[j]
        
        new_centroids /= count
        
        # TODO: prevent empty cluster
        # mask = count > 0
        # new_centroids[mask] /= count[mask,None]
        # new_centroids[~mask] = centroids[~mask]

        # new_centroids[g] /= count[g]

        centroids = new_centroids

    return centroids.tolist()


def main():
    print(
        k_means_clustering(
            [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
            2,
            [(1, 1), (10, 1)],
            10,
        )
    )


if __name__ == "__main__":
    main()


#
# Created By jing At 2026-08-18 17:18:00
#
