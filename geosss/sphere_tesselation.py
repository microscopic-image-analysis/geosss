import itertools

import numpy as np
from scipy.spatial.transform import Rotation


def get_unique_rotations(rotations, decimals=10):
    """
    Remove duplicate rotation matrices by rounding and finding unique entries.

    Parameters
    ----------
    rotations : array_like
        An array of rotation matrices.
    decimals : int, optional
        Number of decimal places to round to when determining uniqueness. Default is 10.

    Returns
    -------
    array_like
        An array of unique rotation matrices.
    """
    # Get the shape of a single rotation matrix
    shape = np.shape(rotations)[1:]

    # Flatten each rotation matrix for comparison
    rotations = np.reshape(rotations, (len(rotations), -1))

    # Round the rotations and find unique matrices
    unique = np.unique(np.round(rotations, decimals=decimals), axis=0)

    # Reshape back to original matrix dimensions
    return np.reshape(unique, (len(unique),) + shape)


class C600:
    """C600

    600-Cell

    Factory class that generates a tesselation of the unit sphere in 4D
    used to cover rotation space
    """

    even_perms = (
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    )

    def __init__(self, upper_sphere=True):
        self.vertices = self.__class__.make_vertices(upper_sphere)

    @classmethod
    def make_vertices(cls, upper_sphere=True):
        base = [[-1, 1]] * 4
        nodes = np.zeros((120, 4))
        k = 2**4

        nodes[:k] = list(itertools.product(*base))

        for j in range(4):
            for s in [-2, 2]:
                nodes[k, j] = s
                k += 1

        # golden ratio
        phi = (1 + np.sqrt(5)) / 2

        base = base[:3]
        for perm in cls.even_perms:
            for a, b, c in itertools.product(*base):
                nodes[k, perm[0]] = a * phi
                nodes[k, perm[1]] = b
                nodes[k, perm[2]] = c / phi
                nodes[k, perm[3]] = 0
                k += 1

        # normalize
        nodes *= 0.5

        # keep nodes covering half sphere
        if upper_sphere:
            north = np.eye(4)[0]
            mask = np.arccos(np.dot(nodes, north)) <= np.deg2rad(120)
            nodes = nodes[mask]

        return nodes

    def create_tetrahedra(self):
        angles = np.round(np.rad2deg(np.arccos(self.vertices @ self.vertices.T)))
        min_val = 36.0
        mask = angles == min_val

        for i_idx, j_idx, k_idx, l_idx in itertools.combinations(
            range(len(self.vertices)), 4
        ):
            if (
                mask[i_idx, j_idx]
                and mask[i_idx, k_idx]
                and mask[i_idx, l_idx]
                and mask[j_idx, k_idx]
                and mask[j_idx, l_idx]
                and mask[k_idx, l_idx]
            ):
                yield self.vertices[np.array([i_idx, j_idx, k_idx, l_idx])]


def split_tetrahedron(vertices):
    """Subdivide tetrahedron into eight tetrahedra and project the inner
    corners of the new tetrahedra to S3.

    Reference:
    https://www.ams.org/journals/mcom/1996-65-215/S0025-5718-96-00748-X/
    S0025-5718-96-00748-X.pdf
    """
    edge_indices = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

    def normed(x):
        return x / np.linalg.norm(x)

    nodes = np.zeros((6, 4))

    # compute nodes in center of edges
    for k, (i, j) in enumerate(edge_indices):
        nodes[k] = normed(vertices[i] + vertices[j])

    # check which skew edge is the shortest
    pairs = [(0, 5), (2, 4), (3, 1)]
    dots = np.array([nodes[i] @ nodes[j] for i, j in pairs])
    index = np.argmax(dots)

    tetrahedra = [
        (vertices[0], nodes[0], nodes[2], nodes[3]),
        (vertices[1], nodes[0], nodes[1], nodes[4]),
        (vertices[2], nodes[1], nodes[2], nodes[5]),
        (vertices[3], nodes[3], nodes[4], nodes[5]),
    ]

    if index == 0:
        tetrahedra += [
            (nodes[0], nodes[5], nodes[3], nodes[2]),
            (nodes[0], nodes[5], nodes[4], nodes[3]),
            (nodes[5], nodes[0], nodes[1], nodes[4]),
            (nodes[5], nodes[0], nodes[1], nodes[2]),
        ]
    elif index == 1:
        tetrahedra += [
            (nodes[0], nodes[4], nodes[3], nodes[2]),
            (nodes[0], nodes[1], nodes[4], nodes[2]),
            (nodes[5], nodes[2], nodes[1], nodes[4]),
            (nodes[5], nodes[3], nodes[2], nodes[4]),
        ]
    elif index == 2:
        tetrahedra += [
            (nodes[3], nodes[1], nodes[0], nodes[2]),
            (nodes[3], nodes[1], nodes[4], nodes[0]),
            (nodes[3], nodes[1], nodes[5], nodes[4]),
            (nodes[1], nodes[3], nodes[2], nodes[5]),
        ]

    return np.array(tetrahedra)


def tessellate_rotations(n_discretize=2):
    """
    Discretize tetrahedra into finer tetrahedra and returns as quaternion
    based on the degree of discretization.

    :param n_discretize: positive int
        degree of discretization
    :return: discretized_quat: ndarray
        discretized tetrahedra converted to quaternion
    """
    cell600 = C600()
    tetrahedra = np.array(list(cell600.create_tetrahedra()))

    for i in range(n_discretize):
        # split every tetrahedra into eight new tetrahedra
        tetrahedra = np.reshape(list(map(split_tetrahedron, tetrahedra)), (-1, 4, 4))

    discretized_quat = tetrahedra.mean(axis=1)
    discretized_quat /= np.linalg.norm(discretized_quat, axis=1)[:, None]

    # convert to rotation matrix
    rotations = [Rotation.from_quat(q).as_matrix() for q in discretized_quat]
    rotations = get_unique_rotations(rotations)

    # convert back to quaternion
    discretized_quat = [Rotation.from_matrix(R).as_quat() for R in rotations]

    return np.array(discretized_quat)
