"""
Pointcloud class
"""

import numpy as np
from scipy.spatial.transform import Rotation


class RotationMatrix:
    def __init__(self, degree=False):
        """
        Returns either 2 x 2 or 3 x 3 rotation matrix

        :param degree: bool
                    degree of the angles (default:False)
        """

        self.degree = degree
        self.rot_mat = None

    def create(self, angle):
        """create

        Creates a rotation matrix based on the size of the angle

        :param angle: scalar/array_like
                    scalar angle value or a vector of euler angles
        :return: rotation2d/rotation3d: attribute
                    returns the attribute based on the size of the angle
        """

        if self.degree:
            angle = np.deg2rad(angle)

        if angle.size == 1:
            return self.rotation2d(angle)
        elif angle.size == 3:
            return self.rotation3d(angle)
        else:
            ValueError("Angle vector should be of size 1 or 3")

    def rotation2d(self, angle: float) -> np.ndarray:
        """rotation2d

        2D Rotation matrix that rotates a vector in counterclockwise direction

        :param angle: scalar
                    angle
        :return: rot_mat: array_like
                    2 x 2 rotation matrix
        """

        self.rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return self.rot_mat

    def rotation3d(self, euler_angles) -> np.ndarray:
        """rotation3d

        3D intrinsic rotation matrix whose euler angles (alpha, beta, gamma)
        about axis 'z, y, z'

        :param euler_angles: array_like
                    euler_angles of length 3
        :return: rot_mat: array_like
                    3 x 3 rotation matrix
        """

        alpha, beta, gamma = euler_angles

        r_alpha = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )

        r_beta = np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)],
            ]
        )

        r_gamma = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1.0],
            ]
        )

        self.rot_mat = r_alpha @ r_beta @ r_gamma

        return self.rot_mat


def quat2matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : array_like
        A 4-element quaternion (x, y, z, w). Scalar last convention.

    Returns
    -------
    R : array_like
        A 3x3 rotation matrix
    """
    return Rotation.from_quat(q).as_matrix()


def matrix2quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion. Scalar last convention.

    Parameters
    ----------
    R : array_like
        A 3x3 rotation matrix.

    Returns
    -------
    array_like
        A 4-element quaternion (x, y, z, w).
    """
    return Rotation.from_matrix(R).as_quat()


def jacobian_rotation_matrix(q):
    """
    Compute the Jacobian of the rotation matrix with respect to quaternion components.

    Parameters
    ----------
    q : ndarray
        Unit quaternion in scalar-last format (x, y, z, w).

    Returns
    -------
    J : ndarray
        Jacobian tensor ∂R/∂q (shape: 3x3x4).
    """

    x, y, z, w = q
    r = w**2 + x**2 + y**2 + z**2 + 1e-300

    # Initialize Jacobian
    J = np.zeros((3, 3, 4))
    A, B, C, D = J[..., 0], J[..., 1], J[..., 2], J[..., 3]

    # compute Jacobian

    # ∂R/∂x
    A[0, 0] = -2 * x * (w**2 + x**2 - y**2 - z**2) / r**2 + 2 * x / r
    A[0, 1] = -2 * x * (-2 * w * z + 2 * x * y) / r**2 + 2 * y / r
    A[0, 2] = -2 * x * (2 * w * y + 2 * x * z) / r**2 + 2 * z / r
    A[1, 0] = -2 * x * (2 * w * z + 2 * x * y) / r**2 + 2 * y / r
    A[1, 1] = -2 * x * (w**2 - x**2 + y**2 - z**2) / r**2 - 2 * x / r
    A[1, 2] = -2 * w / r - 2 * x * (-2 * w * x + 2 * y * z) / r**2
    A[2, 0] = -2 * x * (-2 * w * y + 2 * x * z) / r**2 + 2 * z / r
    A[2, 1] = 2 * w / r - 2 * x * (2 * w * x + 2 * y * z) / r**2
    A[2, 2] = -2 * x * (w**2 - x**2 - y**2 + z**2) / r**2 - 2 * x / r

    # ∂R/∂y
    B[0, 0] = -2 * y * (w**2 + x**2 - y**2 - z**2) / r**2 - 2 * y / r
    B[0, 1] = 2 * x / r - 2 * y * (-2 * w * z + 2 * x * y) / r**2
    B[0, 2] = 2 * w / r - 2 * y * (2 * w * y + 2 * x * z) / r**2
    B[1, 0] = 2 * x / r - 2 * y * (2 * w * z + 2 * x * y) / r**2
    B[1, 1] = -2 * y * (w**2 - x**2 + y**2 - z**2) / r**2 + 2 * y / r
    B[1, 2] = -2 * y * (-2 * w * x + 2 * y * z) / r**2 + 2 * z / r
    B[2, 0] = -2 * w / r - 2 * y * (-2 * w * y + 2 * x * z) / r**2
    B[2, 1] = -2 * y * (2 * w * x + 2 * y * z) / r**2 + 2 * z / r
    B[2, 2] = -2 * y * (w**2 - x**2 - y**2 + z**2) / r**2 - 2 * y / r

    # ∂R/∂z
    C[0, 0] = -2 * z * (w**2 + x**2 - y**2 - z**2) / r**2 - 2 * z / r
    C[0, 1] = -2 * w / r - 2 * z * (-2 * w * z + 2 * x * y) / r**2
    C[0, 2] = 2 * x / r - 2 * z * (2 * w * y + 2 * x * z) / r**2
    C[1, 0] = 2 * w / r - 2 * z * (2 * w * z + 2 * x * y) / r**2
    C[1, 1] = -2 * z * (w**2 - x**2 + y**2 - z**2) / r**2 - 2 * z / r
    C[1, 2] = 2 * y / r - 2 * z * (-2 * w * x + 2 * y * z) / r**2
    C[2, 0] = 2 * x / r - 2 * z * (-2 * w * y + 2 * x * z) / r**2
    C[2, 1] = 2 * y / r - 2 * z * (2 * w * x + 2 * y * z) / r**2
    C[2, 2] = -2 * z * (w**2 - x**2 - y**2 + z**2) / r**2 + 2 * z / r

    # ∂R/∂w
    D[0, 0] = 4 * w * (y**2 + z**2) / r**2
    D[0, 1] = 2 * (2 * w * (w * z - x * y) - z * r) / r**2
    D[0, 2] = 2 * (-2 * w * (w * y + x * z) + y * r) / r**2
    D[1, 0] = 2 * (-2 * w * (w * z + x * y) + z * r) / r**2
    D[1, 1] = 4 * w * (x**2 + z**2) / r**2
    D[1, 2] = 2 * (2 * w * (w * x - y * z) - x * r) / r**2
    D[2, 0] = 2 * (2 * w * (w * y - x * z) - y * r) / r**2
    D[2, 1] = 2 * (-2 * w * (w * x + y * z) + x * r) / r**2
    D[2, 2] = 4 * w * (x**2 + y**2) / r**2

    return -J


class PointCloud:
    """PointCloud

    Create (weighted) point cloud from a list of points in 2D or 3D and
    optional weights.
    """

    def __init__(self, positions, weights=None):
        """
        :param positions: ndarray (n, m)
                Rank-2 array where the row index (n) enumerates points and a row stores
                the coordinates (m) in a 2D or 3D Euclidean space, i.e., dimension of the
                pasition.
        :param weights: ndarray
                Rank-2 array where the row index enumerates points and a row stores
                the coordinates in a 2D or 3D Euclidean space.
        """

        if np.ndim(positions) != 2:
            raise ValueError("Expected rank-2 array")
        if np.shape(positions)[1] not in (2, 3):
            raise ValueError("Expected 2d or 3d point cloud")
        if weights is None:
            weights = np.ones(len(positions))
        self.positions = np.array(positions)
        self.weights = np.array(weights)

    @property
    def dim(self):
        """
        Dimensionality of space in which point cloud lives.
        """
        return self.positions.shape[1]

    @property
    def size(self):
        return self.positions.shape[0]

    @property
    def center_of_mass(self):
        """
        Center of mass where the weight is considered the mass of the points.
        """

        return self.weights @ self.positions / self.weights.sum()

    def transform_positions(self, rotation, translation=None):
        """
        Returns the rigidly transformed positions, if the rotation is a unit
        quaternion, convert to rotation matrix and transform.
        """

        rotation = quat2matrix(rotation) if np.shape(rotation)[-1] == 4 else rotation

        if translation is None:
            translation = np.zeros(len(rotation))

        # rotate the 3d or 2d point cloud
        return self.positions.dot(rotation.T) + translation

    def transform(self, rotation, translation=None):
        """
        Changes the stored positions.
        """
        self.positions = self.transform_positions(rotation, translation)


class RotationProjection(PointCloud):
    """ProjectedCloud

    Rotation of 3D point cloud followed by a projection (parallel beam geometry)
    into 2D after X-ray transform
    """

    def transform_positions(self, rotation, translation=None):
        """
        Projects a 3D point cloud into 2D by first applying a rotation
        followed by a projection (keeping only the xy coordinates). The
        rotation can be given either as a rotation matrix or a unit quaternion.
        """

        # if rotation is unit quaternion, convert to rotation matrix
        rotation = quat2matrix(rotation) if np.shape(rotation)[-1] == 4 else rotation

        if translation is None:
            translation = np.zeros(self.dim - 1)
        # return the rotated and projected (2D) point cloud
        return self.positions @ rotation[:-1].T + translation
