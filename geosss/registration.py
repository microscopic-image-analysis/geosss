from functools import cached_property
from typing import Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from geosss.distributions import Distribution
from geosss.pointcloud import (
    PointCloud,
    RotationProjection,
    jacobian_rotation_matrix,
    quat2matrix,
)
from geosss.utils import counter


class Registration(Distribution):
    """
    Abstract class assigning a log-probability to a rigid body pose.
    Concrete subclasses need to implement the *SO(3)-level* score.
    """

    def _log_prob_R(self, rotation, translation=None) -> float:
        """Score evaluated at a *rotation matrix* R (and optional translation)."""

    def _grad_R(self, rotation, translation=None) -> np.ndarray:
        """∂ log_prob / ∂ R  as a 3x3 array (same sign as _log_prob_R)."""

    beta: float = 1.0  # can be over-ridden

    @staticmethod
    def _to_quat(rotation):
        """converts to quaterion if a rotation matrix"""
        return (
            rotation
            if rotation.shape == (4,)  # already a quat
            else Rotation.from_matrix(rotation).as_quat()
        )

    @staticmethod
    def _jacobian_R_q(q):
        return jacobian_rotation_matrix(q).reshape(9, 4)  # (9,4)

    # -------- main API the samplers will use -----------------------------
    def log_prob(self, rotation, translation=None):
        R = (
            rotation
            if rotation.shape == (3, 3) or rotation.shape == (2, 2)
            else quat2matrix(rotation)
        )
        return self.beta * self._log_prob_R(R, translation)

    def gradient(self, rotation, translation=None):
        q = self._to_quat(rotation)
        gR_flat = (self.beta * self._grad_R(q, translation)).ravel()
        grad_q = self._jacobian_R_q(q).T @ gR_flat
        return grad_q


@counter(["log_prob", "gradient"])
class GaussianMixtureModel(Registration):
    """
    Gaussian mixture model for scoring 3D-3D or 3D-2D rigid pose by employing
    KD Tree for nearest neighbour search
    """

    def __init__(
        self,
        target: PointCloud,
        source: PointCloud | RotationProjection,
        sigma: float = 1.0,
        k: int = 20,
        *,
        beta: float = 1.0,
    ):
        """
        Initializes the KDTreeGaussianMixtureModel with the target and source point clouds.

        Parameters
        ----------
        target : PointCloud
            The fixed point cloud onto which the moving cloud will be superimposed.
        source : PointCloud | RotationProjection
            The moving point cloud that is rigidly transformed to match the fixed point cloud.
        sigma : float, optional
            The standard deviation of the Gaussian mixture, default is 1.0.
        k : int, optional
            The number of nearest neighbors to consider for each point in the target point cloud, default is 20.
        beta : float, optional
            An inverse temperature parameter that controls the softness of the assignment, default is 1.0.
        """

        self.target = target
        self.source = source
        self.sigma = float(sigma)
        self.k = int(k)
        self.beta = float(beta)

    def _query(
        self, R: np.ndarray, translation: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # create the KD tree with transformed (rotated and projected) source and
        # query the target
        trans_src_pos = self.source.transform_positions(R, translation)  # (Ns,2)
        tree = KDTree(trans_src_pos)
        dist, ind = tree.query(self.target.positions, k=self.k)  # (Lt, k)
        return trans_src_pos, dist, ind

    def _log_prob_R(
        self, R: np.ndarray, translation: np.ndarray | None = None
    ) -> float:
        _, dist, ind = self._query(R, translation)
        log_phi = -0.5 * dist**2 / self.sigma**2
        log_const = self.target.dim / 2 * np.log(2 * np.pi * self.sigma**2)
        log_w = np.log(self.source.weights[ind])  # (Lt,k)
        logp = log_phi - log_const + log_w
        return logsumexp(logp, axis=1) @ self.target.weights

    def _grad_R(
        self, R: np.ndarray, translation: np.ndarray | None = None
    ) -> np.ndarray:
        # Query the KD tree and return the transformed source positions
        trans_src_pos, dist, ind = self._query(R, translation)

        # Calculate log probabilities for numerical stability
        log_phi = -0.5 * dist**2 / self.sigma**2  # (Lt,k)
        log_const = self.target.dim / 2 * np.log(2 * np.pi * self.sigma**2)
        log_w_k = np.log(self.source.weights[ind])  # (Lt,k)

        # Compute log probabilities (same as in _log_prob_R)
        log_p_lk = log_phi - log_const + log_w_k  # (Lt,k)

        # Compute log normalizer for each target point
        log_p_l = logsumexp(log_p_lk, axis=1, keepdims=True)  # (Lt,1)

        # Compute posterior probabilities in log space for stability
        log_gamma = log_p_lk - log_p_l  # (Lt,k)
        log_gamma = np.clip(log_gamma, -20, 0)  # prevent underflow
        gamma = np.exp(log_gamma)  # (Lt,k)

        # Target positions
        y_l = self.target.positions  # (Lt, target_dim)

        # Get original source positions
        x_k_orig = self.source.positions[ind]  # (Lt, k, source_dim)
        x_k_trans = trans_src_pos[ind]  # (Lt, k, target_dim)

        # Handle dimensionality mismatch by padding target and transformed source
        # positions to source dimension
        pad_amount = self.source.dim - self.target.dim
        y_l_padded = np.pad(y_l, ((0, 0), (0, pad_amount)), mode="constant")
        x_k_padded = np.pad(
            x_k_trans, ((0, 0), (0, 0), (0, pad_amount)), mode="constant"
        )

        # Compute residuals and gradient
        d_lk = y_l_padded[:, None, :] - x_k_padded
        coeff = (self.target.weights[:, None] * gamma) / self.sigma**2
        grad_R = np.einsum("lk,lkj,lki->ji", coeff, d_lk, x_k_orig)

        return -grad_R


@counter(["log_prob", "gradient"])
class CoherentPointDrift(GaussianMixtureModel):
    """
    This is essentially the same as GMM except the outlier term
    that will be introduced here.
    """

    def __init__(
        self,
        target: PointCloud,
        source: PointCloud | RotationProjection,
        sigma: float = 1.0,
        k: int = 20,
        *,
        beta: float = 1.0,
        omega: float = 0.0,
    ):
        """
        Parameters
        ----------
        target : PointCloud
            Fixed point cloud onto which moving cloud will be superimposed.
        source : PointCloud | RotationProjection
            Moving point cloud that is rigidly transformed so as to match the
            fixed point cloud.
        sigma : positive float, optional
            Standard deviation of the Gaussian mixture. Default is 1.0.
        k : positive int, optional
            Number of nearest neighbors to consider for each point in the target
            point cloud. Default is 20.
        beta : positive float, optional
            Inverse temperature parameter that controls the softness of the
            assignment. However this is not considered in the probablistic model and
            therefore is set to default as 1.0 always.
        omega : float, optional
            Outlier weight parameter, between 0 (no outliers) and 1 (all points
            are outliers). Default is 0.0 (no outliers).
        """
        super().__init__(target, source, sigma, k, beta=beta)
        self.omega = float(omega)

        # approx. smallest double-precision number
        # to avoid underflow
        self.log_eps = 1e-308

    @cached_property
    def _log_volume(self):
        """
        Log normalizing constant of the outlier distribution (which is assumed
        to be a box containing all points from the target point clouds).
        """
        return np.sum(np.log(np.ptp(self.target.positions, 0)))

    @cached_property
    def _log_constant(self):
        return np.log(1 - self.omega) - (
            self.target.dim / 2 * np.log(2 * np.pi * self.sigma**2)
        )

    def _compute_log_posterior_terms(
        self, R: np.ndarray, translation: np.ndarray | None = None
    ):
        """Compute log probabilities and posteriors needed for both log_prob and gradient."""
        trans_src_pos, dist, ind = self._query(R, translation)

        # Compute log probabilities
        # log likelihood terms
        log_phi = -0.5 * dist**2 / self.sigma**2
        log_w_k = np.log(self.source.weights[ind])
        log_p_lk = log_w_k + log_phi + self._log_constant

        # log outlier term
        log_outlier = np.log(self.omega + self.log_eps) - self._log_volume
        outlier_column = np.full((self.target.size, 1), log_outlier)

        # Stack Gaussian components with outlier term
        log_probs = np.hstack((log_p_lk, outlier_column))  # (Lt, k+1)
        log_p_l = logsumexp(log_probs, axis=1)  # (Lt,)
        return (
            log_p_l,
            ind,
            trans_src_pos,
            log_p_lk,
        )

    def _log_prob_R(
        self, R: np.ndarray, translation: np.ndarray | None = None
    ) -> float:
        """Compute log probability with optional outlier handling."""

        # compute the total log probability
        log_p_l, *_ = self._compute_log_posterior_terms(R, translation)

        return log_p_l @ self.target.weights

    def _grad_R(self, R, translation=None):
        # Compute posterior terms
        log_p_l, ind, trans_src_pos, log_p_lk = self._compute_log_posterior_terms(
            R, translation
        )

        # Compute posterior probabilities (gamma) in log space for stability
        log_gamma = log_p_lk - log_p_l[:, None]  # (Lt, k)
        log_gamma = np.clip(log_gamma, -20, 0)  # prevent underflow
        gamma = np.exp(log_gamma)  # (Lt, k)

        # Now compute gradient using these posterior probabilities
        # Target positions
        y_l = self.target.positions  # (Lt, target_dim)

        # Get source positions (orig and transformed)
        x_k_orig = self.source.positions[ind]  # (Lt, k, source_dim)
        x_k_trans = trans_src_pos[ind]  # (Lt, k, source_dim)

        # handles dimensionality mismatch by padding target and transformed source
        # positions to source dimension
        pad_amount = self.source.dim - self.target.dim
        y_l_padded = np.pad(y_l, ((0, 0), (0, pad_amount)), mode="constant")
        x_k_padded = np.pad(
            x_k_trans, ((0, 0), (0, 0), (0, pad_amount)), mode="constant"
        )

        # Compute residuals and gradient
        d_lk = y_l_padded[:, None, :] - x_k_padded
        coeff = (self.target.weights[:, None] * gamma) / self.sigma**2
        grad_R = np.einsum("lk,lkj,lki->ji", coeff, d_lk, x_k_orig)

        return -grad_R
