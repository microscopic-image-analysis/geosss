# This file tests 3D-3D and 3D-2D CPD implementations on toy data, compares analytical
# with numerical gradients and infered via optimization.

import numpy as np
from scipy.optimize import approx_fprime

from geosss.pointcloud import (
    PointCloud,
    RotationMatrix,
    RotationProjection,
    matrix2quat,
)
from geosss.registration import CoherentPointDrift as CPD


def test_cpd_3d_to_3d():
    """Test CPD on 3D-3D registration with outliers"""
    print("=== Testing 3D-3D Registration with CPD ===")

    # Create a 3D source point cloud (cube vertices)
    source_positions = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    source = PointCloud(source_positions)

    # Create ground truth rotation
    rotmat = RotationMatrix()
    R_true = rotmat.rotation3d(euler_angles=np.array([0.2, 0.3, 0.1]))
    t_true = np.array([0.5, -0.3, 0.2])

    # Transform source to create target
    target_positions = source.transform_positions(R_true, t_true)
    # Add noise
    target_positions += np.random.normal(0, 0.05, target_positions.shape)

    # Add outliers
    outliers = np.random.uniform(-3, 3, (4, 3))
    target_positions = np.vstack([target_positions, outliers])
    target = PointCloud(target_positions)

    # Test with different outlier weights
    for w in [0.0, 0.2, 0.4]:
        print(f"\nTesting with outlier weight w={w}")

        cpd = CPD(target, source, sigma=0.5, k=8, beta=1.0, omega=w)

        # Test log probability at true pose
        log_prob_true = cpd.log_prob(R_true, t_true)
        print(f"Log prob at true pose: {log_prob_true:.4f}")

        # Test at identity (should be lower)
        log_prob_identity = cpd.log_prob(np.eye(3), np.zeros(3))
        print(f"Log prob at identity: {log_prob_identity:.4f}")

        # Numerical gradient check
        q_true = matrix2quat(R_true)
        grad_analytical = cpd.gradient(q_true, t_true)

        # compute numerical gradient
        def log_prob_wrapper(q):
            return cpd.log_prob(q, t_true)

        grad_numerical = approx_fprime(q_true, log_prob_wrapper, epsilon=1e-6)
        print(
            f"Gradient check (max diff): {np.max(np.abs(grad_analytical - grad_numerical)):.6f}"
        )
    return source, target, R_true, t_true, cpd


def test_cpd_3d_to_2d():
    """Test CPD on 3D-2D registration with outliers"""
    print("\n\n=== Testing 3D-2D Registration with CPD ===")

    # Create a 3D source point cloud
    source_positions = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    source = RotationProjection(source_positions)  # Note: using RotationProjection

    # Create ground truth rotation
    rotmat = RotationMatrix()
    R_true = rotmat.rotation3d(euler_angles=np.array([0.2, 0.3, 0.1]))
    t_true = np.array([0.5, -0.3])  # 2D translation

    # Project to create 2D target
    target_positions = source.transform_positions(R_true, t_true)
    # Add noise
    target_positions += np.random.normal(0, 0.05, target_positions.shape)

    # Add 2D outliers
    outliers = np.random.uniform(-3, 3, (4, 2))
    target_positions = np.vstack([target_positions, outliers])
    target = PointCloud(target_positions)

    # Test with different outlier weights
    for w in [0.0, 0.2, 0.4]:
        print(f"\nTesting with outlier weight w={w}")

        cpd = CPD(target, source, sigma=0.5, k=8, beta=1.0, omega=w)

        # Test log probability at true pose
        log_prob_true = cpd.log_prob(R_true, t_true)
        print(f"Log prob at true pose: {log_prob_true:.4f}")

        # Test at identity (should be lower)
        log_prob_identity = cpd.log_prob(np.eye(3), np.zeros(2))
        print(f"Log prob at identity: {log_prob_identity:.4f}")

        # Numerical gradient check
        q_true = matrix2quat(R_true)
        grad_analytical = cpd.gradient(q_true, t_true)

        # compute numerical gradient
        def log_prob_wrapper(q):
            return cpd.log_prob(q, t_true)

        grad_numerical = approx_fprime(q_true, log_prob_wrapper, epsilon=1e-6)

        print(
            f"Gradient check (max diff): {np.max(np.abs(grad_analytical - grad_numerical)):.6f}"
        )

    return source, target, R_true, t_true, cpd


if __name__ == "__main__":
    # Run all tests
    test_cpd_3d_to_3d()
    test_cpd_3d_to_2d()

    print("\n\nAll tests completed!")
