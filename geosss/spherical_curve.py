import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as opt
from scipy.spatial.distance import cdist

from geosss import sphere
from geosss.tsp_solver import solve_tsp


def distance_slerp(x, a, b):
    """Compute the shortest distance between a point `x` on the sphere and the slerp
    connecting two points on the sphere `a`, `b`.

    Parameters
    ----------
    x : point on sphere
        Query point whose distance from the slerp will be determined
    a : point on sphere
        Start of slerp
    b : point on sphere
        End of slerp

    Returns
    -------
    * Shorest geodesic distance from point to slerp
    * Closest point on slerp
    """
    theta = sphere.distance(a, b)
    t = np.arctan2(b @ x - a @ x * np.cos(theta), a @ x * np.sin(theta))
    t = np.clip(t, 0.0, theta)
    y = (np.sin(theta - t) * a + np.sin(t) * b) / (np.sin(theta) + 1e-10)
    return sphere.distance(x, y), y


class SphericalCurve:
    def __init__(self, knots):
        self.knots = knots

    @classmethod
    def random_curve(cls, n_knots=10, seed=None, dimension=3):
        """Generate random points on an N-sphere and order them by finding the shortest
        path that runs through all points. These points will be interpolated by a curve
        on the sphere.
        """
        x = sphere.sample_sphere(dimension - 1, n_knots, seed)
        return cls(x[np.array(solve_tsp(cdist(x, x)))])

    def find_nearest(self, point, *args, **kwargs):
        pass


class SphericalSpline(SphericalCurve):
    def __init__(self, knots):
        super().__init__(knots)
        self.spline, _ = interpolate.splprep(knots.T, s=2)

    def __call__(self, t):
        return sphere.map_to_sphere(np.transpose(interpolate.splev(t, self.spline)))

    def find_nearest(self, point, n_pts=100):
        def func(t, x=point, curve=self):
            return np.linalg.norm(curve(t) - x)

        t = np.linspace(0.0, 1.0, n_pts)
        t0 = t[np.linalg.norm(self(t) - point, axis=1).argmin()]
        if np.isclose(t0, 0.0) or np.isclose(t0, 1.0):
            bracket = (0.0, 1.0)
        else:
            bracket = (0.0, t0, 1.0)
        t = opt.minimize_scalar(func, bracket=bracket).x
        return self(t)


class SlerpCurve(SphericalCurve):
    """SlerpCurve
    Spherical linear interpolation (SLERP) between the defined knots on the N-sphere
    """

    def __init__(self, knots):
        super().__init__(knots)
        self.theta = np.array([sphere.distance(a, b) for a, b in zip(knots, knots[1:])])
        self.theta = np.append(0.0, self.theta)
        self.bins = np.add.accumulate(self.theta)
        self.bins /= self.bins[-1]
        self.widths = np.diff(self.bins)

    def __call__(self, t):
        s = self.bins
        i = np.clip(np.digitize(t, s, right=True), 1, len(self.knots) - 1)
        theta = self.theta[i]
        a = np.sin(theta * (s[i] - t) / (s[i] - s[i - 1])) / np.sin(theta)
        b = np.sin(theta * (t - s[i - 1]) / (s[i] - s[i - 1])) / np.sin(theta)
        return a[:, None] * self.knots[i - 1] + b[:, None] * self.knots[i]

    def find_nearest(self, point):
        distances = []
        points = []
        for a, b in zip(self.knots, self.knots[1:]):
            d, y = distance_slerp(point, a, b)
            distances.append(d)
            points.append(y)
        return points[np.argmin(distances)]


def brownian_curve(n_points=100, dimension=6, step_size=0.05, seed=1234):
    """
    Generate smooth points on a unit d-sphere using Brownian motion.

    Parameters:
    n_points: number of points to generate
    dimension: dimension of the sphere
    step_size: step size for the Brownian motion, increasing it will make the points jump more
    seed: random seed
    """
    rng = np.random.default_rng(seed)

    # Initialize the first point on the dimension-1 -sphere
    points = np.zeros((n_points, dimension))
    points[0] = sphere.radial_projection(rng.standard_normal(dimension))

    # Generate subsequent points via Brownian motion (small random steps)
    for i in range(1, n_points):
        step = rng.normal(size=dimension) * step_size
        new_point = points[i - 1] + step

        # Project back to the unit sphere
        points[i] = sphere.radial_projection(new_point)

    return points


def constrained_brownian_curve(n_points=100, dimension=6, step_size=0.05, seed=1234):
    """
    Generate smooth points on a unit d-sphere using constrained Brownian motion
    to avoid loops and overlaps.

    Parameters:
    n_points: number of points to generate
    dimension: dimension of the sphere (must be at least 2)
    step_size: step size for the motion
    seed: random seed
    """
    rng = np.random.default_rng(seed)

    # Initialize the first point on the (dimension - 1)-sphere
    points = np.zeros((n_points, dimension))
    points[0] = sphere.radial_projection(rng.standard_normal(dimension))

    # Initialize the direction of motion (tangent vector at the first point)
    v = rng.standard_normal(dimension)
    v -= np.dot(v, points[0]) * points[0]  # Make v orthogonal to points[0]
    v /= np.linalg.norm(v)  # Normalize the tangent vector

    for i in range(1, n_points):
        # Generate a small random perturbation orthogonal to both v and points[i - 1]
        random_step = rng.standard_normal(dimension)

        # Make the random perturbation tangent to the sphere at points[i - 1]
        random_step -= np.dot(random_step, points[i - 1]) * points[i - 1]

        # Make the random perturbation orthogonal to the current direction v
        random_step -= np.dot(random_step, v) * v

        # Normalize the random_step so that it lies on the tangent plane
        random_step /= np.linalg.norm(random_step)

        # Scale the random_step with the given step size
        random_step *= step_size

        # Update the tangent vector v
        v += random_step

        # Ensure the updated tangent vector v remains tangent to the sphere
        v -= np.dot(v, points[i - 1]) * points[i - 1]
        v /= np.linalg.norm(v)

        # Move the point along v by a small angle (step_size)
        angle = step_size
        points[i] = points[i - 1] * np.cos(angle) + v * np.sin(angle)

    return points
