import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as opt
from scipy.spatial.distance import cdist
from tsp_solver.greedy import solve_tsp

from geosss import sphere


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
    y = (np.sin(theta - t) * a + np.sin(t) * b) / np.sin(theta)
    return sphere.distance(x, y), y


class SphericalCurve:
    def __init__(self, knots):
        self.knots = knots

    @classmethod
    def random_curve(cls, n_pts=10, seed=None, dimension=3):
        """Generate random points on an N-sphere and order them by finding the shortest
        path that runs through all points. These points will be interpolated by a curve
        on the sphere.
        """
        x = sphere.sample_sphere(dimension - 1, n_pts, seed)
        return cls(x[np.array(solve_tsp(cdist(x, x)))])


class SphericalSpline(SphericalCurve):
    def __init__(self, knots):
        super().__init__(knots)
        self.spline, _ = interpolate.splprep(knots.T, s=2)

    def __call__(self, t):
        return sphere.map_to_sphere(np.transpose(interpolate.splev(t, self.spline)))

    def find_nearest(self, x, n_pts=100):
        def func(t, x=x, curve=self):
            return np.linalg.norm(curve(t) - x)

        t = np.linspace(0.0, 1.0, n_pts)
        t0 = t[np.linalg.norm(self(t) - x, axis=1).argmin()]
        if np.isclose(t0, 0.0) or np.isclose(t0, 1.0):
            bracket = (0.0, 1.0)
        else:
            bracket = (0.0, t0, 1.0)
        t = opt.minimize_scalar(func, bracket=bracket).x
        return self(t)


class SlerpCurve(SphericalCurve):
    """SlerpCurve
    Spherical linear interpolation between the defined knots on the N-sphere
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

    def find_nearest(self, x):
        distances = []
        points = []
        for a, b in zip(self.knots, self.knots[1:]):
            d, y = distance_slerp(x, a, b)
            distances.append(d)
            points.append(y)
        return points[np.argmin(distances)]
