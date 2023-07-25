# preliminary implementation for variants of slice samplers

import numpy as np

from . import sphere

# some nice colors
colors = [
    (0.85, 0.3, 0.1),
    (0.15, 0.35, 0.6),
    (0.95, 0.7, 0.1),
    (0.0, 0.0, 0.0),
    (0.8, 0.8, 0.8),
]


def sample_slice(log_prob, current, bounds, return_n_reject=False):
    """
    Bracketing procedure as presented in the paper.
    """
    threshold = log_prob(current) + np.log(np.random.rand())
    L, R = bounds
    x = np.random.uniform(0, 2 * np.pi)
    L, R = x - 2 * np.pi, x
    n_reject = 0

    while True:
        x = np.random.uniform(L, R)
        if log_prob(x) > threshold:
            if return_n_reject:
                return x, n_reject
            else:
                return x
        elif x < current:
            L = x
        else:
            R = x
        n_reject += 1


def sample_slice2(log_prob, current, bounds):
    """
    Alternative version in which the brackets are first shifted.
    """
    threshold = log_prob(current) + np.log(np.random.rand())
    x = np.random.uniform(*bounds)
    L, R = x + bounds[0], x + bounds[1]

    while True:
        if log_prob(x) > threshold:
            return x
        elif x < current:
            L = x
        else:
            R = x
        x = np.random.uniform(L, R)


def metropolis_hastings(
    pdf, n_samples, stepsize=0.5, adapt_stepsize=False, reproject=True
):
    """
    Sample hyperspherical distribution with Metropolis Hastings.
    """
    samples = []
    x = sphere.sample_sphere(pdf.d - 1)
    n_acc = 0

    while len(samples) < n_samples:
        # propose
        r = np.sqrt(np.random.gamma(0.5 * len(x)) / 0.5) if reproject else 1.0
        y = sphere.radial_projection(r * x + stepsize * np.random.randn(pdf.d))

        # accept / reject
        accept = np.log(np.random.rand()) < pdf.log_prob(y) - pdf.log_prob(x)
        x = y if accept else x

        # adapt stepsize
        if adapt_stepsize:
            stepsize *= 1.02 if accept else 0.98

        samples.append(np.copy(x))
        n_acc += int(accept)

    print(f"stepsize: {stepsize}, acceptance rate: {(n_acc/n_samples):.1%}")

    return np.array(samples)


def slice_sampling(pdf, n_samples=1e3):
    """
    Slice sampling on the sphere using first bracketing procedure.
    """
    bounds = (-np.pi, np.pi)

    # initialization
    v = sphere.sample_sphere(pdf.d - 1)
    samples = [v]
    n_reject = []

    while len(samples) < n_samples:
        # draw uniformly from subsphere
        u = sphere.sample_subsphere(v)

        # define probability on slice
        def log_prob(theta, pdf=pdf, v=v, u=u):
            return pdf.log_prob(np.cos(theta) * v + np.sin(theta) * u)

        # slice sampling using first bracketing procedure starting at current
        # sample (corresponding to theta=0)
        theta, n = sample_slice(log_prob, 0.0, bounds, return_n_reject=True)

        v = np.cos(theta) * v + np.sin(theta) * u

        samples.append(v)
        n_reject.append(n)

    print(f"average number of rejections: {np.mean(n_reject)}")

    return np.array(samples)


def slice_sampling2(pdf, n_samples=1e3):
    """
    Slice sampling on the sphere using second bracketing procedure.
    """
    bounds = (-np.pi, np.pi)

    # initialization
    v = sphere.sample_sphere(pdf.d - 1)
    samples = [v]
    n_reject = []

    while len(samples) < n_samples:
        # draw uniformly from subsphere
        u = sphere.sample_subsphere(v)

        # define probability on slice
        def log_prob(theta, pdf=pdf, v=v, u=u):
            return pdf.log_prob(np.cos(theta) * v + np.sin(theta) * u)

        # slice sampling using second bracketing procedure starting at current
        # sample (corresponding to theta=0)
        theta, n = sample_slice2(log_prob, 0.0, bounds, return_n_reject=True)

        v = np.cos(theta) * v + np.sin(theta) * u

        samples.append(v)
        n_reject.append(n)

    print(f"average number of rejections: {np.mean(n_reject)}")

    return np.array(samples)


def slice_sampling3(pdf, n_samples=1e3):
    """
    Slice sampling on the sphere using slerp and bracketing.
    """
    # initialization
    v = sphere.sample_sphere(pdf.d - 1)

    samples = [v]
    n_reject = []

    while len(samples) < n_samples:
        # draw uniformly from sphere
        u = sphere.sample_sphere(pdf.d - 1)

        # slerp
        slerp = sphere.slerp(u, v)
        theta = np.arccos(u @ v)  # - 0.5 * np.pi

        bounds = (theta - np.pi, theta + np.pi)

        # define probability on slice
        def log_prob(theta, pdf=pdf, slerp=slerp):
            return pdf.log_prob(slerp(theta))

        # slice sampling using first bracketing procedure starting at current
        # sample (corresponding to theta=0)
        theta, n = sample_slice(log_prob, theta, bounds, return_n_reject=True)

        v = slerp(theta)

        samples.append(v)
        n_reject.append(n)

    print(f"average number of rejections: {np.mean(n_reject)}")

    return np.array(samples)


def slice_sampling4(pdf, n_samples=1e3):
    """
    Slice sampling on the sphere using random Givens rotations and bracketing.
    """
    bounds = (-np.pi, np.pi)

    # initialization
    x = sphere.sample_sphere(pdf.d - 1)

    samples = [x]
    n_reject = []

    while len(samples) < n_samples:
        # draw uniformly from sphere
        u = sphere.sample_sphere(pdf.d - 1)
        v = sphere.sample_sphere(pdf.d - 1)  # generates a small circle?

        # how to construct a great circle?
        v = x  # also generates a small circle
        # unless
        u = sphere.spherical_projection(u, x)

        # Givens rotation
        givens = sphere.givens(u, v, x)

        # define probability on slice
        def log_prob(theta, pdf=pdf, givens=givens):
            return pdf.log_prob(givens(theta))

        assert np.isclose(log_prob(0.0), pdf.log_prob(x))

        # slice sampling using first bracketing procedure starting at current
        # sample (corresponding to theta=0)
        theta, n = sample_slice(log_prob, 0.0, bounds, return_n_reject=True)

        x = givens(theta)

        samples.append(x)
        n_reject.append(n)

    print(f"average number of rejections: {np.mean(n_reject)}")

    return np.array(samples)


def spherical_rejection_sampling(pdf, n_samples=1e3):
    """
    Use rejection sampling to sample from slice.
    """
    bounds = (-np.pi, np.pi)
    bounds = (0, 2 * np.pi)

    # initialization
    v = sphere.sample_sphere(pdf.d - 1)
    samples = [v]

    n_reject = []
    while len(samples) < n_samples:
        # draw uniformly from subsphere
        u = sphere.sample_subsphere(v)

        # define probability on slice
        def log_prob(theta, pdf=pdf, v=v, u=u):
            return pdf.log_prob(np.cos(theta) * v + np.sin(theta) * u)

        # rejection sampling where current sample defines threshold
        threshold = log_prob(0.0) + np.log(np.random.rand())
        n = 0
        while True:
            theta = np.random.uniform(*bounds)
            if log_prob(theta) > threshold:
                n_reject.append(n)
                break
            n += 1

        v = np.cos(theta) * v + np.sin(theta) * u

        samples.append(v)

    print(f"average number of rejections: {np.mean(n_reject)}")

    return np.array(samples)
