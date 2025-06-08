import contextlib
import logging
import time
from functools import wraps

import numpy as np

from geosss.mcmc import (
    MetropolisHastings,
    RejectionSphericalSliceSampler,
    ShrinkageSphericalSliceSampler,
    SphericalHMC,
)

EXP_MIN = -308
EXP_MAX = +709

LOG_MIN = 1e-308
LOG_MAX = 1e308

# some nice colors
colors = [
    (0.85, 0.3, 0.1),
    (0.15, 0.35, 0.6),
    (0.95, 0.7, 0.1),
    (0.0, 0.0, 0.0),
    (0.8, 0.8, 0.8),
]


def format_time(t):
    units = [(1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")]
    for scale, unit in units:
        if t > scale or t == 0:
            break

    return "{0:.1f} {1}".format(t / scale, unit)


@contextlib.contextmanager
def take_time(desc, mute=False):
    t0 = time.process_time()
    yield
    dt = time.process_time() - t0
    if not mute:
        logging.info("{0} took {1}".format(desc, format_time(dt)))


def exp(x, x_min=EXP_MIN, x_max=EXP_MAX):
    """
    Safe version of exp, clips argument such that overflow does not occur.

    Parameters
    ----------
    x : input array or float or int
    x_min : lower value for clipping
    x_max : upper value for clipping

    Returns
    -------
    numpy array
    """

    x_min = max(x_min, EXP_MIN)
    x_max = min(x_max, EXP_MAX)

    return np.exp(np.clip(x, x_min, x_max))


def log(x, x_min=LOG_MIN, x_max=LOG_MAX):
    """
    Safe version of log, clips argument such that overflow does not occur.

    Parameters
    ----------
    x : input array or float or int
    x_min : lower value for clipping
    x_max : upper value for clipping
    """
    x_min = max(x_min, LOG_MIN)
    x_max = min(x_max, LOG_MAX)

    return np.log(np.clip(x, x_min, x_max))


def relative_entropy(p, q):
    """Kullback-Leibler divergence (aka as relative entropy)."""
    return p @ (log(p) - log(q))


def acf(x, n_max=None):
    """Autocorrelation.

    Parameters
    ----------
    x : 1d array
    n_max : maximum lag (if None then len(x) // 2)

    Returns
    -------
    Array storing the estimated autocorrelation.
    """
    n_max = n_max or len(x) // 2
    x = x - np.mean(x)
    ac = [np.mean(x[i:] * x[: len(x) - i]) for i in range(n_max)]
    return np.array(ac) / ac[0]


def acf_fft(x):
    """Compute autocorrelation function using the convolution theorem."""
    X = np.fft.rfft((x - x.mean()) / x.std())
    return np.real(np.fft.irfft(X.conj() * X))[: len(x) // 2] / len(x)


def IAT(x, n=None):
    """Computes the integrated autocorrelation time for given values by the heuristics
    described in Gelman et al "Bayesian Data Analysis", Chapter 11.5
    """
    ac = acf_fft(x)
    if n:
        ac = ac[:n]
    sums = (
        ac[2:-1].reshape(-1, 2).sum(1)
        if (len(ac) % 2 != 0)
        else ac[2:].reshape(-1, 2).sum(1)
    )
    T = np.nonzero(sums < 0)[0][0]
    L = (1 + 2 * T) if np.sum(sums < 0) else len(ac) - 1
    return 1.0 + np.max([2 * np.sum(ac[1 : L + 1]), 0.0])


def n_eff(x, n=None):
    """Computes effective sample size n_eff for given values"""
    return len(x) / IAT(x, n)


def count_calls(func):
    """
    decorator that counts how many times a method/function was called. If needed
    could be called directly only on a function with @count_calls
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper inside the `count_calls`"""
        # updates the counter everytime func is called
        wrapper.num_calls += 1
        return func(*args, **kwargs)

    # counter starts
    wrapper.num_calls = 0

    def reset_counters():
        """Reset the num_calls counter"""
        wrapper.num_calls = 0

    wrapper.reset_counters = reset_counters

    return wrapper


def counter(method_names):
    """
    A decorator that counts how many times a method was called. It calls the `count_calls` function to
    wrap around a class. So it should be used only on a class. Example: @counter("some_method") or
    @counter(["some_method", "another_method"])

    Args:
        method_names (str or list): the name(s) of the method(s) to decorate.

    Returns:
        A class decorator.
    """
    # if argument is a single method as string
    if isinstance(method_names, str):
        method_names = [method_names]

    # `class_decorator` calls `count_calls` decorator
    def class_decorator(cls):
        for method_name in method_names:
            if hasattr(cls, method_name):
                setattr(cls, method_name, count_calls(getattr(cls, method_name)))
        return cls

    return class_decorator


class SamplerLauncher:
    """Just an interface for launching all the samplers"""

    def __init__(self, pdf, initial, n_samples, burnin=0.2, seed=None):
        self.pdf = pdf
        self.initial = initial
        self.n_samples = n_samples
        self.burnin = burnin
        self.seed = seed

    def run_sss_reject(self):
        sampler = RejectionSphericalSliceSampler(self.pdf, self.initial, self.seed)
        self.rsss = sampler

        return sampler.sample(self.n_samples, burnin=self.burnin)

    def run_sss_shrink(self):
        sampler = ShrinkageSphericalSliceSampler(self.pdf, self.initial, self.seed)
        self.ssss = sampler

        return sampler.sample(self.n_samples, burnin=self.burnin)

    def run_rwmh(self):
        sampler = MetropolisHastings(self.pdf, self.initial, self.seed, stepsize=1e-1)
        self.rwmh = sampler

        return sampler.sample(self.n_samples, burnin=self.burnin)

    def run_hmc(self):
        sampler = SphericalHMC(self.pdf, self.initial, self.seed, stepsize=1e-1)
        self.hmc = sampler

        return sampler.sample(self.n_samples, burnin=self.burnin)

    def run(self, method):
        if method == "sss-reject":
            return self.run_sss_reject()
        elif method == "sss-shrink":
            return self.run_sss_shrink()
        elif method == "rwmh":
            return self.run_rwmh()
        elif method == "hmc":
            return self.run_hmc()
        else:
            raise ValueError(f"method {method} not known")
