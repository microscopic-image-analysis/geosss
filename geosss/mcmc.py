"""
General purpose Markov chain Monte Carlo algorithms:
- Random walk Metropolis-Hastings algorithm on the sphere
- Spherical Hamiltonian Monte Carlo
- Slice sampling on the sphere using a rejection and a shrinkage sampler
"""

import numpy as np

from geosss import sphere


def determine_burnin(n_samples, burnin):
    if isinstance(burnin, float):
        assert 0 <= burnin <= 1.0
        return int(burnin * n_samples)
    else:
        assert burnin >= 0
        return int(burnin)


class Sampler:
    """Sampler

    Base class for Markov chain Monte Carlo samplers.
    """

    def __init__(self, distribution, initial_state, seed=None, **kwargs):
        """
        Parameters
        ----------
        distribution: instance of distributions.Distribution
            Target distribution implementing `log_prob`, stationary distribution
            of the Markov chain.

        initial_state: numpy ndarray
            Initial state of the Markov chain.

        seed: int, optional
            Seed for the random number generator.
        """
        super().__init__(**kwargs)
        self.target = distribution
        self.state = initial_state
        self.rng = np.random.default_rng(seed)

    def __next__(self):
        """
        Markov chain Monte Carlo sampler is implemented as an `iterator`. Subclasses
        need to implement the `__next__` routine which is supposed to simulate a single
        step of the Markov chain.
        """
        raise NotImplementedError

    def sample(self, n_samples, burnin=0, return_all_samples=False):
        """
        Generates a Markov chain consisting of the desired size. An optional burnin can
        be specified, either as an integer (in which case it is taken as is) or a float
        smaller than one (in which case it is interpreted as percentage of the desired
        number of samples).
        """
        assert n_samples > 0

        burnin = determine_burnin(n_samples, burnin)

        # simulate Markov chain
        samples = [self.state]
        while len(samples) < (n_samples + burnin):
            samples.append(np.copy(next(self)))

        if return_all_samples:
            # especially if we want to adapt step size during burnin, but want
            # all samples
            return np.array(samples)

        # chop away burn-in and return remaining samples
        return np.array(samples[burnin:])


class AdaptiveStepsize:
    """AdaptiveStepsize

    Mixin for adapting the stepsize in versions of Metropolis-Hastings such as random
    walk MH or spherical Hamiltonian Monte Carlo.

    Attributes
    ----------
    stepsize: float > 0
        Step-size parameter, e.g. variance of an isotropic Gaussian proposal.

    adapt_stepsize: boolean
        Flag indicating if the stepsize should be adapted or not. Typically, this flag
        is switched on in the initial phase of the simulation (burnin) and switched off
        after burn-in in order to ensure detailed balance.
    """

    def __init__(self, stepsize=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.stepsize = float(stepsize)
        assert self.stepsize > 0.0
        self.reset(0)

    def reset(self, burnin):
        self._counter = 0
        self._burnin = int(burnin)
        self._burnin >= 0

    def adapt_stepsize(self, accepted):
        """
        Adapts stepsize depending on whether the last attempt to generate a new state
        was successful (accepted=True) or not (accepted=False).
        """
        if self._counter < self._burnin:
            self.stepsize *= 1.02 if accepted else 0.98
            self._counter += 1


class MetropolisHastings(Sampler, AdaptiveStepsize):
    def __init__(self, distribution, initial_state, seed=None, stepsize=1e-1):
        """
        Parameters
        ----------
        distribution: instance of distributions.Distribution
            Target distribution implementing `log_prob`

        initial_state: numpy ndarray
            Initial state of the Markov chain.

        seed: int, optional
            Seed for the random number generator.

        stepsize: float > 0, optional
            Variance of isotropic Gaussian proposal.
        """
        super().__init__(distribution, initial_state, seed, stepsize=stepsize)
        self.n_accept = 0

    def propose(self):
        """
        Project into ambient space, perturb and project back to sphere.
        """
        x = np.copy(self.state)
        d = len(x)
        r = np.sqrt(2 * self.rng.gamma(d / 2))
        y = r * x + self.stepsize * self.rng.standard_normal(d)
        return sphere.radial_projection(y)

    def accept_reject(self, state):
        """
        Returns boolean indicating if new state can be accepted or should be
        rejected.
        """
        prob = self.target.log_prob(state) - self.target.log_prob(self.state)
        return np.log(self.rng.random()) < prob

    def __next__(self):
        state = self.propose()
        accepted = self.accept_reject(state)
        self.n_accept += int(accepted)
        self.adapt_stepsize(accepted)
        if accepted:
            self.state = state
        return np.copy(self.state)

    def sample(self, n_samples, burnin=0, return_all_samples=False):
        """
        Generates a Markov chain consisting of the desired size. An optional
        burn-in can be specified, either as an integer (in which case it is
        taken as is) or a float smaller than one (in which case it is
        interpreted as percentage of the desired number of samples).
        """
        self.reset(determine_burnin(n_samples, burnin))
        return super().sample(n_samples, burnin, return_all_samples)


class IndependenceSampler(MetropolisHastings):
    def propose(self):
        x = self.rng.standard_normal(len(self.state))
        return sphere.radial_projection(x)


def project(x, v):
    """
    Project x into complement of v.
    """
    return x - v * (v @ x)


class SphericalHMC(MetropolisHastings):
    """
    Spherical Hamiltonian Monte Carlo.
    """

    def __init__(
        self, distribution, initial_state, seed=None, stepsize=1e-3, n_steps=10
    ):
        """
        Parameters
        ----------
        distribution: instance of distributions.Distribution
            Target distribution implementing `log_prob` and `gradient`

        initial_state: numpy ndarray
            Initial state of the Markov chain.

        seed: int, optional
            Seed for the random number generator.

        stepsize: float > 0
            Stepsize of the leapfrog integrator.

        n_steps: integer > 0
            Number of leapfrog integration steps executed in proposal.
        """
        state = np.hstack([initial_state, np.zeros_like(initial_state)])
        super().__init__(distribution, state, seed, stepsize=stepsize)
        self.n_steps = int(n_steps)

    def sample_momenta(self):
        """
        Sample momenta in spherical HMC.
        """
        x, v = np.reshape(self.state, (2, -1))
        v = project(self.rng.standard_normal(len(x)), x)
        self.state = np.hstack([x, v])

    def hamiltonian(self, state):
        """
        Evaluates Hamiltonian, i.e. kinetic energy minus log probability.
        """
        x, v = np.reshape(state, (2, -1))
        return 0.5 * v @ v - self.target.log_prob(x)

    def propose(self):
        """
        Run leapfrog integrator on sphere.
        """
        # log_prob = self.target.log_prob
        gradient = self.target.gradient
        eps = self.stepsize

        self.sample_momenta()

        x, v = np.reshape(np.copy(self.state), (2, -1))

        v += 0.5 * eps * project(gradient(x), x)

        for t in range(self.n_steps):
            norm = np.linalg.norm(v)

            y = np.copy(x)
            x = y * np.cos(eps * norm) + (v / norm) * np.sin(eps * norm)
            v = v * np.cos(eps * norm) - (y * norm) * np.sin(eps * norm)

            if t < self.n_steps - 1:
                v += eps * project(gradient(x), x)
            else:
                v += 0.5 * eps * project(gradient(x), x)

        return np.hstack([sphere.radial_projection(x), v])

    def accept_reject(self, state):
        prob = self.hamiltonian(self.state) - self.hamiltonian(state)
        return np.log(self.rng.random()) < prob

    def sample(
        self,
        n_samples,
        burnin=0,
        return_momenta=False,
        return_all_samples=False,
    ):
        samples = super().sample(n_samples, burnin, return_all_samples)
        positions, momenta = np.swapaxes(
            np.reshape(samples, (len(samples), 2, -1)), 1, 0
        )
        return (positions, momenta) if return_momenta else positions


class RejectionSphericalSliceSampler(Sampler):
    """
    Geodesic Spherical Rejection Slice Sampler
    """

    def __init__(self, distribution, initial_state, seed=None):
        """
        Parameters
        ----------
        distribution: instance of distributions.Distribution
            Target distribution implementing `log_prob` and `gradient`

        initial_state: numpy ndarray
            Initial state of the Markov chain.

        seed: int, optional
            Seed for the random number generator.
        """

        super().__init__(distribution, initial_state, seed)
        self.n_reject = 0

    def __next__(self):
        x = self.state
        p = self.target.log_prob

        # sample subsphere
        u = sphere.spherical_projection(self.rng.standard_normal(len(x)), x)

        threshold = p(x) + np.log(self.rng.random())

        interval = (0, 2 * np.pi)

        while True:
            theta = self.rng.uniform(*interval)
            y = np.cos(theta) * x + np.sin(theta) * u
            if p(y) > threshold:
                self.state = y
                return y
            self.n_reject += 1


class ShrinkageSphericalSliceSampler(RejectionSphericalSliceSampler):
    """
    Geodesic Spherical Shrinkage Slice Sampler
    """

    def __next__(self):
        x = self.state
        p = self.target.log_prob

        # sample subsphere
        u = sphere.spherical_projection(self.rng.standard_normal(len(x)), x)

        threshold = p(x) + np.log(self.rng.random())

        theta = self.rng.uniform(0, 2 * np.pi)
        interval = (theta - 2 * np.pi, theta)

        while True:
            theta = self.rng.uniform(*interval)
            y = np.cos(theta) * x + np.sin(theta) * u
            if p(y) > threshold:
                self.state = y
                return y
            interval = (theta, interval[1]) if theta < 0 else (interval[0], theta)
            self.n_reject += 1
