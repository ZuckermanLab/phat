import collections
import itertools
import types

from typing import Callable


class WeightedSample:
    """A sequence of (observation, weight) pairs.

    Parameters
    ----------
    observations : iterable, default ()
        A sequence of observations.
    weights : iterable of floats, optional
        The weight of each observation. By default, each observation
        has weight 1. Weights are paired with observations in the
        order they are iterated.

    """

    def __init__(self, observations=None, weights=None):
        observations = [] if observations is None else list(observations)

        if weights is None:
            weights = [1] * len(observations)
        else:
            weights = list(weights)
            if len(weights) != len(observations):
                raise ValueError(
                    'number of weights must match number of observations')
            if any(w < 0 for w in weights):
                raise ValueError('weights must be non-negative')

        self._observations = observations
        self._weights = weights

    def append(self, observation, weight=1):
        if weight < 0:
            raise ValueError('weight must be non-negative')
        self._observations.append(observation)
        self._weights.append(weight)

    @property
    def observations(self):
        """iterator: Iterator over the observations."""
        return iter(self._observations)

    @property
    def weights(self):
        """iterator: Iterator over the weights."""
        return iter(self._weights)

    @property
    def total_weight(self):
        """float: Sum of the weights."""
        return sum(self._weights)

    def __len__(self):
        return len(self._observations)

    def __iter__(self):
        return zip(self._observations, self._weights)

    def __getitem__(self, key):
        if type(key) is int:
            return self._observations[key], self._weights[key]
        return WeightedSample(self._observations[key], self._weights[key])

    def __bool__(self):
        return len(self) > 0

    def __add__(self, other):
        return WeightedSample(
            itertools.chain(self.observations, other.observations),
            itertools.chain(self.weights, other.weights))

    def __concat__(self, other):
        return self + other

    def __rmul__(self, coeff):
        coeff = float(coeff)
        if coeff < 0:
            raise ValueError('scale factor must be non-negative')
        return WeightedSample(self.observations,
                              (coeff * w for w in self.weights))

    def __imul__(self, coeff):
        return coeff * self

    def __repr__(self):
        return (f'<{self.__class__.__name__} {hex(id(self))}, '
                + f'total_weight={self.total_weight}>')


class PathwayHistogram:
    """A pathway histogram.

    Parameters
    ----------
    classifier : callable
        A function that takes a single trajectory as input and returns a
        hashable value representing the pathway class of the trajectory.

    """

    def __init__(self, classifier):
        if not isinstance(classifier, Callable):
            raise TypeError('classifier must be callable')
        self._classifier = classifier
        self._data = collections.defaultdict(WeightedSample)

    @property
    def classifier(self):
        """callable: Mapping from trajectories to pathway classes."""
        return self._classifier

    @property
    def data(self):
        """types.MappingProxyType: Read-only view of histogram data."""
        return types.MappingProxyType(self._data)

    def classes(self):
        """Iterable[Hashable]: Pathway classes (block labels).

        Alias self.data.keys().

        """
        return self._data.keys()

    def blocks(self):
        """Iterable[WeightedSample]: Data belonging to each class.

        Alias self.data.values().

        """
        return self._data.values()

    def add(self, trajectory, weight=1):
        """Add a trajectory to the histogram.

        Parameters
        ----------
        trajectory : object
            Trajectory to be classified.
        weight : float, default 1.0
            The weight of the trajectory.

        """
        pathway_class = self.classifier(trajectory)
        self._data[pathway_class].append(trajectory, weight)

    def fill(self, trajectories, weights=None, accumulate=True):
        """Fill the histogram with data.

        Parameters
        ----------
        trajectories : iterable
            Trajectories to be classified.
        weights : iterable of float, optional
            The weight of each trajectory. By default, each trajectory
            has weight 1. Weights are paired with trajectories in the
            order they are iterated.
        accumulate : bool, default True
            If True, retain any existing data in the histogram. If False,
            clear the histogram before filling with the given data.

        """
        if not accumulate:
            self.clear()

        if weights is None:
            for trajectory in trajectories:
                self.add(trajectory)
        else:
            for trajectory, weight in zip(trajectories, weights):
                self.add(trajectory, weight)

    def clear(self):
        """Remove all data from the histogram."""
        self._data.clear()


def pathway_histogram(classifier, trajectories, weights=None):
    """Construct a pathway histogram.

    Parameters
    ----------
    classifier : PathwayClassifier
        A function that maps a trajectory to its pathway class.
    trajectories : iterable
        Trajectories to be classified.
    weights : iterable of float, optional
        The weight of each trajectory. By default, each trajectory
        has weight 1. Weights are paired with trajectories in the
        order they are iterated.

    Returns
    -------
    PathwayHistogram
        A pathway histogram of the given data.

    """
    return PathwayHistogram(classifier).fill(trajectories, weights=weights)
