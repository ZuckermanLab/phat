"""Distance functions on path space."""
from scipy.spatial.distance import directed_hausdorff


def symmetric_difference_cardinality(s, q):
    """Return the cardinality of the symmetric difference of two sets.

    Parameters
    ----------
    s : iterable
        Elements of the first set. Values must be hashable.
    q : iterable
        Elements of the second set. Values must be hashable.

    Returns
    -------
    int
        ``len(set(s) ^ set(q))``.

    """
    return len(set(s) ^ set(q))


def hausdorff(s, q):
    return max(directed_hausdorff(s, q), directed_hausdorff(q, s))
