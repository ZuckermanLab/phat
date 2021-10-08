import networkx as nx
import numpy as np
import msmtools.analysis as msmana

from ordered_set import OrderedSet


class SurprisalGraph(nx.DiGraph):
    """The surprisal graph of a discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix : (N, N) array_like
        Transition matrix of the Markov chain. Rows must sum to one.
    nodes : (N,) array_like, optional
        List of node (i.e., state) labels. Values must be unique and
        hashable. Default is ``range(N)``.
    symmetrized : bool, default True
        If True, edge weights are round-trip distances. If False, weights
        are one-way distances. See Notes for details.

    Notes
    -----
    The *surprisal graph* of a Markov chain with state space
    :math:`V` and transition matrix :math:`T` is the weighted directed
    graph with vertices :math:`V`, edges

    .. math:: E = \{ (x, y) : T(x, y) > 0 \text{ and } x \ne y \}

    and edge weights

    .. math:: \delta(x, y) = -\log T(x, y).

    The weight :math:`\delta(x, y)` is the
    `information content <https://en.wikipedia.org/wiki/Information_content>`_,
    or *surprisal*, of a one-step transition from :math:`x` to :math:`y`.

    For a reversible Markov chain, the *symmetrized* surprisal graph is
    the graph :math:`(V, E, \delta^*)` with edge weights

    .. math:: \delta^*(x, y) = \delta(x, y) + \delta(y, x).

    """

    def __init__(self, transition_matrix, nodes=None, symmetrized=True):
        matrix = np.asarray(transition_matrix)
        if not msmana.is_transition_matrix(matrix):
            raise ValueError('transition matrix must be row stochastic')

        if nodes is None:
            nodes = OrderedSet(range(matrix.shape[0]))
        else:
            nodes = OrderedSet(nodes)
            if len(nodes) != matrix.shape[0]:
                raise ValueError('number of nodes must match number of states')

        if symmetrized:
            if not msmana.is_reversible(matrix):
                msg = 'Markov chain must be reversible when symmetrized is True'
                raise ValueError(msg)
            matrix *= matrix.T

        np.fill_diagonal(matrix, 0)

        super().__init__()
        for i, j in np.argwhere(matrix > 0):
            x, y = nodes[i], nodes[j]
            self.add_edge(x, y, surprisal=-np.log(matrix[i, j]))

        self._symmetrized = symmetrized

    @property
    def symmetrized(self):
        """bool: Whether edge weights are round-trip distances."""
        return self._symmetrized

    def fundamental_sequence(self, dtraj):
        """Return the fundamental sequence a discrete trajectory.

        Parameters
        ----------
        dtraj : sequence
            A node path in the graph.

        Returns
        -------
        tuple
            The fundamental sequence of `dtraj`.

        """
        return fundamental_sequence(self, dtraj)


def fundamental_sequence(surprisal_graph, dtraj):
    """Return the fundamental sequence of a discrete trajectory.

    Parameters
    ----------
    surprisal_graph : SurprisalGraph
        The surprisal graph of a Markov chain.
    dtraj : sequence
        A node path in `surprisal_graph`.

    Returns
    -------
    tuple
        The fundamental sequence of `dtraj`.

    """
    if not isinstance(surprisal_graph, SurprisalGraph):
        msg = f'surprisal graph must be an instance of {SurprisalGraph}'
        raise TypeError(msg)

    edges = {(x, y) for x, y in nx.utils.pairwise(dtraj) if x != y}
    subgraph = surprisal_graph.edge_subgraph(edges)
    fs = nx.shortest_path(subgraph, source=dtraj[0], target=dtraj[-1],
                          weight='surprisal')
    return tuple(fs)


def loop_erasure(path):
    """Return the loop-erasure of a discrete sample path.

    Parameters
    ----------
    path : sequence
        A sequence of hashable values.

    Returns
    -------
    tuple
        The loop erasure of `path`.

    """
    if len(path) == 0:
        return ()
    last_index = {x: k for k, x in enumerate(path)}
    sel = [0]
    while path[sel[-1]] != path[-1]:
        sel.append(last_index[path[sel[-1]]] + 1)
    return tuple(path[k] for k in sel)
