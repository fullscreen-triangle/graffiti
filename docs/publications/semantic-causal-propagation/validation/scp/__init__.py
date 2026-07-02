"""Semantic Causal Propagation (SCP) validation package.

Self-contained implementation of the finite-weighted-graph calculus of
``semantic-causal-propagation.tex``: contact graphs, the resolution floor,
individuation by negation, representation mobility, path opacity,
catalytic composition, coherence, closure, and scheduler soundness.

No external dependency beyond the Python standard library is required;
maximum flow / minimum cut is computed with a from-scratch exact
Edmonds-Karp implementation (see ``scp.maxflow``).
"""
