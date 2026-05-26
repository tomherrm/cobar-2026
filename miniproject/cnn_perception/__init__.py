"""Heuristic + CNN perception: a CNN segments the fly's raw vision into
object-level classes (grass blade / banana / background); the existing
state-machine controller acts on those features instead of hand-crafted
RGB summaries. Trained by privileged learning — ground-truth segmentation is
rendered from the sim for labels only, never used at deployment.
"""
