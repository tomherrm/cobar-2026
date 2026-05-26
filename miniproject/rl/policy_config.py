"""Shared PPO policy configuration.

BC warm-start (`train_bc.py`), PPO training, and the from-scratch baseline
(`train.py`) all build their networks from this single source of truth, so
the BC `.zip` and the PPO model are architecturally identical — warm-start
loading is then a plain `PPO.load()` with no state_dict mapping.

`net_arch` is only the MLP head sitting on top of SB3's `CombinedExtractor`
(a `NatureCNN` on the 6-channel vision + a flatten on the 8 scalars). The CNN
does the heavy representational lifting; a small 64x64 head is enough on top.
"""
from __future__ import annotations

POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
