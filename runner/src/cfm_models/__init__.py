"""Backward-compatible re-exports.

The CFM classes have been split into individual modules:
    - cfm_base.py: CFMBase, make_warmup_scheduler
    - wgcfm.py: WGCFM
    - geodesic_cfm.py: GeodesicCFM
    - tangent_cfm.py: Tangent_CFM
    - euclidean_cfm.py: EuclideanCFM

Import from the specific module for new code.
"""

from cfm_models.cfm_base import CFMBase  # noqa: F401

# from cfm_models.wgcfm import WGCFM  # noqa: F401
# from cfm_models.geodesic_cfm import GeodesicCFM  # noqa: F401
# from cfm_models.tangent_cfm import Tangent_CFM  # noqa: F401
from cfm_models.euclidean_cfm import EuclideanCFM  # noqa: F401
