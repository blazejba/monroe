import torch

from monroe.mtl.abstract_weighting import AbsWeighting
from monroe.mtl.DWA import DWA
from monroe.mtl.EW import EW
from monroe.mtl.RLW import RLW
from monroe.mtl.STCH import STCH
from monroe.mtl.UW import UW

__all__ = ["AbsWeighting", "DWA", "EW", "UW", "STCH", "RLW", "build_mtl_model_class"]

# Registry of weighting strategies
WEIGHTING_STRATEGIES = {
    "DWA": DWA,
    "EW": EW,
    "UW": UW,
    "STCH": STCH,
    "RLW": RLW,
}


def build_mtl_model_class(weighting_name: str, n_shards: int):
    """Create MTL model class with the specified weighting strategy.

    Args:
        weighting_name: Name of the weighting strategy (EW, UW, STCH, RLW).
        n_shards: Number of training shards (for loss buffer sizing).

    Returns:
        A class that inherits from the specified weighting strategy.

    Raises:
        ValueError: If weighting_name is not recognized.
    """
    if weighting_name not in WEIGHTING_STRATEGIES:
        raise ValueError(
            f"Unknown weighting strategy '{weighting_name}'. "
            f"Available: {list(WEIGHTING_STRATEGIES.keys())}"
        )

    base_class = WEIGHTING_STRATEGIES[weighting_name]

    class MTLModel(base_class):
        def __init__(self, task_types, encoder_cls, decoders, device,
                     dataset_heads=None, pcba_assay_slices=None):
            super().__init__(
                task_types, encoder_cls, decoders, device,
                dataset_heads=dataset_heads,
                pcba_assay_slices=pcba_assay_slices,
            )
            self.encoder = encoder_cls()
            self.decoders = decoders
            self.shard = 0
            self.init_param()
            self.train_loss_buffer = torch.zeros(
                (self.task_num, n_shards), dtype=torch.float32
            )

    return MTLModel
