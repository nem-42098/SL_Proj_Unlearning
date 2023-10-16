from typing import Type
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Layer:
    """Keep information about a layer"""
    kind: Type
    in_channels: int
    out_channels: int
    idx: int = 0
    is_bias: bool = False
    is_running_mean: bool = False
    is_running_var: bool = False
    is_num_batches_tracked: bool = False

    def __post_init__(self):
        assert self.is_bias + self.is_running_mean + \
            self.is_running_var + self.is_num_batches_tracked <= 1
        # assert self.kind in ['Conv2d', 'ConvT2d', 'BatchNorm2D', 'Linear']

    def replace(self, **kw):
        return replace(self, **kw)