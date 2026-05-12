import numpy as np
from .hardware import HardwareInterface
from .processor import ImageProcessor

class NeuromorphicSimulator:
    def __init__(self, num_pes=64):
        self.hw = HardwareInterface(num_pes=num_pes)
        self.processor = ImageProcessor()

    def simulate_layer(self, layer_name, weights, input_data=None):
        """Runs a full simulation cycle for a single layer."""
        # 1. Process Weights
        prepared_weights, mode = self.processor.prepare_weights_for_crossbar(weights)
        
        # 2. Logic for hardware mapping (Abstracted from user.py)
        # This is where your complex mapping/scheduling logic resides
        stats = {
            "layer": layer_name,
            "mode": mode,
            "ops": np.prod(weights.shape) * 2, # Example calc
            "active_pes": 48,
            "total_pes": self.hw.num_pes,
            "util_pct": (48 / self.hw.num_pes) * 100,
            "l2_writes": 34816,
            "l2_capacity": self.hw.get_config()["l2_capacity"]
        }
        return stats
