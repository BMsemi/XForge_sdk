# XForge SDK 🧠💻

XForge SDK is a specialized Python library for simulating the execution of Deep Learning models (like YOLO) on **Neuromorphic X2** hardware. It provides tools for image preprocessing, neural network weight binarization, and hardware performance benchmarking.

---

## 🚀 Features

- **YOLO Integration:** Extract weights directly from YOLOv8 models for hardware analysis.
- **Crossbar Mapping:** Automatically format weights into Signed or Unsigned 64x64/64x32 crossbar grids.
- **Hardware Simulation:** Benchmarks PE (Processing Element) utilization, operation counts, and L2 buffer writes.
- **Image Processing:** Standardized RGB channel splitting and resizing for neuromorphic inputs.

---

## 🛠 Installation

To install the SDK in **Editable Mode** (recommended for developers), run the following in your terminal:

```bash
cd XForge_sdk
pip install -e .
```

### Install from GitHub (Public Repository)

If your GitHub repository is public, anyone can install it using this single command:

```bash
pip install git+https://github.com/username/XForge_sdk.git
```

### Dependencies

The SDK automatically installs:

- `numpy`
- `torch` (PyTorch)
- `ultralytics` (YOLO)
- `Pillow`
- `opencv-python`

---

## 📖 Quick Start

### 1. Simple Layer Simulation

Test how a specific set of weights performs on your neuromorphic hardware.

```python
from XForge import NeuromorphicSimulator
import numpy as np

# Initialize simulator with 64 PEs
sim = NeuromorphicSimulator(num_pes=64)

# Create dummy Conv2D weights (Out_CH, In_CH, H, W)
weights = np.random.randn(32, 3, 3, 3)

# Run simulation
stats = sim.simulate_layer(layer_name="conv1", weights=weights)

print(f"Active PEs: {stats['active_pes']}")
print(f"Utilization: {stats['util_pct']}%")
```

### 2. YOLO Weight Extraction

Analyze a real-world model layer-by-layer.

```python
from XForge import YOLOWrapper, NeuromorphicSimulator

# Load YOLOv8
yolo = YOLOWrapper('yolov8n.pt')

# Extract weights from the first layer
layer_0_weights = yolo.get_layer_weights(0)

# Simulate hardware execution
sim = NeuromorphicSimulator()
results = sim.simulate_layer("YOLO_Layer_0", layer_0_weights)

print(f"Ops processed: {results['ops']}")
```

---

## 📂 Project Structure

```plaintext
XForge_Project/          <-- Root
├── setup.py
├── README.md                # Documentation for other developers
├── test_sdk.py              # Test code
├── XForge/              <-- Internal package renamed
    ├── __init__.py          # Main entry point for the SDK
    ├── simulator.py         # Hardware simulation engine (Refactored from user.py)
    ├── processor.py         # Image & Weight preprocessing utilities
    ├── hardware.py          # Interface for Neuromorphic chip components
    └── models/
        └── yolo_wrapper.py  # YOLO integration for detection

```

---

## 🧪 Development & Testing

To run the internal test suite or your custom test scripts:

```bash
python test_sdk.py
```

If using the Streamlit Dashboard:

```bash
streamlit run app.py
```

---

## 🤝 Contributing

For internal use by **BM Labs**. Please ensure all hardware-specific logic is validated against the NeuromorphicX2 core modules before pushing updates.
