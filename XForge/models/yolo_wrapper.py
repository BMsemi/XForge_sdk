from ultralytics import YOLO
import torch

class YOLOWrapper:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    # Make sure THIS function is added:
    def predict(self, source):
        """Runs standard YOLO inference on an image or array."""
        return self.model(source)

    def get_layer_weights(self, layer_index):
        """Extracts weights from a specific layer for simulation."""
        try:
            layer = self.model.model.model[layer_index]
            if hasattr(layer, 'conv'):
                return layer.conv.weight.detach().cpu().numpy()
            return layer.weight.detach().cpu().numpy()
        except Exception as e:
            print(f"Error extracting weights: {e}")
            return None