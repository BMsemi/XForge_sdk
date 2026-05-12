import numpy as np
from PIL import Image

class ImageProcessor:
    @staticmethod
    def load_rgb_image(img_path, size=(640, 640)):
        """Loads and resizes image for neuromorphic processing."""
        if hasattr(img_path, 'read'): # Handling uploaded files (e.g. Streamlit)
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')
            
        img = img.resize(size)
        img_array = np.array(img)
        
        # Split channels for hardware mapping
        channels = (img_array[:,:,0], img_array[:,:,1], img_array[:,:,2])
        return img_array, channels

    @staticmethod
    def prepare_weights_for_crossbar(conv_weights, target_size=64, mode="auto"):
        """Converts Conv2D weights to crossbar-compatible format."""
        out_ch, in_ch, H, W = conv_weights.shape
        flattened = conv_weights.reshape(out_ch, in_ch * H * W)
        transposed = flattened.T
        in_features, out_features = transposed.shape
        
        has_negative = np.any(transposed < 0)
        actual_mode = "signed" if (mode == "auto" and has_negative) else (mode if mode != "auto" else "unsigned")
        
        if actual_mode == "signed":
            if not has_negative: transposed = 2 * transposed - 1
            if out_features > target_size // 2: transposed = transposed[:, :target_size // 2]
            
            crossbar_weights = np.zeros((target_size, target_size // 2))
            crossbar_weights[:min(in_features, target_size), :min(out_features, target_size // 2)] = \
                transposed[:min(in_features, target_size), :min(out_features, target_size // 2)]
        else:
            if has_negative: transposed = (transposed + 1) / 2
            if out_features > target_size: transposed = transposed[:, :target_size]
            
            crossbar_weights = np.zeros((target_size, target_size))
            crossbar_weights[:min(in_features, target_size), :min(out_features, target_size)] = \
                transposed[:min(in_features, target_size), :min(out_features, target_size)]
        
        return crossbar_weights, actual_mode
