# app/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class GradCAMVisualizer:
    """Grad-CAM визуализатор для интерпретации решений модели"""
    
    def __init__(self, model, target_layer='conv4'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Регистрируем хуки
        self._register_hooks()
    
    def _register_hooks(self):
        """Регистрация хуков для получения градиентов и активаций"""
        def save_gradient(module, grad_input, grad_output):
            # grad_output содержит градиенты по отношению к выходу модуля
            self.gradients = grad_output[0]
        
        def save_activation(module, input, output):
            self.activations = output
        
        # Находим целевой слой
        found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer or (hasattr(module, '_get_name') and module._get_name() == self.target_layer):
                module.register_forward_hook(save_activation)
                # Используем register_full_backward_hook для получения градиентов
                module.register_full_backward_hook(save_gradient)
                print(f"✅ Hook registered on layer: {name}")
                found = True
                break
        
        if not found:
            print(f"⚠️ Warning: Layer '{self.target_layer}' not found. Available layers:")
            for name, _ in self.model.named_modules():
                print(f"  - {name}")
    
    def generate_heatmap(self, input_tensor, target_class):
        """Генерация heatmap для заданного класса"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target class score
        target_score = output[0, target_class]
        
        # Backward pass
        target_score.backward(retain_graph=False)
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations. Check if hooks are properly registered.")
        
        gradients = self.gradients[0].cpu().data.numpy()  # [C, H, W]
        activations = self.activations[0].cpu().data.numpy()  # [C, H, W]
        
        # Weighted average
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU activation
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """Наложение heatmap на изображение"""
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        
        # Convert to RGB colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert image to numpy
        img_np = np.array(image.convert('RGB'))
        
        # Overlay
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlay)
    
    def generate_and_save(self, input_tensor, target_class, original_image, save_path='gradcam_output.png'):
        """Генерация и сохранение Grad-CAM визуализации"""
        heatmap = self.generate_heatmap(input_tensor, target_class)
        overlay = self.overlay_heatmap(original_image, heatmap)
        overlay.save(save_path)
        print(f"✅ Grad-CAM saved to {save_path}")
        return overlay