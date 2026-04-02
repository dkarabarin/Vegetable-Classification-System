# app/utils.py
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np

def load_image(contents):
    """Загрузка изображения из bytes"""
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    return image

def preprocess_image(image, transform):
    """Предобработка изображения для модели"""
    return transform(image).unsqueeze(0)

def cosine_similarity(model, tensor1, tensor2, device):
    """Вычисление косинусного сходства между двумя изображениями"""
    model.eval()
    
    with torch.no_grad():
        # Получаем фичи из предпоследнего слоя
        if hasattr(model, 'classifier'):
            # Для собственной модели
            features1 = model.conv1(tensor1)
            features1 = model.conv2(features1)
            features1 = model.conv3(features1)
            features1 = model.conv4(features1)
            features1 = model.classifier[0](features1)  # AdaptiveAvgPool2d
            features1 = features1.view(features1.size(0), -1)
            
            features2 = model.conv1(tensor2)
            features2 = model.conv2(features2)
            features2 = model.conv3(features2)
            features2 = model.conv4(features2)
            features2 = model.classifier[0](features2)
            features2 = features2.view(features2.size(0), -1)
        else:
            # Для ResNet
            features1 = model.backbone(tensor1)
            features2 = model.backbone(tensor2)
            features1 = features1.view(features1.size(0), -1)
            features2 = features2.view(features2.size(0), -1)
        
        # Cosine similarity
        similarity = F.cosine_similarity(features1, features2, dim=1)
        
        return float(similarity[0])

def convert_to_onnx(model, input_shape, output_path):
    """Конвертация модели в ONNX формат"""
    model.eval()
    dummy_input = torch.randn(1, 3, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Model converted to ONNX: {output_path}")