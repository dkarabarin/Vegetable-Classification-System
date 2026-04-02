# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os
import sys
from pathlib import Path

# Добавляем путь для импорта модулей
sys.path.append(str(Path(__file__).parent))

from models import VegetableCNNImproved, ResNet50V2Pretrained
from gradcam import GradCAMVisualizer
from utils import cosine_similarity, load_image, preprocess_image

# Инициализация приложения
app = FastAPI(
    title="Vegetable Classification API",
    description="API для классификации овощей и вычисления схожести изображений",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Классы овощей
CLASS_NAMES = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]
NUM_CLASSES = len(CLASS_NAMES)

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка модели
def load_models():
    """Загрузка обученных моделей"""
    
    # Собственная модель
    custom_model = VegetableCNNImproved(num_classes=NUM_CLASSES)
    model_path = Path(__file__).parent.parent / 'models' / 'custom_cnn_best.pth'
    
    if model_path.exists():
        custom_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Custom model loaded from {model_path}")
    else:
        print(f"⚠️ Custom model not found at {model_path}")
    
    custom_model = custom_model.to(DEVICE)
    custom_model.eval()
    
    # Предобученная модель
    pretrained_model = ResNet50V2Pretrained(num_classes=NUM_CLASSES, dropout_rate=0.5)
    pretrained_path = Path(__file__).parent.parent / 'models' / 'resnet50v2_best.pth'
    
    if pretrained_path.exists():
        pretrained_model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print(f"✅ Pretrained model loaded from {pretrained_path}")
    else:
        print(f"⚠️ Pretrained model not found at {pretrained_path}")
    
    pretrained_model = pretrained_model.to(DEVICE)
    pretrained_model.eval()
    
    return custom_model, pretrained_model

# Загрузка моделей при старте
custom_model, pretrained_model = load_models()

# Инициализация Grad-CAM
gradcam_visualizer = GradCAMVisualizer(custom_model, target_layer='conv4')

# ==========================================
# ЭНДПОИНТЫ
# ==========================================

@app.get("/ping")
async def ping():
    """
    Проверка статуса сервиса
    """
    return {
        "status": "alive",
        "message": "Vegetable Classification Service is running",
        "device": str(DEVICE),
        "models_loaded": {
            "custom_cnn": custom_model is not None,
            "resnet50v2": pretrained_model is not None
        }
    }

@app.get("/health")
async def health():
    """
    Детальная проверка здоровья сервиса
    """
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "custom_model_params": sum(p.numel() for p in custom_model.parameters()) / 1e6,
        "pretrained_model_params": sum(p.numel() for p in pretrained_model.parameters()) / 1e6
    }

@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    model_type: str = "custom"
):
    """
    Классификация изображения
    
    Args:
        file: Изображение (JPEG, PNG)
        model_type: Тип модели ("custom" или "pretrained")
    
    Returns:
        Предсказанный класс и вероятности
    """
    # Проверка формата файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Загрузка и предобработка
        contents = await file.read()
        image = load_image(contents)
        input_tensor = preprocess_image(image, transform).to(DEVICE)
        
        # Выбор модели
        model = custom_model if model_type == "custom" else pretrained_model
        
        # Инференс
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = float(confidence.item())
        
        # Все вероятности
        all_probabilities = {
            CLASS_NAMES[i]: float(probabilities[0][i]) 
            for i in range(NUM_CLASSES)
        }
        
        # Топ-3 предсказания
        top3_idx = torch.topk(probabilities, 3, dim=1)[1][0]
        top3_predictions = [
            {"class": CLASS_NAMES[idx.item()], "probability": float(probabilities[0][idx])}
            for idx in top3_idx
        ]
        
        return JSONResponse(content={
            "success": True,
            "model_used": model_type,
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "top_3_predictions": top3_predictions,
            "all_probabilities": all_probabilities
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/similarity")
async def similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    model_type: str = "custom"
):
    """
    Вычисление схожести между двумя изображениями
    
    Args:
        file1: Первое изображение
        file2: Второе изображение
        model_type: Тип модели ("custom" или "pretrained")
    
    Returns:
        Значение схожести от -1 до 1
    """
    # Проверка форматов файлов
    for file in [file1, file2]:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
    
    try:
        # Загрузка изображений
        contents1 = await file1.read()
        contents2 = await file2.read()
        
        image1 = load_image(contents1)
        image2 = load_image(contents2)
        
        tensor1 = preprocess_image(image1, transform).to(DEVICE)
        tensor2 = preprocess_image(image2, transform).to(DEVICE)
        
        # Выбор модели
        model = custom_model if model_type == "custom" else pretrained_model
        
        # Вычисление схожести
        similarity_score = cosine_similarity(model, tensor1, tensor2, DEVICE)
        
        # Интерпретация
        if similarity_score > 0.7:
            interpretation = "Highly similar - likely the same vegetable"
        elif similarity_score > 0.3:
            interpretation = "Moderately similar - may be related vegetables"
        else:
            interpretation = "Dissimilar - different vegetables"
        
        return JSONResponse(content={
            "success": True,
            "model_used": model_type,
            "similarity_score": similarity_score,
            "interpretation": interpretation,
            "scale": "(-1 to 1, where 1 = identical)"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    target_class: Optional[str] = None,
    model_type: str = "custom"
):
    """
    Визуализация Grad-CAM для интерпретации решения модели
    
    Args:
        file: Изображение
        target_class: Целевой класс (опционально)
        model_type: Тип модели ("custom" или "pretrained")
    
    Returns:
        Heatmap в формате base64
    """
    import base64
    from PIL import ImageDraw
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Загрузка изображения
        contents = await file.read()
        image = load_image(contents)
        input_tensor = preprocess_image(image, transform).to(DEVICE)
        
        # Выбор модели
        model = custom_model if model_type == "custom" else pretrained_model
        
        # Получение предсказания
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, 1).item()
        
        # Определение целевого класса
        if target_class:
            target_idx = CLASS_NAMES.index(target_class)
        else:
            target_idx = predicted_class_idx
        
        # Генерация Grad-CAM
        heatmap = gradcam_visualizer.generate_heatmap(input_tensor, target_idx)
        
        # Наложение heatmap на изображение
        overlay = gradcam_visualizer.overlay_heatmap(image, heatmap)
        
        # Конвертация в base64
        buffered = io.BytesIO()
        overlay.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse(content={
            "success": True,
            "model_used": model_type,
            "predicted_class": CLASS_NAMES[predicted_class_idx],
            "target_class": CLASS_NAMES[target_idx],
            "confidence": float(probabilities[0][predicted_class_idx]),
            "heatmap_base64": img_str
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid target class: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM: {str(e)}")

# ==========================================
# СТАТИЧЕСКИЕ ФАЙЛЫ (ФРОНТЕНД)
# ==========================================

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с интерфейсом"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Vegetable Classification API</h1><p>Visit /docs for API documentation</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)