# 🥬 Vegetable Classification System

Система компьютерного зрения для классификации овощей с использованием глубокого обучения.

## 📋 Содержание

- [Описание](#описание)
- [Функциональность](#функциональность)
- [Установка и запуск](#установка-и-запуск)
- [API Эндпоинты](#api-эндпоинты)
- [Использование](#использование)
- [Docker](#docker)
- [Структура проекта](#структура-проекта)

## 🎯 Описание

Система предназначена для автоматической идентификации 15 видов овощей:
- Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli
- Cabbage, Capsicum, Carrot, Cauliflower, Cucumber
- Papaya, Potato, Pumpkin, Radish, Tomato

## ⚡ Функциональность

- ✅ Классификация изображений (2 модели: Custom CNN и ResNet50V2)
- ✅ Вычисление схожести между изображениями
- ✅ Grad-CAM визуализация для интерпретации решений
- ✅ REST API с документацией Swagger
- ✅ Веб-интерфейс для тестирования
- ✅ Docker контейнеризация

## 🚀 Установка и запуск

### Локальный запуск

### Клонирование репозитория
- git clone 
- cd vegetable-classifier

### Установка зависимостей
-pip install -r requirements.txt

### Docker запуск
### Запуск сервиса
-uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

### Сборка образа
docker build -t vegetable-classifier .

### Запуск контейнера
docker run -p 8000:8000 vegetable-classifier

### Или с docker-compose
docker-compose up --build

# 📡 API Эндпоинты
## GET /ping
### {
###     "status": "alive",
###     "message": "Vegetable Classification Service is running",
###     "device": "cuda",
###     "models_loaded": {"custom_cnn": true, "resnet50v2": true}
## POST /classify
{
    "success": true,
    "model_used": "custom",
    "predicted_class": "Tomato",
    "confidence": 0.95,
    "top_3_predictions": [...],
    "all_probabilities": {...}
}
## POST /similarity
{
    "success": true,
    "similarity_score": 0.87,
    "interpretation": "Highly similar",
    "scale": "(-1 to 1, where 1 = identical)"
}
POST /gradcam
Визуализация Grad-CAM

Параметры:

file: Изображение

target_class: Целевой класс (опционально)

model_type: "custom" или "pretrained"
# 📊 Примеры использования
cURL
bash
## Классификация
curl -X POST "http://localhost:8000/classify?model_type=custom" \
  -F "file=@tomato.jpg"

## Сравнение
curl -X POST "http://localhost:8000/similarity?model_type=custom" \
  -F "file1=@tomato1.jpg" \
  -F "file2=@tomato2.jpg"

## Grad-CAM
curl -X POST "http://localhost:8000/gradcam?model_type=custom" \
  -F "file=@tomato.jpg" \
  -o heatmap.png
Python
python
import requests

## Классификация
with open('tomato.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        params={'model_type': 'custom'},
        files={'file': f}
    )
    print(response.json())
# 🐳 Docker команды
### Сборка образа
docker build -t vegetable-classifier .

### Запуск контейнера
docker run -d -p 8000:8000 --name veggie-api vegetable-classifier

### Просмотр логов
docker logs -f veggie-api

### Остановка контейнера
docker stop veggie-api

### Удаление контейнера
docker rm veggie-api

### Запуск с docker-compose
docker-compose up -d
docker-compose down

# 📁 Структура проекта
text
vegetable-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── models.py            # Архитектуры моделей
│   ├── utils.py             # Утилиты
│   ├── gradcam.py           # Grad-CAM визуализация
│   └── static/
│       └── index.html       # Веб-интерфейс
├── models/
│   ├── custom_cnn_best.pth  # Обученная модель
│   └── resnet50v2_best.pth  # Предобученная модель
├── docker/
│   └── Dockerfile
├── requirements.txt
├── docker-compose.yml
├── .gitignore
└── README.md

# 🔧 Требования
Python 3.9+
CUDA (опционально, для GPU)
Docker (опционально)
4GB RAM минимум

# 📈 Метрики моделей
| Модель          | Accuracy | Precision | Recall | F1-Score | Параметры |
|-----------------|----------|-----------|--------|----------|-----------|
| Custom CNN      | 95,96%   | 0.962     | 0.956  | 0.959    | 0.36M     |
| ResNet50V2      | 99,6%    | 0.996     | 0.996  | 0.996    | 24.69M    |
