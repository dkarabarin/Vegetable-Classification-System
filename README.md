# 🥬 Vegetable Classification System

Система компьютерного зрения для классификации 15 видов овощей с использованием глубокого обучения.

## 🎯 Описание

Система автоматически определяет следующие виды овощей:
- Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli
- Cabbage, Capsicum, Carrot, Cauliflower, Cucumber
- Papaya, Potato, Pumpkin, Radish, Tomato

## ⚡ Функциональность

- ✅ Классификация изображений (Custom CNN и ResNet50V2)
- ✅ Вычисление схожести между изображениями
- ✅ Grad-CAM визуализация
- ✅ REST API + Swagger документация
- ✅ Веб-интерфейс для тестирования
- ✅ Docker контейнеризация

## 🚀 Установка и запуск

### 1. Клонирование репозитория

### Клонирование репозитория
- git clone 
- cd vegetable-classifier

### Установка зависимостей
-pip install -r requirements.txt

### Запуск сервера
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 📡 API Эндпоинты

📍 Доступные эндпоинты:

- Swagger документация: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Веб-интерфейс: http://localhost:8000
- Health Check: http://localhost:8000/health
- Ping: http://localhost:8000/ping

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

- **app/**
  - `__init__.py`
  - `main.py` (FastAPI приложение)
  - `models.py` (архитектуры моделей)
  - `utils.py` (утилиты)
  - `gradcam.py` (Grad-CAM визуализация)
  - **static/**
    - `index.html` (веб-интерфейс)
- **models/**
  - `custom_cnn_best.pth` (обученная модель)
  - `resnet50v2_best.pth` (предобученная модель)
- **docker/**
  - `Dockerfile`
- `requirements.txt`
- `docker-compose.yml`
- `.gitignore`
- `README.md`

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

# Внешний вид и пример использования

![start](https://github.com/user-attachments/assets/6d8c018a-dc95-46d5-a744-a0752dd27a91)



