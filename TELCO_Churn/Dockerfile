# Dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения и обученных артефактов
# Убедитесь, что эти файлы находятся в той же директории, что и Dockerfile
COPY api_churn.py .
COPY xgb_churn_model.joblib .
COPY preprocessor_churn.joblib .
COPY churn_feature_names.joblib .

# Установка переменных окружения для путей к моделям
# Каждая переменная ENV теперь на отдельной строке для максимальной совместимости
ENV MODEL_PATH=/app/xgb_churn_model.joblib
ENV PREPROCESSOR_PATH=/app/preprocessor_churn.joblib
ENV FEATURE_NAMES_PATH=/app/churn_feature_names.joblib

# Запуск приложения Uvicorn (рекомендуется для продакшена)
CMD ["uvicorn", "api_churn:app", "--host", "0.0.0.0", "--port", "8000"]



