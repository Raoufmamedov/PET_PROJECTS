from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Инициализация FastAPI приложения
app = FastAPI(
    title="Модель Прогнозирования Оттока Клиентов",
    description="API для прогнозирования оттока клиентов на основе их характеристик."
)

# Загрузка обученных моделей и препроцессоров
# Убедитесь, что пути соответствуют месту хранения в Docker-образе
MODEL_PATH = os.getenv("MODEL_PATH", "/app/xgb_churn_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "/app/preprocessor_churn.joblib")
FEATURE_NAMES_PATH = os.getenv("FEATURE_NAMES_PATH", "/app/churn_feature_names.joblib")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    # feature_names = joblib.load(FEATURE_NAMES_PATH) 
    print("Модель и препроцессор успешно загружены.")
except Exception as e:
    print(f"Ошибка загрузки модели или препроцессора: {e}")
    model = None
    preprocessor = None

# Схема входных данных для API
class ChurnFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict_churn/")
async def predict_churn(features: ChurnFeatures):
    if model is None or preprocessor is None:
        return {"error": "Модель или препроцессор не загружены."}

    # Преобразование входных данных в DataFrame
    input_df = pd.DataFrame([features.dict()])

    try:
        # Применяем препроцессор
        processed_features = preprocessor.transform(input_df)

        # Получаем предсказания
        churn_probability = model.predict_proba(processed_features)[:, 1][0]
        
        # churn_prediction = int(model.predict(processed_features)[0])
        churn_prediction = (churn_probability > 0.43).astype(int)


        return {
            "churn_probability": float(churn_probability),
            "churn_prediction": churn_prediction,
            "message": "Успешное предсказание."
        }
    except Exception as e:
        return {"error": f"Ошибка при предсказании: {str(e)}"}
