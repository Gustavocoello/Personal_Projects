from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import fastparquet
import joblib

# 1. App
app = FastAPI(title="Energy Prediction API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de CO2"}

    
#2. Cargar modelo, encoders y estadisticas
model = joblib.load("model/model_pki.pkl")
le_country = joblib.load("model/label_encoder_ct.joblib") # Encoder de Country
le_energy = joblib.load("model/label_encoder_ty.joblib") # Encoder de Energy_type
stats = pd.read_parquet("model/country_stats.parquet", engine="fastparquet")

# 3. Esquemas de entrada
class PredictionResquest(BaseModel):
    country: str
    energy_type: str
    year: int
    
# 4. Funciones para manejar categorias desconocidas
def safe_encode(encoder, value, default_category):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return encoder.transform([default_category])[0]
    
# 5. Ruta de prediccion
@app.post("/predict")
def predict(request: PredictionResquest):
    try:
        # ===== Obtener estadísticas del país =====
        country_stats = stats[stats['Country'] == request.country]
        
        if country_stats.empty:
            # Usar promedio global si el país no existe
            country_stats = stats.mean(numeric_only=True).to_frame().T
        else:
            country_stats = country_stats.iloc[0]

        # ===== Codificar variables categóricas =====
        country_encoded = safe_encode(
            le_country, 
            request.country, 
            default_category="United States"  # País por defecto
        )
        
        energy_encoded = safe_encode(
            le_energy,
            request.energy_type,
            default_category="coal"  # Tipo de energía por defecto
        )

        # ===== Calcular variables derivadas =====
        population = country_stats['Population']
        energy_consumption = country_stats['Energy_consumption'] * (population / 1e6)
        gdp = country_stats['GDP'] * (population / 1e6)
        
        # ===== Crear DataFrame para predicción =====
        input_data = pd.DataFrame([{
            'Year'                       : request.year,
            'Country_encoded'            : country_encoded,
            'Energy_type_encoded'        : energy_encoded,
            'Energy_consumption'         : energy_consumption,
            'Energy_production'          : country_stats['Energy_production'],
            'GDP'                        : gdp,
            'Population'                 : population,
            'Energy_intensity_per_capita': country_stats['Energy_intensity_per_capita'],
            'Energy_intensity_by_GDP'    : country_stats['Energy_intensity_by_GDP']
        }])
        
         # ===== Depuración: Verificar columnas y valores =====
        print("Columnas esperadas por el modelo:", model.feature_names_in_)
        print("Columnas en input_data:", input_data.columns.tolist())
        print("Valores en input_data:\n", input_data)
        print("Forma de input_data:", input_data.shape)
        print("Tipos de datos en input_data:\n", input_data.dtypes)
        
        # ===== Ordenar columnas como el modelo espera =====
        input_data = input_data[model.feature_names_in_]

        # ===== Hacer predicción =====
        prediction = model.predict(input_data)[0]
        
        return {
            "country": request.country,
            "year": request.year,
            "predicted_co2": round(prediction, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Endpoint para listar países válidos
@app.get("/valid_countries")
def get_valid_countries():
    return {"countries": le_country.classes_.tolist()}

# 7. Endpoint para tipos de energía válidos
@app.get("/valid_energy_types")
def get_valid_energy_types():
    return {"energy_types": le_energy.classes_.tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)