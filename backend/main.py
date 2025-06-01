from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="A simple API to predict wine quality based on chemical properties",
    version="1.0.0"
)

# Load the trained model, scaler, and feature names
try:
    with open('../ml/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('../ml/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('../ml/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
        
    print("Model loaded successfully!")
    print(f"Expected features: {feature_names}")
    
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Make sure you've run the training notebook first!")

# Define the input data model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    
    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }

# Define the response model
class PredictionResponse(BaseModel):
    predicted_quality: float
    quality_category: str
    confidence: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Wine Quality Prediction API",
        "description": "Send POST request to /predict with wine features",
        "documentation": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_wine_quality(wine_data: WineFeatures):
    try:
        # Convert input to numpy array
        features = np.array([[
            wine_data.fixed_acidity,
            wine_data.volatile_acidity,
            wine_data.citric_acid,
            wine_data.residual_sugar,
            wine_data.chlorides,
            wine_data.free_sulfur_dioxide,
            wine_data.total_sulfur_dioxide,
            wine_data.density,
            wine_data.pH,
            wine_data.sulphates,
            wine_data.alcohol
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Round to nearest 0.1
        prediction = round(prediction, 1)
        
        # Determine quality category
        if prediction <= 4:
            category = "Poor"
        elif prediction <= 5:
            category = "Fair"
        elif prediction <= 6:
            category = "Good"
        elif prediction <= 7:
            category = "Very Good"
        else:
            category = "Excellent"
        
        # Simple confidence estimation based on prediction value
        if prediction in [3, 4, 5, 6, 7, 8]:
            confidence = "High"
        else:
            confidence = "Medium"
        
        return PredictionResponse(
            predicted_quality=prediction,
            quality_category=category,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict_batch")
async def predict_batch(wine_samples: List[WineFeatures]):
    try:
        predictions = []
        
        for wine_data in wine_samples:
            # Convert input to numpy array
            features = np.array([[
                wine_data.fixed_acidity,
                wine_data.volatile_acidity,
                wine_data.citric_acid,
                wine_data.residual_sugar,
                wine_data.chlorides,
                wine_data.free_sulfur_dioxide,
                wine_data.total_sulfur_dioxide,
                wine_data.density,
                wine_data.pH,
                wine_data.sulphates,
                wine_data.alcohol
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = round(model.predict(features_scaled)[0], 1)
            
            # Determine category
            if prediction <= 4:
                category = "Poor"
            elif prediction <= 5:
                category = "Fair"
            elif prediction <= 6:
                category = "Good"
            elif prediction <= 7:
                category = "Very Good"
            else:
                category = "Excellent"
            
            predictions.append({
                "predicted_quality": prediction,
                "quality_category": category
            })
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Get model info
@app.get("/model_info")
async def get_model_info():
    return {
        "model_type": "Random Forest Regressor",
        "features": feature_names,
        "target": "wine_quality",
        "quality_range": "3-9 (higher is better)",
        "description": "Predicts wine quality based on chemical properties"
    }

if __name__ == "__main__":
    uvicorn.run(app,  port=8000)