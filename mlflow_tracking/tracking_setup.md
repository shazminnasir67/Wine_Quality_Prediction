# MLflow Experiment Tracking Setup

## Overview
This guide explains how to set up and use MLflow for tracking your wine quality prediction model experiments.

## Installation
```bash
pip install mlflow
```

## Starting MLflow UI

1. **Open Terminal/Command Prompt** in your project directory

2. **Start MLflow UI Server:**
```bash
mlflow ui
```

3. **Access MLflow UI:**
   - Open your browser and go to: `http://localhost:5000`
   - You should see the MLflow tracking interface

## What Gets Tracked

Our training script automatically logs:

### Parameters
- `n_estimators`: Number of trees in Random Forest (100)
- `max_depth`: Maximum depth of trees (10)
- `random_state`: Random seed for reproducibility (42)
- `model_type`: Algorithm used (RandomForestRegressor)
- `feature_scaling`: Preprocessing method (StandardScaler)

### Metrics
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error  
- `mae`: Mean Absolute Error
- `r2_score`: R-squared coefficient
- `training_time`: Time taken to train model (seconds)

### Model Artifacts
- The trained model is saved and registered as "WineQualityPredictor"

## Viewing Experiments

1. **Experiment List**: See all experiments on the main page
2. **Run Comparison**: Click on "wine_quality_prediction" experiment
3. **Individual Runs**: Click on any run to see detailed metrics and parameters
4. **Model Registry**: View registered models in the "Models" tab

## Key MLflow UI Sections

### 1. Experiments Page
- Lists all your ML experiments
- Shows run status, metrics summary
- **Screenshot Location**: Take screenshot of this page

### 2. Run Details Page  
- Detailed view of individual training runs
- Shows all parameters, metrics, and artifacts
- **Screenshot Location**: Take screenshot showing metrics like R² and RMSE

### 3. Model Registry
- Centralized model store
- Version management for deployed models
- **Screenshot Location**: Take screenshot of registered "WineQualityPredictor"

## Screenshots Required for Assignment

Take the following screenshots and save them in this folder:

1. **`mlflow_experiments_overview.png`**: Main experiments page showing wine_quality_prediction experiment
2. **`mlflow_run_details.png`**: Individual run showing parameters and metrics
3. **`mlflow_model_registry.png`**: Model registry showing WineQualityPredictor model

## Sample MLflow Commands

```bash
# Start MLflow UI (run in project root)
mlflow ui --port 5000

# View specific experiment
mlflow experiments list

# Compare runs programmatically
mlflow runs list --experiment-id 1
```

## Troubleshooting

### Common Issues:

1. **Port 5000 already in use:**
   ```bash
   mlflow ui --port 5001
   ```

2. **No experiments showing:**
   - Make sure you ran the training notebook first
   - Check if `mlruns` folder exists in project directory

3. **Model not appearing in registry:**
   - Ensure model registration code ran successfully in training script
   - Check for any error messages during training

## Integration with Training Script

The training notebook includes these MLflow tracking calls:

```python
import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("wine_quality_prediction")

# Start run and log everything
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    
    # Log metrics  
    mlflow.log_metric("r2_score", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "wine_quality_model")
```

## Next Steps

1. Run the training notebook to generate experiment data
2. Start MLflow UI with `mlflow ui`  
3. Take required screenshots
4. Compare different model configurations by running training with different parameters

## Benefits of MLflow Tracking

- **Reproducibility**: Track exact parameters used for each model
- **Comparison**: Compare different model versions easily
- **Deployment**: Deploy models directly from MLflow registry
- **Collaboration**: Share experiment results with team members