import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import requests
# from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
# from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather Analysis API with Ollama", version="1.0.0")
origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://10.42.0.243:3000",
    "http://10.0.0.243:3000"
]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Pydantic models
class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    parameter: str

class WeatherAnalysisResponse(BaseModel):
    parameter: str
    llm_analysis: str
    statistical_analysis: Dict[str, Any]
    numerical_predictions: Dict[str, Any]
    visualization: str  # base64 encoded image
    metadata: Dict[str, Any]

class WeatherDataResponse(BaseModel):
    current_conditions: Dict[str, Any]
    hourly_data: Dict[str, List[Any]]
    daily_data: Dict[str, List[Any]]

# Initialize Ollama LLM
try:
    # llm = Ollama(model="phi3:latest", base_url="http://localhost:11434")
    llm = ChatOllama(
            model="phi3:latest",
            num_keep=-1,
            seed=42,
            num_predict=-1,
            top_k=20,
            top_p=0.9,
            num_thread=2,
            num_gpu=-1
        )
except Exception as e:
    logger.warning(f"Warning: Could not initialize Ollama: {e}")
    llm = None

# Weather parameter configurations
WEATHER_PARAMETERS = {
    "temperature": {
        "name": "Temperature 2m",
        "unit": "°C",
        "description": "Air temperature at 2 meters above ground"
    },
    "humidity": {
        "name": "Relative Humidity 2m", 
        "unit": "%",
        "description": "Relative humidity at 2 meters above ground"
    },
    "wind_speed": {
        "name": "Wind Speed 10m",
        "unit": "km/h", 
        "description": "Wind speed at 10 meters above ground"
    },
    "precipitation": {
        "name": "Precipitation",
        "unit": "mm",
        "description": "Accumulated precipitation"
    },
    "pressure": {
        "name": "Pressure MSL",
        "unit": "hPa",
        "description": "Mean sea level pressure"
    },
    "cloud_cover": {
        "name": "Cloud Cover",
        "unit": "%",
        "description": "Total cloud cover percentage"
    }
}

class WeatherAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self, weather_data: Dict, parameter: str) -> pd.DataFrame:
        """Prepare weather data for analysis"""
        try:
            hourly_data = weather_data.get('hourly', {})
            
            # Map parameter names to data keys
            param_mapping = {
                'temperature': 'temperature_2m',
                'humidity': 'relative_humidity_2m', 
                'wind_speed': 'wind_speed_10m',
                'precipitation': 'precipitation',
                'pressure': 'pressure_msl',
                'cloud_cover': 'cloud_cover'
            }
            
            data_key = param_mapping.get(parameter, parameter)
            
            if data_key not in hourly_data:
                raise ValueError(f"Parameter {data_key} not found in weather data")
            
            df = pd.DataFrame({
                'time': pd.to_datetime(hourly_data['time']),
                'value': hourly_data[data_key],
                'temperature': hourly_data.get('temperature_2m', [0] * len(hourly_data['time'])),
                'humidity': hourly_data.get('relative_humidity_2m', [0] * len(hourly_data['time'])),
                'wind_speed': hourly_data.get('wind_speed_10m', [0] * len(hourly_data['time'])),
                'pressure': hourly_data.get('pressure_msl', [0] * len(hourly_data['time']))
            })
            
            # Add time-based features
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error preparing data: {str(e)}")
    
    def statistical_analysis(self, df: pd.DataFrame, parameter: str) -> Dict[str, Any]:
        """Perform statistical analysis on weather parameter"""
        try:
            values = df['value'].dropna()
            
            analysis = {
                "basic_stats": {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "range": float(np.max(values) - np.min(values))
                },
                "trend_analysis": {
                    "correlation_with_time": float(np.corrcoef(range(len(values)), values)[0, 1]) if len(values) > 1 else 0,
                    "is_increasing": bool(np.polyfit(range(len(values)), values, 1)[0] > 0) if len(values) > 1 else False
                },
                "variability": {
                    "coefficient_of_variation": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
                    "percentiles": {
                        "25th": float(np.percentile(values, 25)),
                        "75th": float(np.percentile(values, 75)),
                        "90th": float(np.percentile(values, 90))
                    }
                }
            }
            
            # Cross-correlations with other parameters
            correlations = {}
            for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
                if col != parameter and col in df.columns:
                    corr_values = df[col].dropna()
                    if len(corr_values) == len(values) and len(values) > 1:
                        correlations[col] = float(np.corrcoef(values, corr_values)[0, 1])
            
            analysis["cross_correlations"] = correlations
            
            return analysis
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def numerical_weather_prediction(self, df: pd.DataFrame, parameter: str) -> Dict[str, Any]:
        """Implement NWP-style prediction using ML"""
        try:
            if len(df) < 5:
                return {"error": "Insufficient data for prediction"}
            
            # Prepare features and targets
            features = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'day_of_week', 'month']
            available_features = [f for f in features if f in df.columns]
            
            X = df[available_features].fillna(df[available_features].mean())
            y = df['value'].fillna(df['value'].mean())
            
            if len(X) < 3:
                return {"error": "Insufficient data for model training"}
            
            # Create lagged features for time series prediction
            for i in range(1, min(4, len(df))):
                if i < len(y):
                    X[f'lag_{i}'] = y.shift(i).fillna(method='bfill')
            
            # Split data (use last few points for validation)
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            else:
                X_train, X_test, y_train, y_test = X[:-1], X[-1:], y[:-1], y[-1:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions) if len(y_test) > 0 else 0
            rmse = np.sqrt(mean_squared_error(y_test, predictions)) if len(y_test) > 0 else 0
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Future predictions (next 6 hours)
            last_row = X.iloc[-1:].copy()
            future_predictions = []
            
            for i in range(6):
                pred = model.predict(scaler.transform(last_row))[0]
                future_predictions.append(float(pred))
                
                # Update lagged features for next prediction
                for lag in range(1, 4):
                    if f'lag_{lag}' in last_row.columns:
                        if lag == 1:
                            last_row[f'lag_{lag}'] = pred
                        else:
                            last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].iloc[0]
            
            return {
                "model_performance": {
                    "mean_absolute_error": float(mae),
                    "root_mean_squared_error": float(rmse),
                    "prediction_accuracy": float(max(0, 1 - mae / (y.std() if y.std() > 0 else 1)))
                },
                "feature_importance": {k: float(v) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]},
                "future_predictions": {
                    "next_6_hours": future_predictions,
                    "timestamps": [(datetime.now() + timedelta(hours=i+1)).isoformat() for i in range(6)]
                },
                "model_info": {
                    "algorithm": "Random Forest Regressor",
                    "features_used": available_features,
                    "training_samples": len(X_train)
                }
            }
            
        except Exception as e:
            logger.error(f"NWP prediction failed: {str(e)}")
            return {"error": f"NWP prediction failed: {str(e)}"}
    
    def create_visualization(self, df: pd.DataFrame, parameter: str, analysis: Dict) -> str:
        """Create visualization for weather parameter"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{WEATHER_PARAMETERS[parameter]["name"]} Analysis', fontsize=16)
            
            # Time series plot
            axes[0, 0].plot(df['time'], df['value'], 'b-', linewidth=2, alpha=0.7)
            axes[0, 0].set_title('Time Series')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel(f'{WEATHER_PARAMETERS[parameter]["name"]} ({WEATHER_PARAMETERS[parameter]["unit"]})')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Distribution histogram
            axes[0, 1].hist(df['value'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Value Distribution')
            axes[0, 1].set_xlabel(f'{WEATHER_PARAMETERS[parameter]["name"]} ({WEATHER_PARAMETERS[parameter]["unit"]})')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Correlation heatmap (if we have other parameters)
            corr_data = df[['value', 'temperature', 'humidity', 'wind_speed', 'pressure']].corr()
            im = axes[1, 0].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 0].set_title('Parameter Correlations')
            axes[1, 0].set_xticks(range(len(corr_data.columns)))
            axes[1, 0].set_yticks(range(len(corr_data.columns)))
            axes[1, 0].set_xticklabels(corr_data.columns, rotation=45)
            axes[1, 0].set_yticklabels(corr_data.columns)
            plt.colorbar(im, ax=axes[1, 0])
            
            # Box plot by hour of day
            if 'hour' in df.columns and len(df) > 10:
                hourly_data = df.groupby('hour')['value'].apply(list)
                axes[1, 1].boxplot([hourly_data[hour] if hour in hourly_data else [] for hour in range(24)], 
                                 positions=range(24))
                axes[1, 1].set_title('Hourly Distribution')
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel(f'{WEATHER_PARAMETERS[parameter]["name"]} ({WEATHER_PARAMETERS[parameter]["unit"]})')
            else:
                # Simple statistics text if not enough data
                stats_text = f"Mean: {analysis['statistical_analysis']['basic_stats']['mean']:.2f}\n"
                stats_text += f"Std: {analysis['statistical_analysis']['basic_stats']['std']:.2f}\n"
                stats_text += f"Min: {analysis['statistical_analysis']['basic_stats']['min']:.2f}\n"
                stats_text += f"Max: {analysis['statistical_analysis']['basic_stats']['max']:.2f}"
                axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Summary Statistics')
            
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            # Return a simple error plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{WEATHER_PARAMETERS[parameter]["name"]} - Visualization Error')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            logger.error(f'{WEATHER_PARAMETERS[parameter]["name"]} - Visualization Error | {str(e)}')
            return image_base64
    
    def generate_llm_analysis(self, df: pd.DataFrame, parameter: str, statistical_analysis: Dict, 
                             numerical_predictions: Dict) -> str:
        """Generate LLM analysis using Ollama"""
        try:
            if llm is None:
                logger.warning("LLM analysis unavailable - Ollama not connected")
                return "LLM analysis unavailable - Ollama not connected"
            
            # Prepare context for LLM
            current_value = df['value'].iloc[-1] if len(df) > 0 else "N/A"
            param_info = WEATHER_PARAMETERS[parameter]
            
            # Create comprehensive prompt
            prompt_template = PromptTemplate(
                input_variables=["parameter", "current_value", "unit", "description", "stats", "predictions"],
                template="""
                As a meteorological expert, analyze the following weather parameter data:
                
                Parameter: {parameter}
                Current Value: {current_value} {unit}
                Description: {description}
                
                Statistical Analysis Summary:
                - Mean: {stats_mean}
                - Standard Deviation: {stats_std}
                - Trend: {trend}
                
                Model Predictions:
                - Next 6 hours: {predictions}
                - Model Accuracy: {accuracy}
                
                Please provide a comprehensive analysis including:
                1. Current conditions assessment
                2. Short-term trends and patterns
                3. Weather implications and significance
                4. Any notable observations or concerns
                5. Confidence in predictions
                
                Keep the analysis professional, concise, and practical for weather forecasting.
                """
            )
            
            # Prepare prompt variables
            stats = statistical_analysis.get('basic_stats', {})
            trend_info = "increasing" if statistical_analysis.get('trend_analysis', {}).get('is_increasing', False) else "stable/decreasing"
            
            predictions_summary = "No predictions available"
            accuracy = "N/A"
            if 'future_predictions' in numerical_predictions:
                pred_values = numerical_predictions['future_predictions']['next_6_hours'][:3]  # First 3 hours
                predictions_summary = f"{pred_values}"
                accuracy = f"{numerical_predictions.get('model_performance', {}).get('prediction_accuracy', 0):.2%}"
            
            # Create and run chain
            # chain = LLMChain(llm=llm, prompt=prompt_template)
            chain = prompt_template | llm
            
            input_data = {
                "parameter": param_info['name'],
                "current_value": current_value,
                "unit": param_info['unit'],
                "description": param_info['description'],
                "stats_mean": stats.get('mean', 'N/A'),
                "stats_std": stats.get('std', 'N/A'),
                "trend": trend_info,
                "predictions": predictions_summary,
                "accuracy": accuracy
            }
            analysis = chain.invoke(input_data)
            
            return analysis.content.strip()
            
        except Exception as e:
            logger.error(f"LLM analysis error: {str(e)}. Using fallback analysis: The {parameter} parameter shows current conditions with available statistical patterns. Manual interpretation of the data suggests normal variability within expected ranges for this meteorological parameter.")
            
            return f"LLM analysis error: {str(e)}. Using fallback analysis: The {parameter} parameter shows current conditions with available statistical patterns. Manual interpretation of the data suggests normal variability within expected ranges for this meteorological parameter."


import oracledb
import json
from datetime import datetime


async def insert_temperature_analysis(json_data: dict):
    # Connect to Oracle
    connection = oracledb.connect(
        user="SYS",
        password="oracle",
        dsn="10.42.0.243:1521/FREE",
        mode=oracledb.SYSDBA
    )
    cursor = connection.cursor()

    # Helper to get nested fields safely
    def safe_get(d, keys, default=None):
        try:
            for k in keys:
                d = d[k]
            return d
        except Exception:
            return default

    try:
        stats = safe_get(json_data, ["statistical_analysis", "basic_stats"], {})
        trend = safe_get(json_data, ["statistical_analysis", "trend_analysis"], {})
        variability = safe_get(json_data, ["statistical_analysis", "variability"], {})
        cross = safe_get(json_data, ["statistical_analysis", "cross_correlations"], {})
        perf = safe_get(json_data, ["numerical_predictions", "model_performance"], {})
        fi = safe_get(json_data, ["numerical_predictions", "feature_importance"], {})
        preds = safe_get(json_data, ["numerical_predictions", "future_predictions"], {})
        model_info = safe_get(json_data, ["numerical_predictions", "model_info"], {})
        meta = safe_get(json_data, ["metadata"], {})

        sql = """INSERT INTO analyze_weather_states (
            parameter, llm_analysis,
            mean_value, median_value, std_dev, min_value, max_value, range_value,
            correlation_with_time, is_increasing,
            coeff_of_variation, percentile_25, percentile_75, percentile_90,
            corr_humidity, corr_wind_speed, corr_pressure,
            mae, rmse, prediction_accuracy,
            fi_pressure, fi_temperature, fi_lag_1, fi_lag_2, fi_wind_speed,
            pred_hour1, pred_hour2, pred_hour3, pred_hour4, pred_hour5, pred_hour6,
            ts_hour1, ts_hour2, ts_hour3, ts_hour4, ts_hour5, ts_hour6,
            algorithm, features_used, training_samples,
            analysis_timestamp, latitude, longitude, data_points, parameter_unit, methods_used,
            visualization
        ) VALUES (
            :parameter, :llm_analysis,
            :mean_value, :median_value, :std_dev, :min_value, :max_value, :range_value,
            :correlation_with_time, :is_increasing,
            :coeff_of_variation, :percentile_25, :percentile_75, :percentile_90,
            :corr_humidity, :corr_wind_speed, :corr_pressure,
            :mae, :rmse, :prediction_accuracy,
            :fi_pressure, :fi_temperature, :fi_lag_1, :fi_lag_2, :fi_wind_speed,
            :pred_hour1, :pred_hour2, :pred_hour3, :pred_hour4, :pred_hour5, :pred_hour6,
            :ts_hour1, :ts_hour2, :ts_hour3, :ts_hour4, :ts_hour5, :ts_hour6,
            :algorithm, :features_used, :training_samples,
            :analysis_timestamp, :latitude, :longitude, :data_points, :parameter_unit, :methods_used,
            :visualization
        )"""

        values = {
            "parameter": json_data.get("parameter"),
            "llm_analysis": json_data.get("llm_analysis"),
            "mean_value": stats.get("mean"),
            "median_value": stats.get("median"),
            "std_dev": stats.get("std"),
            "min_value": stats.get("min"),
            "max_value": stats.get("max"),
            "range_value": stats.get("range"),
            "correlation_with_time": trend.get("correlation_with_time"),
            "is_increasing": "Y" if trend.get("is_increasing") else "N",
            "coeff_of_variation": variability.get("coefficient_of_variation"),
            "percentile_25": safe_get(variability, ["percentiles", "25th"]),
            "percentile_75": safe_get(variability, ["percentiles", "75th"]),
            "percentile_90": safe_get(variability, ["percentiles", "90th"]),
            "corr_humidity": cross.get("humidity"),
            "corr_wind_speed": cross.get("wind_speed"),
            "corr_pressure": cross.get("pressure"),
            "mae": perf.get("mean_absolute_error"),
            "rmse": perf.get("root_mean_squared_error"),
            "prediction_accuracy": perf.get("prediction_accuracy"),
            "fi_pressure": fi.get("pressure"),
            "fi_temperature": fi.get("temperature"),
            "fi_lag_1": fi.get("lag_1"),
            "fi_lag_2": fi.get("lag_2"),
            "fi_wind_speed": fi.get("wind_speed"),
            "pred_hour1": safe_get(preds, ["next_6_hours", 0]),
            "pred_hour2": safe_get(preds, ["next_6_hours", 1]),
            "pred_hour3": safe_get(preds, ["next_6_hours", 2]),
            "pred_hour4": safe_get(preds, ["next_6_hours", 3]),
            "pred_hour5": safe_get(preds, ["next_6_hours", 4]),
            "pred_hour6": safe_get(preds, ["next_6_hours", 5]),
            "ts_hour1": safe_get(preds, ["timestamps", 0]),
            "ts_hour2": safe_get(preds, ["timestamps", 1]),
            "ts_hour3": safe_get(preds, ["timestamps", 2]),
            "ts_hour4": safe_get(preds, ["timestamps", 3]),
            "ts_hour5": safe_get(preds, ["timestamps", 4]),
            "ts_hour6": safe_get(preds, ["timestamps", 5]),
            "algorithm": model_info.get("algorithm"),
            "features_used": json.dumps(model_info.get("features_used", [])),
            "training_samples": model_info.get("training_samples"),
            "analysis_timestamp": meta.get("analysis_timestamp"),
            "latitude": safe_get(meta, ["location", "latitude"]),
            "longitude": safe_get(meta, ["location", "longitude"]),
            "data_points": meta.get("data_points"),
            "parameter_unit": meta.get("parameter_unit"),
            "methods_used": json.dumps(meta.get("methods_used", [])),
            "visualization": json_data.get("visualization"),
        }

        # Convert ISO timestamps safely
        for key in ["ts_hour1","ts_hour2","ts_hour3","ts_hour4","ts_hour5","ts_hour6","analysis_timestamp"]:
            if values[key]:
                try:
                    values[key] = datetime.fromisoformat(values[key])
                except Exception:
                    values[key] = None

        cursor.execute(sql, values)
        connection.commit()

    except Exception as e:
        print(f"❌ Insert failed: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()
        


analyzer = WeatherAnalyzer()
from WeatherDataService import WeatherDataService

def transform_hourly(hourly: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = hourly.get("time", [])
    n = len(times)
    records = []
    for i in range(n):
        rec = {"time": times[i]}
        for key, vals in hourly.items():
            if key == "time":
                continue
            rec[key] = vals[i] if i < len(vals) else None
        records.append(rec)
    return records

def transform_daily(daily: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = daily.get("time", [])
    n = len(times)
    records = []
    for i in range(n):
        rec = {"time": times[i]}
        for key, vals in daily.items():
            if key == "time":
                continue
            rec[key] = vals[i] if i < len(vals) else None
        records.append(rec)
    return records

# Mock weather data endpoint (replace with your actual weather API)
@app.post("/weather/current", response_model=WeatherDataResponse)
async def get_current_weather(request: WeatherRequest):
    try:
        weather_data = await WeatherDataService.fetch_weather_data(
            request.latitude, request.longitude
        )
        if not isinstance(weather_data, dict):
            logger.error(f"Expected dict from fetch_weather_data, got {type(weather_data)}: {weather_data}")
            raise ValueError(f"Invalid weather data type: {type(weather_data)}")
        
        return WeatherDataResponse(
            current_conditions=weather_data["current"],
            hourly_data=weather_data["hourly"],
            daily_data=weather_data["daily"]
        )
    except Exception as e:
        logger.error(f"Weather data retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weather data retrieval failed: {str(e)}")

class SectionResult(BaseModel):
    tag: str  # e.g. "basic_stats"
    sentence: str
    generated_text: str
    data: dict


class WeatherResult(BaseModel):
    latitude: float
    longitude: float
    basic_stats: Optional[SectionResult] = None
    trend_analysis: Optional[SectionResult] = None
    variability: Optional[SectionResult] = None
    cross_correlations: Optional[SectionResult] = None
    model_performance: Optional[SectionResult] = None
    feature_importance: Optional[SectionResult] = None
    future_predictions: Optional[SectionResult] = None

from config.model_options import ModelOptions
import httpx
from text_generation.text_generation_v2 import TextGeneration

class WeatherTextGenerator:
    def __init__(self, text_gen_url: str, latitude: float, longitude: float):
        self.text_gen_url = text_gen_url
        self.latitude = latitude
        self.longitude = longitude
        # self.analysis_url = analysis_url
        self.connection = oracledb.connect(
            user="SYS",
            password="oracle",
            dsn="10.42.0.243:1521/FREE",
            mode=oracledb.SYSDBA
        )
        self.locations = self.select_locations_by_latitude_longitude()
        self.city = self.locations[0]["city"]
        self.state = self.locations[0]["state"]
        self.county = self.locations[0]["county"]
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.directive = f"Meteorology inputs: \n\n"
        self.text_gen = TextGeneration()
    
    # -------- safe formatter that only applies :.2f when the value is numeric. 
    def _fmt(self, value, decimals=2, suffix=""):
        """Safely format a number to X decimals, or return 'N/A' if not numeric"""
        try:
            if value is None:
                return "N/A"
            return f"{float(value):.{decimals}f}{suffix}"
        except (ValueError, TypeError):
            return "N/A"

    # -------- Select Locations --------
    def select_locations_by_latitude_longitude(self):
        """Filter US Zips by Latitude & Longitude. Get Locations"""
        try:
            cursor = self.connection.cursor()
            
            # SQL Query to filter table
            cursor.execute(
                f"""
                SELECT
                    *
                FROM us_zips 
                WHERE latitude = '{self.latitude}' AND longitude = '{self.longitude}'                
                """
            )
            rows = cursor.fetchall()  # Fetch all rows efficiently
            results = []
            for row in rows:
                results.append({
                    "zip": row[0],
                    "city": row[1],
                    "state": row[2],
                    "state_abbr": row[3],
                    "county": row[4],
                    "count_code": row[5],
                    "latitude": row[6],
                    "longitude": row[7]
                })
            return results
        except Exception as e:
            logger.warning(f"Exception error: {e}")
            return []
        except oracledb.Error as e:
            logger.warning(f"Database exception error: {e}")
            return []
    
    
    async def call_ollama_model(self, model_name: str, system_prompt: str, user_prompt: str):
        """Call Ollama API with the specified model"""
        start_time = datetime.now()
        MODEL_OPTIONS = ModelOptions().MODEL_OPTIONS
        # Default to safe fallback if model not in dict
        options = MODEL_OPTIONS.get(model_name, {
            "num_keep": -1,
            "seed": 42,
            "num_predict": -1,
            "top_k": 30,
            "top_p": 0.9,
            "temperature": 0.3,
            "num_thread": 2,
            "num_gpu": -1
        })

        payload = {
            "model": model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": options
        }

        try:
            async with httpx.AsyncClient(timeout=1500) as client:
                response = await client.post(
                    f"{self.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                )
                response.raise_for_status()

                end_time = datetime.now()
                response_time_ms = (end_time - start_time).total_seconds() * 1000

                result = response.json()
                return result.get("response", ""), response_time_ms

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request to Ollama timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    def build_basic_stats_sentences(self, stats: Dict[str, Any]) -> Dict[str, str]:
        base_sentence = (
            f"The mean temperature is {self._fmt(stats.get('mean'))}°C with a median of {self._fmt(stats.get('median'))}°C. "
            f"The standard deviation is {self._fmt(stats.get('std'))}, ranging from {self._fmt(stats.get('min'))}°C "
            f"to {self._fmt(stats.get('max'))}°C."
        )
        generation_result = self.text_gen.text_generation(base_sentence, "basic_stats", stats)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": stats
        }

    def build_trend_analysis_sentences(self, trend: Dict[str, Any]) -> Dict[str, str]:
        trend_text = "increasing" if trend.get("is_increasing") else "not increasing"
        base_sentence = (
            f"The correlation with time is {self._fmt(trend.get('correlation_with_time'))}, "
            f"indicating that the temperature trend is {trend_text}."
        )
        generation_result = self.text_gen.text_generation(base_sentence, "trend_analysis", trend)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": trend
        }

    def build_variability_sentences(self, var: Dict[str, Any]) -> Dict[str, str]:
        base_sentence = (
            f"The coefficient of variation is {self._fmt(var.get('coefficient_of_variation'))}. "
            f"Temperatures are spread between the 25th percentile {self._fmt(var['percentiles'].get('25th'))}°C "
            f"and the 75th percentile {self._fmt(var['percentiles'].get('75th'))}°C, "
            f"with a 90th percentile reaching {self._fmt(var['percentiles'].get('90th'))}°C."
        )
        generation_result = self.text_gen.text_generation(base_sentence, "variability", var)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": var
        }

    def build_cross_correlation_sentences(self, corr: Dict[str, Any]) -> Dict[str, str]:
        base_sentence = (
            f"Temperature correlates {self._fmt(corr.get('humidity'))} with humidity, "
            f"{self._fmt(corr.get('wind_speed'))} with wind speed, "
            f"and {self._fmt(corr.get('pressure'))} with pressure."
        )
        generation_result = self.text_gen.text_generation(base_sentence, "cross_correlations", corr)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": corr
        }

    def build_model_performance_sentences(self, perf: Dict[str, Any]) -> Dict[str, str]:
        base_sentence = (
            f"The prediction model has a mean absolute error of {self._fmt(perf.get('mean_absolute_error'))}, "
            f"a root mean squared error of {self._fmt(perf.get('root_mean_squared_error'))}, "
            f"and an accuracy of {self._fmt(perf.get('prediction_accuracy'))}."
        )
        generation_result = self.text_gen.text_generation(base_sentence, "model_performance", perf)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": perf
        }

    def build_feature_importance_sentences(self, feats: Dict[str, Any]) -> Dict[str, str]:
        ordered_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{k} ({v:.2f})" for k, v in ordered_feats]
        base_sentence = "The most important features influencing the model are: " + ", ".join(parts) + "."
        generation_result = self.text_gen.text_generation(base_sentence, "feature_importance", feats)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": feats
        }

    def build_future_predictions_sentences(self, preds: Dict[str, Any]) -> Dict[str, str]:
        temps = preds.get("next_6_hours", [])
        times = preds.get("timestamps", [])
        # Format timestamps more readably
        formatted_pairs = []
        for i, (t, v) in enumerate(zip(times, temps)):
            # Extract just the time part for readability
            time_part = t.split('T')[1][:5] if 'T' in t else t
            formatted_pairs.append(f"{time_part} → {v:.1f}°C")
        
        base_sentence = "Future temperature predictions over the next 6 hours are: " + "; ".join(formatted_pairs) + "."
        generation_result = self.text_gen.text_generation(base_sentence, "future_predictions", preds)
        return {
            "sentence": base_sentence,
            "generated_text": generation_result["generated_text"],
            "data": preds
        }

    # -------- API Calls --------
    def send_to_text_generation(self, sentence: str) -> Dict[str, Any]:
        """Send a sentence to the /text_generation endpoint and return its response."""
        payload = {"text": sentence}
        response = requests.post(self.text_gen_url, json=payload)
        response.raise_for_status()
        return response.json()

    
    def insert_weather_result(self, final_result: dict):
        conn = oracledb.connect(
            user="SYS",
            password="oracle",
            dsn="10.42.0.243:1521/FREE",
            mode=oracledb.SYSDBA
        )
        # Extract lat/lon
        lat = final_result["temperature_analysis"]["metadata"]["location"]["latitude"]
        lon = final_result["temperature_analysis"]["metadata"]["location"]["longitude"]

        # Helper to extract safely
        def extract(section):
            if section in final_result.get("statistical_analysis", {}):
                sec = final_result["statistical_analysis"][section]
            elif section in final_result.get("numerical_predictions", {}):
                sec = final_result["numerical_predictions"][section]
            else:
                return None, None
            logger.info(f"Extracting section {section}: {sec}")
            sentence = sec.get("sentence")
            text_generation = sec.get("text_generation")
            return sentence, text_generation
        
        values = {
            "lat": lat,
            "lon": lon,
            "basic_stats_sentence": extract("basic_stats")[0],
            "basic_stats_response": extract("basic_stats")[1],
            "trend_analysis_sentence": extract("trend_analysis")[0],
            "trend_analysis_response": extract("trend_analysis")[1],
            "variability_sentence": extract("variability")[0],
            "variability_response": extract("variability")[1],
            "cross_corr_sentence": extract("cross_correlations")[0],
            "cross_corr_response": extract("cross_correlations")[1],
            "model_perf_sentence": extract("model_performance")[0],
            "model_perf_response": extract("model_performance")[1],
            "feat_imp_sentence": extract("feature_importance")[0],
            "feat_imp_response": extract("feature_importance")[1],
            "future_pred_sentence": extract("future_predictions")[0],
            "future_pred_response": extract("future_predictions")[1],
        }

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO WEATHER_ANALYSIS (
                LATITUDE, LONGITUDE,
                BASIC_STATS_SENTENCE, BASIC_STATS_RESPONSE,
                TREND_ANALYSIS_SENTENCE, TREND_ANALYSIS_RESPONSE,
                VARIABILITY_SENTENCE, VARIABILITY_RESPONSE,
                CROSS_CORR_SENTENCE, CROSS_CORR_RESPONSE,
                MODEL_PERF_SENTENCE, MODEL_PERF_RESPONSE,
                FEATURE_IMPORTANCE_SENTENCE, FEATURE_IMPORTANCE_RESPONSE,
                FUTURE_PRED_SENTENCE, FUTURE_PRED_RESPONSE
            ) VALUES (
                :lat, :lon,
                :basic_stats_sentence, :basic_stats_response,
                :trend_analysis_sentence, :trend_analysis_response,
                :variability_sentence, :variability_response,
                :cross_corr_sentence, :cross_corr_response,
                :model_perf_sentence, :model_perf_response,
                :feat_imp_sentence, :feat_imp_response,
                :future_pred_sentence, :future_pred_response
            )
        """, values)

        conn.commit()
    def serialize(self, final_result: Dict[str, Any]) -> WeatherResult:
        lat = self.latitude
        lon = self.longitude

        def build_section(tag, section_dict):
            if section_dict:
                return SectionResult(
                    tag=tag,
                    sentence=section_dict.get("sentence"),
                    generated_text=section_dict.get("text_generation", {}).get("response"),
                    data=section_dict.get("data", {}),
                )
            return None

        return WeatherResult(
            latitude=lat,
            longitude=lon,
            basic_stats=build_section("basic_stats", final_result["statistical_analysis"].get("basic_stats")),
            trend_analysis=build_section("trend_analysis", final_result["statistical_analysis"].get("trend_analysis")),
            variability=build_section("variability", final_result["statistical_analysis"].get("variability")),
            cross_correlations=build_section("cross_correlations", final_result["statistical_analysis"].get("cross_correlations")),
            model_performance=build_section("model_performance", final_result["numerical_predictions"].get("model_performance")),
            feature_importance=build_section("feature_importance", final_result["numerical_predictions"].get("feature_importance")),
            future_predictions=build_section("future_predictions", final_result["numerical_predictions"].get("future_predictions")),
        )
    # -------- Processing --------
    async def process_weather(self, analysis: dict, lat: float, lon: float, parameter: str) -> Dict[str, Any]:
        """Process weather data section by section and nest text generation results."""
        # latitude = weather_json["metadata"]["location"]["latitude"]
        # longitude = weather_json["metadata"]["location"]["longitude"]

        # Get fresh analysis
        # analysis = self.get_temperature_analysis(latitude, longitude)
        # async def call_ollama_model(self, model_name: str, system_prompt: str, user_prompt: str):
        logger.info(analysis)
        
        results = {
            "statistical_analysis": {},
            "numerical_predictions": {}
        }
        # --- statistical_analysis ---
        sa = analysis.get("statistical_analysis", {})

        if "basic_stats" in sa:
            s = self.build_basic_stats_sentences(sa["basic_stats"])
            results["statistical_analysis"]["basic_stats"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": sa["basic_stats"]
            }

        if "trend_analysis" in sa:
            s = self.build_trend_analysis_sentences(sa["trend_analysis"])
            results["statistical_analysis"]["trend_analysis"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": sa["trend_analysis"]
            }

        if "variability" in sa:
            s = self.build_variability_sentences(sa["variability"])
            results["statistical_analysis"]["variability"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": sa["variability"]
            }

        if "cross_correlations" in sa:
            s = self.build_cross_correlation_sentences(sa["cross_correlations"])
            results["statistical_analysis"]["cross_correlations"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": sa["cross_correlations"]
            }

        # --- numerical_predictions ---
        npred = analysis.get("numerical_predictions", {})

        if "model_performance" in npred:
            s = self.build_model_performance_sentences(npred["model_performance"])
            results["numerical_predictions"]["model_performance"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": npred["model_performance"]
            }

        if "feature_importance" in npred:
            s = self.build_feature_importance_sentences(npred["feature_importance"])
            results["numerical_predictions"]["feature_importance"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": npred["feature_importance"]
            }

        if "future_predictions" in npred:
            s = self.build_future_predictions_sentences(npred["future_predictions"])
            results["numerical_predictions"]["future_predictions"] = {
                "sentence": s["sentence"],
                "text_generation": s["generated_text"],
                "data": npred["future_predictions"]
            }

        # Attach global analysis
        results["temperature_analysis"] = analysis

        logger.info(f"Results before insert_weather_result: {results}")
        with open('/home/labrigui/Software/microservices/python-software/conversationalai/apis/weather/repos/cache/weather_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        WeatherVisualizer(latitude=lat, longitude=lon, parameter=parameter).insert_weather_visualization()

        self.insert_weather_result(results)
        return results


from text_generation.selects import Selects
from fastapi.responses import JSONResponse
from WeatherVisualization import WeatherVisualizer
@app.post("/weather/information", response_model=WeatherResult)
async def information_weather_temperature(request: WeatherRequest):
    try:
        if request.parameter == "temperature":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "temperature")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)

            
            
            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "temperature")                
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )
        if request.parameter == "humidity":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "humidity")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)            

            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "humidity")                   
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )
        if request.parameter == "wind_speed":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "wind_speed")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)            

            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "wind_speed")   
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )
        if request.parameter == "precipitation":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "precipitation")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)


            
            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "precipitation")   
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )
        if request.parameter == "pressure":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "pressure")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)
                    

            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "pressure")   
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )
        if request.parameter == "cloud_cover":
            text_gen_url = "http://10.42.0.243:8004/text_generation"
            analyzer = WeatherTextGenerator(
                text_gen_url=text_gen_url,
                latitude=request.latitude,
                longitude=request.longitude
            )

            result = await analyze_parameter(request, "cloud_cover")

            if not isinstance(result, WeatherAnalysisResponse):
                logger.error(f"Expected WeatherAnalysisResponse, got {type(result)}: {result}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid analysis result type"}
                )

            result_dict = result.dict()
            if not isinstance(result_dict, dict):
                logger.error(f"Expected dict from result.dict(), got {type(result_dict)}: {result_dict}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": "Invalid result_dict type"}
                )

            await insert_temperature_analysis(result_dict)
            
            
            try:
                final_result = await analyzer.process_weather(result_dict, request.latitude, request.longitude, "cloud_cover")                   
                return JSONResponse(status_code=200, content=final_result)
            except Exception as e:
                logger.error(f"Unexpected error during weather processing: {str(e)}")
                return JSONResponse(
                    status_code=200,
                    content={"status": "error", "message": f"Weather processing failed: {str(e)}"}
                )

        # If parameter is not supported
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": f"Unsupported parameter: {request.parameter}"}
        )

    except Exception as e:
        logger.error(f"Error in informative insights generation: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": f"Generation failed: {str(e)}"}
        )

# Weather parameter analysis endpoints
@app.post("/analyze/temperature", response_model=WeatherAnalysisResponse)
async def analyze_temperature(request: WeatherRequest):
    """Analyze temperature data with LLM insights, statistical analysis, and ML predictions"""
    result = await analyze_parameter(request, "temperature")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result

@app.post("/analyze/humidity", response_model=WeatherAnalysisResponse) 
async def analyze_humidity(request: WeatherRequest):
    """Analyze humidity data with comprehensive analysis"""
    # return await analyze_parameter(request, "humidity")
    result = await analyze_parameter(request, "humidity")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result

@app.post("/analyze/wind_speed", response_model=WeatherAnalysisResponse)
async def analyze_wind_speed(request: WeatherRequest):
    """Analyze wind speed data with comprehensive analysis"""
    # return await analyze_parameter(request, "wind_speed")
    result = await analyze_parameter(request, "wind_speed")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result

@app.post("/analyze/precipitation", response_model=WeatherAnalysisResponse)
async def analyze_precipitation(request: WeatherRequest):
    """Analyze precipitation data with comprehensive analysis"""
    # return await analyze_parameter(request, "precipitation")
    result = await analyze_parameter(request, "precipitation")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result

@app.post("/analyze/pressure", response_model=WeatherAnalysisResponse)
async def analyze_pressure(request: WeatherRequest):
    """Analyze pressure data with comprehensive analysis"""
    # return await analyze_parameter(request, "pressure")
    result = await analyze_parameter(request, "pressure")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result

@app.post("/analyze/cloud_cover", response_model=WeatherAnalysisResponse)
async def analyze_cloud_cover(request: WeatherRequest):
    """Analyze cloud cover data with comprehensive analysis"""
    # return await analyze_parameter(request, "cloud_cover")
    result = await analyze_parameter(request, "cloud_cover")
    await insert_temperature_analysis(result.dict())  # or result if already dict
    return result



async def analyze_parameter(request: WeatherRequest, parameter: str) -> WeatherAnalysisResponse:
    try:
        weather_data = await get_current_weather(request)
        if not isinstance(weather_data, WeatherDataResponse):
            logger.error(f"Expected WeatherDataResponse, got {type(weather_data)}: {weather_data}")
            raise ValueError(f"Invalid weather data type: {type(weather_data)}")
        
        weather_dict = {
            "current": weather_data.current_conditions,
            "hourly": weather_data.hourly_data,
            "daily": weather_data.daily_data
        }
        
        df = analyzer.prepare_data(weather_dict, parameter)
        statistical_analysis = analyzer.statistical_analysis(df, parameter)
        numerical_predictions = analyzer.numerical_weather_prediction(df, parameter)
        visualization = analyzer.create_visualization(df, parameter, {
            "statistical_analysis": statistical_analysis,
            "numerical_predictions": numerical_predictions
        })
        text_generation = TextGeneration()
        llm_analysis = text_generation.generate_llm_analysis_advanced(df, parameter, statistical_analysis, numerical_predictions)
        # llm_analysis = analyzer.generate_llm_analysis(df, parameter, statistical_analysis, numerical_predictions)
        
        return WeatherAnalysisResponse(
            parameter=WEATHER_PARAMETERS[parameter]["name"],
            llm_analysis=llm_analysis,
            statistical_analysis=statistical_analysis,
            numerical_predictions=numerical_predictions,
            visualization=visualization,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "location": {"latitude": request.latitude, "longitude": request.longitude},
                "data_points": len(df),
                "parameter_unit": WEATHER_PARAMETERS[parameter]["unit"],
                "methods_used": ["Statistical Analysis", "Numerical Weather Prediction", "LLM Analysis"]
            }
        )
    except Exception as e:
        logger.error(f"Analysis failed for {parameter}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed for {parameter}: {str(e)}")


# Comprehensive analysis endpoint
@app.post("/analyze/comprehensive")
async def comprehensive_analysis(request: WeatherRequest):
    """Perform comprehensive analysis on all weather parameters"""
    try:
        results = {}
        for param in WEATHER_PARAMETERS.keys():
            try:
                analysis = await analyze_parameter(request, param)
                results[param] = analysis.dict()
            except Exception as e:
                results[param] = {"error": str(e)}
        
        return {
            "comprehensive_analysis": results,
            "summary": {
                "total_parameters": len(WEATHER_PARAMETERS),
                "successful_analyses": len([r for r in results.values() if "error" not in r]),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


class USZipRequest(BaseModel):
    city: str
    state: str
    county: str

def filter_us_zips_get_longitude_latitude(city: str, state: str, county: str) -> list:
    """Filter US. zips table to get Longitude & Latitude"""
    try:
        user = "SYS"
        password = "oracle"
        dsn = "10.42.0.243:1521/FREE"
        connection = oracledb.connect(
            user=user, password=password, dsn=dsn, mode=oracledb.SYSDBA
        )
        cursor = connection.cursor()
        
        # SQL Query to filter table
        sql = """
            SELECT
                * 
            FROM us_zips 
            WHERE 
                LOWER(CITY) LIKE :city OR   
                LOWER(STATE) LIKE :state OR   
                LOWER(COUNTY) LIKE :county
            """
        cursor.execute(
            sql,
            {'city': f'%{city}%', 'state': f'%{state}%', 'county': f'%{county}%'}
        )
        rows = cursor.fetchall()  # Fetch all rows efficiently
        results = []
        for row in rows:
            # logger.debug(row)
            results.append({
                "zip": row[0], 
                "city": row[1], 
                "state": row[2], 
                "state_abbr": row[3], 
                "county": row[4], 
                "count_code": row[5], 
                "latitude": row[6], 
                "longitude": row[7]
            })
        return results

    except Exception as e:
        logger.warning(f"Exception error: {e}")
        return []
    except oracledb.Error as e:
        logger.warning(f"Database exception error: {e}")
        return []

@app.post("/us/zips")
async def get_us_zips(request: USZipRequest):
    """Perform US Zips search"""
    try:
        zips = filter_us_zips_get_longitude_latitude(city=request.city, state=request.state, county=request.state)        
        return zips        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perform US Zips search failed: {str(e)}")
    
def filter_weather_generated_analysis(latitude: float, longitude: float) -> list:
    """Filter Weather Generated Analysis Search by Latitude & Longitude"""
    try:
        user = "SYS"
        password = "oracle"
        dsn = "10.42.0.243:1521/FREE"
        connection = oracledb.connect(
            user=user, password=password, dsn=dsn, mode=oracledb.SYSDBA
        )
        cursor = connection.cursor()
        sql = f"""
        SELECT * FROM WEATHER_ANALYSIS WHERE latitude = :latitude AND longitude = :longitude ORDER BY id DESC FETCH FIRST 1 ROWS ONLY
        """
        # SQL Query to filter table
        cursor.execute(
            sql,
            {'latitude': latitude, 'longitude': longitude}
        )
        rows = cursor.fetchall()  # Fetch all rows efficiently
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "latitude": float(row[1]),
                "longitude": float(row[2]),
                "basic_stats_sentence": row[3].read(),
                "basic_stats_response": row[4].read(),
                "trend_analysis_sentence": row[5].read(),
                "trend_analysis_response": row[6].read(),
                "variability_sentence": row[7].read(),
                "variability_response": row[8].read(),
                "cross_corr_sentence": row[9].read(),
                "cross_corr_response": row[10].read(),
                "model_perf_sentence": row[11].read(),
                "model_perf_response": row[12].read(),
                "feature_importance_sentence": row[13].read(),
                "feature_importance_response": row[14].read(),
                "future_pred_sentence": row[15].read(),
                "future_pred_response": row[16].read(),
                "created": row[17]
            })
        return results
    except Exception as e:
        logger.warning(f"Exception error: {e}")
        return []
    except oracledb.Error as e:
        logger.warning(f"Database exception error: {e}")
        return []

class CoordinateRequest(BaseModel):
    latitude: float
    longitude: float


@app.post("/weather/generated-analysis")
async def get_weather_generated_analysis(request: CoordinateRequest):
    """Perform Weather Generated Analysis Search by Latitude & Longitude"""
    try:
        results = filter_weather_generated_analysis(latitude=request.latitude, longitude=request.longitude)        
        return results        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perform Weather Generated Analysis Search failed: {str(e)}")

# CREATE TABLE analyze_weather_states (
#         id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
#         parameter               VARCHAR2(100),
#         llm_analysis            CLOB,
#         mean_value              NUMBER(10,4),
#         median_value            NUMBER(10,4),
#         std_dev                 NUMBER(10,4),
#         min_value               NUMBER(10,4),
#         max_value               NUMBER(10,4),
#         range_value             NUMBER(10,4),
#         correlation_with_time   NUMBER(10,6),
#         is_increasing           CHAR(1),
#         coeff_of_variation      NUMBER(10,6),
#         percentile_25           NUMBER(10,4),
#         percentile_75           NUMBER(10,4),
#         percentile_90           NUMBER(10,4),
#         corr_humidity           NUMBER(10,6),
#         corr_wind_speed         NUMBER(10,6),
#         corr_pressure           NUMBER(10,6),
#         mae                     NUMBER(10,6),
#         rmse                    NUMBER(10,6),
#         prediction_accuracy     NUMBER(10,6),
#         fi_pressure             NUMBER(10,6),
#         fi_temperature          NUMBER(10,6),
#         fi_lag_1                NUMBER(10,6),
#         fi_lag_2                NUMBER(10,6),
#         fi_wind_speed           NUMBER(10,6),
#         pred_hour1              NUMBER(10,4),
#         pred_hour2              NUMBER(10,4),
#         pred_hour3              NUMBER(10,4),
#         pred_hour4              NUMBER(10,4),
#         pred_hour5              NUMBER(10,4),
#         pred_hour6              NUMBER(10,4),
#         ts_hour1                TIMESTAMP,
#         ts_hour2                TIMESTAMP,
#         ts_hour3                TIMESTAMP,
#         ts_hour4                TIMESTAMP,
#         ts_hour5                TIMESTAMP,
#         ts_hour6                TIMESTAMP,
#         algorithm               VARCHAR2(100),
#         features_used           CLOB,
#         training_samples        NUMBER,
#         analysis_timestamp      TIMESTAMP,
#         latitude                NUMBER(10,6),
#         longitude               NUMBER(10,6),
#         data_points             NUMBER,
#         parameter_unit          VARCHAR2(20),
#         methods_used            CLOB,
#         visualization           CLOB,
#         CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
def filter_analyzed_weather_states(latitude: float, longitude: float) -> list:
    """Filter Analyze Weather States by Latitude & Longitude"""
    try:
        user = "SYS"
        password = "oracle"
        dsn = "10.42.0.243:1521/FREE"
        connection = oracledb.connect(
            user=user, password=password, dsn=dsn, mode=oracledb.SYSDBA
        )
        cursor = connection.cursor()
        sql = f"""
        SELECT 
    id, parameter, llm_analysis, mean_value, median_value, std_dev,
    min_value, max_value, range_value, correlation_with_time, is_increasing,
    coeff_of_variation, percentile_25, percentile_75, percentile_90,
    corr_humidity, corr_wind_speed, corr_pressure, mae, rmse,
    prediction_accuracy, fi_pressure, fi_temperature, fi_lag_1, fi_lag_2,
    fi_wind_speed, pred_hour1, pred_hour2, pred_hour3, pred_hour4,
    pred_hour5, pred_hour6, ts_hour1, ts_hour2, ts_hour3, ts_hour4,
    ts_hour5, ts_hour6, algorithm, features_used, training_samples,
    analysis_timestamp, latitude, longitude, data_points, parameter_unit,
    methods_used, visualization, created_at
FROM analyze_weather_states
WHERE latitude = :latitude AND longitude = :longitude
ORDER BY id DESC
        """
        # SQL Query to filter table
        cursor.execute(
            sql,
            {'latitude': f'{latitude}', 'longitude': f'{longitude}'}
        )
        def safe_lob(value):
            return value.read() if hasattr(value, "read") else value
        rows = cursor.fetchall()  # Fetch all rows efficiently
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "parameter": row[1],
                "llm_analysis": safe_lob(row[2]),
                "mean_value": row[3],
                "median_value": row[4],
                "std_dev": row[5],
                "min_value": row[6],
                "max_value": row[7],
                "range_value": row[8],
                "correlation_with_time": row[9],
                "is_increasing": row[10],
                "coeff_of_variation": row[11],
                "percentile_25": row[12],
                "percentile_75": row[13],
                "percentile_90": row[14],
                "corr_humidity": row[15],
                "corr_wind_speed": row[16],
                "corr_pressure": row[17],
                "mae": row[18],
                "rmse": row[19],
                "prediction_accuracy": row[20],
                "fi_pressure": row[21],
                "fi_temperature": row[22],
                "fi_lag_1": row[23],
                "fi_lag_2": row[24],
                "fi_wind_speed": row[25],
                "pred_hour1": row[26],
                "pred_hour2": row[27],
                "pred_hour3": row[28],
                "pred_hour4": row[29],
                "pred_hour5": row[30],
                "pred_hour6": row[31],
                "ts_hour1": row[32],
                "ts_hour2": row[33],
                "ts_hour3": row[34],
                "ts_hour4": row[35],
                "ts_hour5": row[36],
                "ts_hour6": row[37],
                "algorithm": row[38],
                "features_used": safe_lob(row[39]),
                "training_samples": row[40],
                "analysis_timestamp": row[41],
                "latitude": row[42],
                "longitude": row[43],
                "data_points": row[44],
                "parameter_unit": row[45],
                "methods_used": safe_lob(row[46]),
                "visualization": safe_lob(row[47]),
                "created": row[48]
            })
        # logger.info(results)
        return results
    except Exception as e:
        logger.warning(f"Exception error: {e}")
        return []
    except oracledb.Error as e:
        logger.warning(f"Database exception error: {e}")
        return []

@app.post("/weather/states-analysis")
async def get_weather_states_analysis(request: CoordinateRequest):
    """Perform Weather Generated Analysis Search by Latitude & Longitude"""
    try:
        results = filter_analyzed_weather_states(latitude=request.latitude, longitude=request.longitude)        
        return results        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perform Weather Generated Analysis Search failed: {str(e)}")

def filter_weather_visualization_analysis(latitude: float, longitude: float) -> list:
    """Filter Weather Visualization Analysis by Latitude & Longitude"""
    try:
        user = "SYS"
        password = "oracle"
        dsn = "10.42.0.243:1521/FREE"
        connection = oracledb.connect(
            user=user, password=password, dsn=dsn, mode=oracledb.SYSDBA
        )
        cursor = connection.cursor()
        sql = f"""
        SELECT * FROM WeatherVisualization WHERE latitude = :latitude AND longitude = :longitude ORDER BY id DESC
        """
        def safe_lob(value):
            return value.read() if hasattr(value, "read") else value
        # SQL Query to filter table
        cursor.execute(
            sql,
            {'latitude': latitude, 'longitude': longitude}
        )
        rows = cursor.fetchall()  # Fetch all rows efficiently
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "basic_stats": safe_lob(row[1]),
                "cross_correlations": safe_lob(row[2]),
                "feature_importance": safe_lob(row[3]),
                "future_predictions": safe_lob(row[4]),
                "variability": safe_lob(row[5]),
                "latitude": float(row[6]),
                "longitude": float(row[7]),
                "created": row[8]
            })
        return results
    except Exception as e:
        logger.warning(f"Exception error: {e}")
        return []
    except oracledb.Error as e:
        logger.warning(f"Database exception error: {e}")
        return []

@app.post("/weather/analysis-visualization")
async def get_weather_analysis_visualization(request: CoordinateRequest):
    """Perform Weather Visualization Analysis Search by Latitude & Longitude"""
    try:
        results = filter_weather_visualization_analysis(latitude=request.latitude, longitude=request.longitude)        
        return results        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perform Weather Generated Analysis Search failed: {str(e)}")

from analysis.AtlanticHurricanes import HurricaneWeatherAnalyzer
from elasticsearch8 import Elasticsearch, helpers

# Elasticsearch client
es = Elasticsearch("http://10.42.0.243:9200")

# Helper function to create index with mapping if not exists
def create_index_if_not_exists(index_name, mapping):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={"mappings": mapping})

# Mappings for each index
hurricanes_mapping = {
    "properties": {
        "metadata": {
            "properties": {
                "max_hurricane": {"type": "integer"},
                "east_us_lat": {"type": "keyword"},
                "east_us_lon": {"type": "keyword"},
                "west_africa_lat": {"type": "keyword"},
                "west_africa_lon": {"type": "keyword"},
                "month": {"type": "keyword"}
            }
        },
        "regional_comparison": {"type": "object"},
        "us_detailed": {"type": "object"},
        "africa_detailed": {"type": "object"},
        "visualization": {"type": "object"},
        "explanation": {"type": "text"},
        "results": {"type": "text"}
    }
}

clusters_mapping = {
    "properties": {
        "metadata": {
            "properties": {
                "location": {"type": "geo_point"},  # Combined lat/lon for geo queries
                "elevation": {"type": "float"},
                "timezone": {"type": "keyword"},
                "analysis_period": {"type": "object"},
                "clustering_method": {"type": "keyword"},
                "number_of_clusters": {"type": "integer"}
            }
        },
        "clusters": {"type": "object"},
        "timestamps": {"type": "object"},
        "visualizations": {"type": "object"},
        "full_report": {"type": "text"}
    }
}

segments_mapping = {
    "properties": {
        "original_data": {
            "properties": {
                "location": {"type": "geo_point"},  # Combined lat/lon
                "city": {"type": "keyword"},
                "state": {"type": "keyword"},
                "county": {"type": "keyword"},
                "start_date": {"type": "date"},
                "end_date": {"type": "date"},
                "hourly_units": {"type": "object"},
                "segmented_data": {"type": "nested"}
            }
        },
        "temperature_analysis": {"type": "object"}
    }
}

# Create indices on startup (or call manually)
create_index_if_not_exists('hurricanes', hurricanes_mapping)
create_index_if_not_exists('weather_clusters', clusters_mapping)
create_index_if_not_exists('weather_segments', segments_mapping)

class HurrricanSeason(BaseModel):  # Note: Typo in original 'HurrricanSeason' -> should be 'HurricaneSeason'
    max_hurricane: int
    east_us_lat: str
    east_us_lon: str 
    west_africa_lat: str 
    west_africa_lon: str 
    month: str
from analysis.AtlanticHurricanes import HurricaneWeatherAnalyzer
@app.post("/weather/hurricane")
async def get_weather_hurricanes(request: HurrricanSeason):
    try:
        hurricanes = HurricaneWeatherAnalyzer(
            max_hurricane=request.max_hurricane, 
            east_us_lat=request.east_us_lat, 
            east_us_lon=request.east_us_lon, 
            west_africa_lat=request.west_africa_lat, 
            west_africa_lon=request.west_africa_lon, 
            month=request.month)
        results: dict = await hurricanes.print_summary()
        
        # Index to Elasticsearch
        es.index(index='hurricanes', body=results)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WeatherCluster(BaseModel):
    filepath: str 
    max_winters: int 
    latitude: float 
    longitude: float 
    month: str
from analysis.MeteoUrlSerializer import MeteoPicksIdentifier, WeatherClusterAnalyzer
@app.post("/weather/clusters")
async def get_weather_clusters(request: WeatherCluster):
    try:
        meteo_picks = MeteoPicksIdentifier(
            max_winters=request.max_winters, latitude=request.latitude, longitude=request.longitude, month=request.month)
        weather_data = await meteo_picks.fetch_weather_data()

        meteo = WeatherClusterAnalyzer(request.filepath, weather_data)
        meteo.raw_data = weather_data  # Assign weather_data to raw_data
        meteo.df = meteo._preprocess_data()  # Preprocess the data
        meteo.fit_clusters() # Fit the clusters
        results: dict = meteo.save_analysis()
        
        # Add location to metadata for geo_point
        if 'metadata' not in results:
            results['metadata'] = {}
        results['metadata']['location'] = f"{request.latitude},{request.longitude}"
        
        # Index to Elasticsearch
        es.index(index='weather_clusters', body=results)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WeatherSegment(BaseModel):
    month: str
from analysis.RegionWeather import RegionWeather
@app.post("/weather/segments")
async def get_weather_segments(request: WeatherSegment):
    try:
        region = RegionWeather(request.month)
        results_dict: dict = await region.fetch_weather_data_with_analysis()
        
        # Assuming results_dict is the JSON structure, and it's a list with one item, we index the first item
        # Add location to original_data
        if isinstance(results_dict, list) and len(results_dict) > 0:
            item = results_dict[0]
            if 'original_data' in item and 'latitude' in item['original_data'] and 'longitude' in item['original_data']:
                item['original_data']['location'] = f"{item['original_data']['latitude']},{item['original_data']['longitude']}"
            es.index(index='weather_segments', body=item)
        
        return results_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New search endpoints

from fastapi import Query

@app.get("/search/hurricane")
async def search_hurricane(lat: float = Query(...), lon: float = Query(...)):
    try:
        # Search all docs, then filter in code if point in boxes
        res = es.search(index='hurricanes', body={"query": {"match_all": {}}})
        hits = res['hits']['hits']
        filtered = []
        for hit in hits:
            meta = hit['_source'].get('metadata', {})
            # Parse east_us box
            if 'east_us_lat' in meta and 'east_us_lon' in meta:
                min_lat, max_lat = map(float, meta['east_us_lat'].split(', '))
                min_lon, max_lon = map(float, meta['east_us_lon'].split(', '))
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    filtered.append(hit['_source'])
                    continue
            # Parse west_africa box
            if 'west_africa_lat' in meta and 'west_africa_lon' in meta:
                min_lat, max_lat = map(float, meta['west_africa_lat'].split(', '))
                min_lon, max_lon = map(float, meta['west_africa_lon'].split(', '))
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    filtered.append(hit['_source'])
        return {"results": filtered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/clusters")
async def search_clusters(lat: float = Query(...), lon: float = Query(...), radius: str = Query('10km')):
    try:
        query = {
            "query": {
                "geo_distance": {
                    "distance": radius,
                    "metadata.location": {
                        "lat": lat,
                        "lon": lon
                    }
                }
            }
        }
        res = es.search(index='weather_clusters', body=query)
        return {"results": [hit['_source'] for hit in res['hits']['hits']]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/segments")
async def search_segments(lat: float = Query(...), lon: float = Query(...), radius: str = Query('10km')):
    try:
        query = {
            "query": {
                "geo_distance": {
                    "distance": radius,
                    "original_data.location": {
                        "lat": lat,
                        "lon": lon
                    }
                }
            }
        }
        res = es.search(index='weather_segments', body=query)
        return {"results": [hit['_source'] for hit in res['hits']['hits']]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To populate with provided sample data (run once manually or via script)
# Assuming you have the JSON files: hurricanes.json, meteo_1.json, weather_segment.json

def populate_sample_data():
    # Hurricanes
    with open('hurricanes.json', 'r') as f:
        data = json.load(f)
        es.index(index='hurricanes', body=data)
    
    # Clusters (meteo_1.json) - add location
    with open('meteo_1.json', 'r') as f:
        data = json.load(f)
        data['metadata']['location'] = f"{data['metadata']['latitude']},{data['metadata']['longitude']}"
        es.index(index='weather_clusters', body=data)
    
    # Segments (weather_segment.json) - it's a list, take first
    with open('weather_segment.json', 'r') as f:
        data_list = json.load(f)
        if data_list:
            data = data_list[0]
            data['original_data']['location'] = f"{data['original_data']['latitude']},{data['original_data']['longitude']}"
            es.index(index='weather_segments', body=data)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama_connected": llm is not None,
            "analysis_ready": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)