# Weather Analysis API

A comprehensive weather analysis service that provides statistical analysis, machine learning predictions, and LLM-powered insights for various weather parameters.

## Features

- **Real-time Weather Data Analysis**: Process current and historical weather data
- **Statistical Analysis**: Calculate mean, median, standard deviation, trends, and correlations
- **Machine Learning Predictions**: Forecast weather parameters using Random Forest models
- **LLM-Powered Insights**: Generate natural language explanations using Ollama
- **Data Visualization**: Create informative plots and charts
- **Geospatial Analysis**: Support for location-based weather patterns
- **Comprehensive Database Storage**: Store analysis results in Oracle database
- **Elasticsearch Integration**: Index and search weather analysis results

## Supported Weather Parameters

- Temperature (2m)
- Humidity (2m)
- Wind Speed (10m)
- Precipitation
- Pressure (MSL)
- Cloud Cover

## API Endpoints

### Weather Data
- `POST /weather/current` - Get current weather data for a location
- `POST /weather/information` - Get comprehensive weather information with analysis

### Parameter Analysis
- `POST /analyze/temperature` - Analyze temperature data
- `POST /analyze/humidity` - Analyze humidity data
- `POST /analyze/wind_speed` - Analyze wind speed data
- `POST /analyze/precipitation` - Analyze precipitation data
- `POST /analyze/pressure` - Analyze pressure data
- `POST /analyze/cloud_cover` - Analyze cloud cover data
- `POST /analyze/comprehensive` - Comprehensive analysis of all parameters

### Geospatial Analysis
- `POST /weather/hurricane` - Atlantic hurricane analysis
- `POST /weather/clusters` - Weather pattern clustering
- `POST /weather/segments` - Weather data segmentation

### Search Endpoints
- `GET /search/hurricane` - Search hurricane analysis by location
- `GET /search/clusters` - Search weather clusters by location
- `GET /search/segments` - Search weather segments by location

### Location Services
- `POST /us/zips` - Search US ZIP codes by city/state/county
- `POST /weather/generated-analysis` - Get generated weather analysis
- `POST /weather/states-analysis` - Get weather states analysis
- `POST /weather/analysis-visualization` - Get weather analysis visualizations

### Health Check
- `GET /health` - API health status

## Request Models

### WeatherRequest
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "parameter": "temperature"
}
```

### USZipRequest
```json
{
  "city": "San Francisco",
  "state": "California",
  "county": "San Francisco"
}
```

### CoordinateRequest
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194
}
```

## Response Models

### WeatherAnalysisResponse
```json
{
  "parameter": "Temperature 2m",
  "llm_analysis": "Natural language analysis...",
  "statistical_analysis": {
    "basic_stats": {...},
    "trend_analysis": {...},
    "variability": {...},
    "cross_correlations": {...}
  },
  "numerical_predictions": {
    "model_performance": {...},
    "feature_importance": {...},
    "future_predictions": {...}
  },
  "visualization": "base64_encoded_image",
  "metadata": {...}
}
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn oracledb pandas scikit-learn matplotlib seaborn langchain ollama elasticsearch
   ```
3. Set up environment variables:
   ```bash
   export OLLAMA_URL="http://localhost:11434"
   ```

## Configuration

1. **Oracle Database**: Configure connection parameters in the code
2. **Ollama**: Ensure Ollama is running with the required models (phi3:latest)
3. **Elasticsearch**: Configure the Elasticsearch client to point to your instance

## Running the API

```bash
uvicorn WeatherAnalyzer:app --host 0.0.0.0 --port 8003
```

## Database Schema

The API stores analysis results in several Oracle tables:
- `analyze_weather_states` - Comprehensive weather analysis
- `WEATHER_ANALYSIS` - Generated weather analysis text
- `WeatherVisualization` - Weather visualization data

## Elasticsearch Indices

- `hurricanes` - Hurricane analysis data
- `weather_clusters` - Weather pattern clusters
- `weather_segments` - Segmented weather data

## Dependencies

- FastAPI
- Uvicorn
- Oracle DB Client
- Pandas
- Scikit-learn
- Matplotlib/Seaborn
- LangChain
- Ollama
- Elasticsearch

## Example Usage

```bash
# Get current weather
curl -X POST "http://localhost:8003/weather/current" \
  -H "Content-Type: application/json" \
  -d '{"latitude":37.7749,"longitude":-122.4194,"parameter":"temperature"}'

# Analyze temperature
curl -X POST "http://localhost:8003/analyze/temperature" \
  -H "Content-Type: application/json" \
  -d '{"latitude":37.7749,"longitude":-122.4194,"parameter":"temperature"}'

# Search US ZIP codes
curl -X POST "http://localhost:8003/us/zips" \
  -H "Content-Type: application/json" \
  -d '{"city":"San Francisco","state":"California","county":"San Francisco"}'
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request
- 404: Not found
- 500: Internal server error

Error responses include a JSON body with details:
```json
{
  "status": "error",
  "message": "Error description"
}
```

## Notes

1. The API requires Ollama to be running with the phi3 model for LLM analysis
2. Oracle database connection requires proper credentials and network access
3. Elasticsearch must be configured and running for search functionality
4. For production use, consider:
   - Adding authentication
   - Implementing rate limiting
   - Setting up proper logging
   - Configuring HTTPS
