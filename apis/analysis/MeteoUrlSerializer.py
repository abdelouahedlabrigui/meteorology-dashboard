import base64
import io
import oracledb
import pandas as pd
from datetime import datetime
import pandas as pd
from IPython.display import display
import logging

import requests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeteoUrlSerializer:
    def __init__(self, max_winters: int, latitude: float, longitude: float, month: str):
        self.max_winters: int = max_winters
        self.month: str = month
        self.latitude: float = latitude
        self.longitude: float = longitude

    def save_resources(self, meteo_url: str, created: str):
        """Save LLM interaction to database"""
        try:
            connection = oracledb.connect(user="SYS", password="oracle", dsn="10.42.0.243:1521/FREE", mode=oracledb.SYSDBA)
            cursor = connection.cursor()
            sql = """
            INSERT INTO WeatherUrlTriggers 
            (meteo_url, created)
            VALUES (:meteo_url, :created)
            """
            cursor.execute(sql, meteo_url=meteo_url, created=created)            
            connection.commit()
                
        except Exception as e:
            logger.error(f"Error saving meteo url to database: {e}")

    def serialize_meteo_url(self) -> list:
        urls = []
        # Get the current date and time
        current_datetime = datetime.now()
        url_prefix = "https://archive-api.open-meteo.com/v1/era5"
        url_coordinates = f"latitude={self.latitude}&longitude={self.longitude}"
        url_hourly = "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,pressure_msl,cloud_cover"
        # Extract the year
        current_year = current_datetime.year
        for winter in range(self.max_winters):
            if winter >= 0: 
                year: int = int(current_year) - winter
                start_date, end_date = self.get_start_end_dates(year=year)
                url_dates = f"start_date={start_date}&end_date={end_date}"
                url = f"{url_prefix}?{url_coordinates}&{url_dates}&{url_hourly}"
                meteo_data = {
                    "meteo_url": url,
                    "created": datetime.now().isoformat()
                }
                urls.append(meteo_data)
                self.save_resources(meteo_url=meteo_data["meteo_url"], 
                                    created=meteo_data["created"])

        return urls

    def get_start_end_dates(self, year: int) -> tuple[str, str]:
        """
        Calculate the start and end dates for a given year and month.

        Args:
            year (int): The year for which to calculate the dates.
            month (str): The month for which to calculate the dates.

        Returns:
            tuple[str, str]: A tuple containing the start date and end date as strings in 'yyyy-MM-dd' format.
        """
        import calendar
        month_number = datetime.strptime(self.month, "%B").month
        num_days = calendar.monthrange(year, month_number)[1]
        start_date = f"{year}-{month_number:02d}-01"
        end_date = f"{year}-{month_number:02d}-{num_days:02d}"
        return start_date, end_date

import httpx
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks

class MeteoPicksIdentifier:
    def __init__(self, max_winters: int, latitude: float, longitude: float, month: str):
        self.max_winters = max_winters
        self.latitude = latitude
        self.longitude = longitude
        self.month = month
        self.meteoUrls = MeteoUrlSerializer(max_winters=self.max_winters, latitude=self.latitude, longitude=self.longitude, month=self.month)
        self.urls = self.meteoUrls.serialize_meteo_url()
        self.weather_data = None # Add this line
    
    async def fetch_weather_data(self) -> list:
        """Fetch weather data from Open-Meteo API"""  
        results = []      
        for url in self.urls:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url['meteo_url']}")
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch weather data")
                results.append(response.json())
        self.weather_data = results # And this line
        return results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict
import json
from typing import Dict, List, Optional
from analysis.llms.WeatherClustersInterpreter import MeteoClustersExpertExplainer
import base64
from io import BytesIO
from PIL import Image
import time
import oracledb
from typing import Dict, Any
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherClusterAnalyzer:
    """
    A class to analyze and cluster weather states from historical meteorological data.
    Handles temperature, humidity, wind speed, precipitation, weather codes, and pressure.
    """
    def __init__(self, filepath: str, weather_data: list):
        """
        Initialize with JSON weather data.

        Args:
            json_data: Dictionary containing weather data (from hist_meteo.json)
        """
        # , max_winters: int, latitude: float, longitude: float, month: str
        self.raw_data = None
        self.filepath = filepath
        self.weather_data = weather_data
        self.df: pd.DataFrame = None
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.optimal_clusters = None
        self.cluster_labels = None
        self.weather_code_map = self._create_weather_code_map()
        self.API_KEY = ""
        self.MODEL_NAME = "gemini-2.0-flash"
        self.API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent?key={self.API_KEY}"

        # The mimeType must match the actual format of the image you encoded (e.g., 'image/png' or 'image/jpeg').
        self.MIME_TYPE = "image/png"
        self.user = "SYS"
        self.password = "oracle"
        self.dsn = "10.42.0.243:1521/FREE"
    
    def explain_plots(self, img_url):
        """
        Calls the Gemini API with an image and a system prompt to solve a math problem.

        Args:
            encoded_image_data: The Base64 encoded string of the image.
            mime_type: The MIME type of the image (e.g., 'image/png').
        """
        
        # --- System and User Prompts ---
        # The system instruction defines the model's persona and rules.
        system_prompt = (
            "You are an expert in data visualization and analysis, particularly with geographical and meteorological data. "
            "You are capable of interpreting various types of visualizations, including scatter plots, histograms, time series graphs, heatmaps, and radar charts. "
            "When you receive a base64 encoded image of a visualization, analyze it carefully to understand the data patterns, trends, and relationships being presented."
            "Identify key features, anomalies, and potential insights that can be derived from the visualization."
            "Your analysis should consider temporal distributions, hourly distributions displayed as heatmaps, and cluster characteristics shown in radar charts."
        )
        # load image and convert to base64 data URL
        image_bytes = base64.b64decode(img_url)
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Re-encode to ensure consistent format
        buf = BytesIO()
        pil.save(buf, format="png")  # convert to JPEG for smaller size
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        # data_url = f"data:image/jpeg;base64,{img_b64}"

        # The user query is the instruction to execute for the current request.
        user_query = "Analyze the images and describe the key insights and patterns observed in the visualizations. Focus on data trends, correlations, and anomalies, considering temporal distributions, hourly distributions (heatmaps), and cluster characteristics (radar charts)."

        # --- Constructing the Multimodal Payload ---
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        # 1. The text prompt/query
                        {"text": user_query},
                        # 2. The inline image data (your encoded string)
                        {
                            "inlineData": {
                                "mimeType": self.MIME_TYPE,
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
        }

        headers = {
            'Content-Type': 'application/json'
        }

        # --- API Call with Exponential Backoff ---
        max_retries = 5
        delay = 1  # Initial delay in seconds

        # for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Gemini API (Attempt {1})...")
            response = requests.post(self.API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract and logger.info the generated text content
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            
            logger.info("\n" + "="*50)
            logger.info("✨ Gemini API Solution Received ✨")
            logger.info("="*50)
            logger.info(text_response)
            logger.info("="*50)
            return text_response

        except requests.exceptions.HTTPError as e:
            # logger.info(f"HTTP Error on attempt {attempt + 1}: {e}")
            if response.status_code == 429:
                # 429 is Too Many Requests, implement exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                # Other HTTP errors or last retry failed
                logger.info("Failed to get a successful response after all retries.")
                logger.info(f"Final response status code: {response.status_code}")
                logger.info(f"Response content: {response.text}")
                return response.text
        
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")
    
    def _create_weather_code_map(self) -> Dict[int, str]:
        """Create mapping of WMO weather codes to human-readable descriptions."""
        return {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

    def _preprocess_data(self) -> pd.DataFrame:
        """Convert JSON data to structured DataFrame with proper feature engineering."""
        # Use self.raw_data instead of fetching again
        all_dfs = []

        for data in self.raw_data:
            hourly = data['hourly']
            n_samples = len(hourly['time'])

            # Create DataFrame with all features
            data_dict = {
                'timestamp': hourly['time'],
                'temperature': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'wind_speed': hourly['wind_speed_10m'],
                'precipitation': hourly['precipitation'],
                'pressure': hourly['pressure_msl'],
                'cloud_cover': hourly['cloud_cover'],
                'weather_code': hourly['weather_code'],
                'hour': [int(ts.split('T')[1].split(':')[0]) for ts in hourly['time']],
                'day_of_month': [int(ts.split('T')[0].split('-')[2]) for ts in hourly['time']]
            }

            # One-hot encode weather codes
            weather_categories = pd.Series(hourly['weather_code']).map({
                code: category for category, codes in {
                    'clear': [0, 1, 2],
                    'overcast': [3],
                    'fog': [45, 48],
                    'drizzle': [51, 53, 55, 56, 57],
                    'rain': [61, 63, 65, 66, 67],
                    'snow': [71, 73, 75, 77, 85, 86],
                    'thunderstorm': [95, 96, 99]
                }.items() for code in codes
            }).fillna('other')

            for category in weather_categories.unique():
                data_dict[f'weather_{category}'] = (weather_categories == category).astype(int)

            all_dfs.append(pd.DataFrame(data_dict))

        # Combine all DataFrames
        self.df = pd.concat(all_dfs, ignore_index=True)
        return self.df
    
    def find_optimal_clusters(self, max_clusters: int = 10, method: str = 'kmeans') -> int:
        """
        Determine optimal number of clusters using silhouette score.

        Args:
            max_clusters: Maximum number of clusters to try
            method: Clustering method ('kmeans' or 'dbscan')

        Returns:
            Optimal number of clusters
        """
        if self.df is None:
            self._preprocess_data()

        # Select features for clustering
        features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']
        X = self.df[features]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        best_score = -1
        best_k = 2

        if method == 'kmeans':
            for k in range(2, max_clusters + 1):
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            self.optimal_clusters = best_k
            return best_k

        elif method == 'dbscan':
            # For DBSCAN we'll use a different approach since it doesn't require k
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            self.optimal_clusters = n_clusters
            return n_clusters
        
    def fit_clusters(self, n_clusters: Optional[int] = None, method: str = 'kmeans') -> np.ndarray:
        """
        Fit clustering model to the weather data.

        Args:
            n_clusters: Number of clusters (if None, uses optimal_clusters)
            method: Clustering method ('kmeans' or 'dbscan')

        Returns:
            Cluster labels for each data point
        """
        if self.df is None:
            self._preprocess_data()

        if n_clusters is None:
            if self.optimal_clusters is None:
                self.find_optimal_clusters(method=method)
            n_clusters = self.optimal_clusters

        features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']
        X = self.df[features]
        X_scaled = self.scaler.fit_transform(X)

        if method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)

        self.cluster_labels = self.cluster_model.fit_predict(X_scaled)
        self.df['cluster'] = self.cluster_labels
        return self.cluster_labels
    
    def analyze_clusters(self) -> Dict[int, Dict]:
        """
        Analyze the characteristics of each cluster.

        Returns:
            Dictionary with cluster statistics and descriptions
        """
        if self.cluster_labels is None:
            self.fit_clusters()

        analysis = {}
        features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']

        for cluster_id in np.unique(self.cluster_labels):
            if cluster_id == -1:  # Skip noise for DBSCAN
                continue

            cluster_data: pd.DataFrame = self.df[self.df['cluster'] == cluster_id]
            cluster_stats = cluster_data[features].describe().to_dict()

            # ✅ Convert keys/values to JSON-safe types
            cluster_stats = {
                feature: {
                    str(stat): float(val) if pd.notna(val) else None  # Handle NaN values
                    for stat, val in stats.items()
                }
                for feature, stats in cluster_stats.items()
            }

            logger.info(cluster_stats)

            # Get most common weather category
            weather_cols = [col for col in self.df.columns if col.startswith('weather_')]
            weather_profile = cluster_data[weather_cols].mean().sort_values(ascending=False)
            dominant_weather = weather_profile.idxmax().replace('weather_', '')

            # Get most common hour and day patterns
            hour_profile = cluster_data['hour'].value_counts(normalize=True)
            day_profile = cluster_data['day_of_month'].value_counts(normalize=True)

            analysis[cluster_id] = {
                'size': int(len(cluster_data)),
                'statistics': cluster_stats,
                'dominant_weather': dominant_weather,
                'hour_distribution': {str(k): float(v) for k, v in hour_profile.items()},
                'day_distribution': {str(k): float(v) for k, v in day_profile.items()},
                'description': self._generate_cluster_description(cluster_id, cluster_stats, dominant_weather)
            }

        return analysis
    
    def _generate_cluster_description(self, cluster_id: int,
                                    stats: Dict[str, Dict],
                                    dominant_weather: str) -> str:
        """Generate human-readable description of a weather cluster."""
        temp_mean = stats['temperature']['mean']
        humidity_mean = stats['humidity']['mean']
        wind_mean = stats['wind_speed']['mean']
        precip_mean = stats['precipitation']['mean']
        pressure_mean = stats['pressure']['mean']

        temp_desc = "cold" if temp_mean < 5 else "cool" if temp_mean < 15 else "mild" if temp_mean < 25 else "warm" if temp_mean < 30 else "hot"
        humidity_desc = "dry" if humidity_mean < 30 else "moderate humidity" if humidity_mean < 70 else "humid"
        wind_desc = "calm" if wind_mean < 5 else "light wind" if wind_mean < 15 else "windy" if wind_mean < 30 else "very windy"
        precip_desc = "dry" if precip_mean == 0 else "light precipitation" if precip_mean < 1 else "moderate precipitation" if precip_mean < 5 else "heavy precipitation"
        pressure_desc = "low pressure" if pressure_mean < 1010 else "normal pressure" if pressure_mean < 1020 else "high pressure"

        return (f"Cluster {cluster_id}: {temp_desc} ({temp_mean:.1f}°C), {humidity_desc} ({humidity_mean:.0f}%), "
                f"{wind_desc} ({wind_mean:.1f} km/h), {precip_desc} ({precip_mean:.1f} mm), "
                f"{pressure_desc} ({pressure_mean:.1f} hPa), typically {dominant_weather}")
    
    def get_cluster_occurrences(self) -> Dict[int, List[str]]:
        """
        Get timestamps for each cluster occurrence.

        Returns:
            Dictionary mapping cluster IDs to lists of timestamps
        """
        if self.cluster_labels is None:
            self.fit_clusters()

        occurrences = defaultdict(list)
        for cluster_id, timestamp in zip(self.cluster_labels, self.df["timestamp"]):
            if cluster_id != -1:
                occurrences[cluster_id].append(timestamp)
        
        return dict(occurrences)
    
    def parse_json_file_to_dict(self, file_path) -> dict:
        """
        Parses a JSON file and returns its content as a Python dictionary.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: A Python dictionary representing the JSON data, or None if an error occurs.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{file_path}'. Check for valid JSON format.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}
        
    def save_analysis(self) -> dict:
        """Save cluster analysis to JSON file with safe serialization."""

        clusters = self.analyze_clusters()
        occurrences = self.get_cluster_occurrences()

        # ✅ Convert top-level cluster keys to str
        clusters_safe = {str(k): v for k, v in clusters.items()}
        occurrences_safe = {str(k): v for k, v in occurrences.items()}
        

        analysis = {
            'metadata': {
                'latitude': float(self.weather_data[0]['latitude']),
                'longitude': float(self.weather_data[0]['longitude']),
                'elevation': float(self.weather_data[0]['elevation']),
                'timezone': str(self.weather_data[0]['timezone']),
                'analysis_period': {
                    'start': str(self.df['timestamp'].min()),
                    'end': str(self.df['timestamp'].max())
                },
                'clustering_method': str(type(self.cluster_model).__name__),
                'number_of_clusters': int(len(np.unique(self.cluster_labels)))
            },
            'clusters': clusters_safe,      # ✅ fixed
            'occurrences': occurrences_safe, # ✅ fixed
            "visualizations": {
                "temporal_distribution": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
                "hourly_distribution": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
                "cluster_characteristics": {
                    "plot_encoding": "",
                    "plot_description": ""
                }
            },
            "full_report": ""
        }

        
        
        # temperature_distribution = self.visualize_temperature_distribution(analysis)
        temporal_distribution = self.visualize_temporal_distribution(analysis)
        hourly_distribution = self.visualize_hourly_distribution_heatmap(analysis)
        cluster_characteristics = self.visualize_radar_chart_for_cluster_characteristics(analysis)

        # analysis["visualizations"]["temperature_distribution"] = temperature_distribution[:100]
        analysis["visualizations"]["temporal_distribution"]["plot_encoding"] = temporal_distribution
        analysis["visualizations"]["hourly_distribution"]["plot_encoding"] = hourly_distribution
        analysis["visualizations"]["cluster_characteristics"]["plot_encoding"] = cluster_characteristics
        
        analysis["visualizations"]["temporal_distribution"]["plot_description"] = self.explain_plots(temporal_distribution)
        analysis["visualizations"]["hourly_distribution"]["plot_description"] = self.explain_plots(hourly_distribution)
        analysis["visualizations"]["cluster_characteristics"]["plot_description"] = self.explain_plots(cluster_characteristics)


        report_path = "/home/labrigui/Software/microservices/python-software/conversationalai/apis/weather/repos/analysis/data"
        explainer = MeteoClustersExpertExplainer(model_name="gpt2-medium")
        full_report = explainer.generate_full_report(analysis, save_to_file=f"{report_path}/meteo_clusters_report.txt")
        analysis["full_report"] = full_report

        with open(self.filepath, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"✅ Cluster analysis saved to {self.filepath}")
        return analysis
    
    
    # Oracle SQL database
    

    def visualize_temporal_distribution(self, data: dict) -> str:
        try:
            
            # Convert occurrences to datetime objects
            cluster_times = defaultdict(list)
            for cluster_id, times in data['occurrences'].items():
                for time_str in times:
                    cluster_times[cluster_id].append(pd.to_datetime(time_str))

            # Plot temporal distribution
            fig, ax = plt.subplots(figsize=(15, 6))

            for cluster_id, times in cluster_times.items():
                ax.scatter(times, [cluster_id]*len(times),
                        label=f'Cluster {cluster_id} ({len(times)} points)',
                        alpha=0.5)

            ax.set_yticks(range(len(data['clusters'])))
            ax.set_yticklabels([f'Cluster {i}' for i in range(len(data['clusters']))])
            ax.set_ylabel('Cluster')
            ax.set_xlabel('Date')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')    
            buffer.seek(0)

            # Encode the image in base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Close the buffer and plot
            buffer.close()
            plt.close()
            return encoded_image
        except Exception as e:
            logger.error(str(e))

    def visualize_hourly_distribution_heatmap(self, data: dict) -> str:
        try:
            
            hourly_data = []
            for cluster_id, cluster_info in data['clusters'].items():
                for hour, prob in cluster_info['hour_distribution'].items():
                    hourly_data.append({
                        'Cluster': f'Cluster {cluster_id}',
                        'Hour': int(hour),
                        'Probability': prob
                    })

            df_hourly = pd.DataFrame(hourly_data)
            pivot_df = df_hourly.pivot(index='Cluster', columns='Hour', values='Probability')

            # Plot heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot_df, cmap='viridis', annot=True, fmt='.2f')
            plt.title('Hourly Distribution by Cluster')
            plt.tight_layout()
            # plt.show()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')    
            buffer.seek(0)

            # Encode the image in base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Close the buffer and plot
            buffer.close()
            plt.close()
            return encoded_image
        except Exception as e:
            logger.error(str(e))

    def visualize_radar_chart_for_cluster_characteristics(self, data: dict) -> str:
        try:
            
            categories = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Pressure']

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)

            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
            angles += angles[:1]

            for cluster_id, cluster_info in data['clusters'].items():
                stats = cluster_info['statistics']
                values = [
                    stats['temperature']['mean'],
                    stats['humidity']['mean'],
                    stats['wind_speed']['mean'],
                    stats['precipitation']['mean'],
                    stats['pressure']['mean']
                ]
                values += values[:1]

                ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster_id}')
                ax.fill(angles, values, alpha=0.25)

            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_title('Cluster Characteristics Radar Chart', pad=20)
            ax.legend(loc='upper right')
            plt.tight_layout()
            # plt.show()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')    
            buffer.seek(0)

            # Encode the image in base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Close the buffer and plot
            buffer.close()
            plt.close()
            return encoded_image
        except Exception as e:
            logger.error(str(e))
    
        

import asyncio

# if __name__ == "__main__":
#     async def main():
#         meteo_picks = MeteoPicksIdentifier(max_winters=1, latitude=38.917, longitude=-119.9865, month="january")
#         weather_data = await meteo_picks.fetch_weather_data()

#         meteo = WeatherClusterAnalyzer(filepath="/home/labrigui/Software/microservices/python-software/conversationalai/apis/weather/repos/analysis/data/meteo_1.json",
#                                        weather_data=weather_data)
#         meteo.raw_data = weather_data  # Assign weather_data to raw_data
#         meteo.df = meteo._preprocess_data()  # Preprocess the data
#         meteo.fit_clusters() # Fit the clusters
#         meteo.save_analysis()

#     asyncio.run(main())
    