import base64
import io
import oracledb
import pandas as pd
from datetime import datetime
import logging

import requests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AtlanticHurricanesUrlSerializer:
    def __init__(self, 
                 max_hurricane: int, east_us_lat: str, east_us_lon: str, 
                 west_africa_lat: str, west_africa_lon: str, month: str):
        self.us_lat: list = str(east_us_lat).split(',')
        self.us_lon: list = str(east_us_lon).split(',')
        self.month: str = month
        self.max_hurricane = max_hurricane
        self.africa_lat: list = str(west_africa_lat).split(',')
        self.africa_lon: list = str(west_africa_lon).split(',')
    
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

    def map_coordinates(self) -> dict:
        us_coordinates = []
        africa_coordinates = []
        for us_lat, us_lon in zip(list(self.us_lat), list(self.us_lon)):
            eastern_us = {
                "latitude": str(us_lat).strip(),
                "longitude": str(us_lon).strip()
            }
            us_coordinates.append({"coordinates": eastern_us})
        for africa_lat, africa_lon in zip(list(self.africa_lat), list(self.africa_lon)):
            west_africa = {
                "latitude": str(africa_lat).strip(),
                "longitude": str(africa_lon).strip()
            }
            africa_coordinates.append({"coordinates": west_africa})
        return {"eastern_us": us_coordinates, "west_africa": africa_coordinates}
    
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

    def serialize_meteo_url(self) -> list:
        us_urls = []
        africa_urls = []

        us = self.map_coordinates()["eastern_us"]
        africa = self.map_coordinates()["west_africa"]

        # Get the current date and time
        current_datetime = datetime.now()
        url_prefix = "https://archive-api.open-meteo.com/v1/era5"
        
        
        url_hourly = "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,pressure_msl,cloud_cover"
        # Extract the year
        current_year = current_datetime.year
        for hurricane in range(self.max_hurricane):
            if hurricane >= 0: 
                year: int = int(current_year) - hurricane
                start_date, end_date = self.get_start_end_dates(year=year)
                for lat_lon in us:
                    logger.info(lat_lon)
                    lat_lon = lat_lon["coordinates"]
                    url_coordinates = f"latitude={str(lat_lon['latitude']).strip()}&longitude={str(lat_lon['longitude']).strip()}"
                    url_dates = f"start_date={start_date}&end_date={end_date}"
                    url = f"{url_prefix}?{url_coordinates}&{url_dates}&{url_hourly}"
                    meteo_data = {
                        "meteo_url": url,
                        "created": datetime.now().isoformat()
                    }
                    us_urls.append(meteo_data)
                    self.save_resources(meteo_url=meteo_data["meteo_url"], 
                                        created=meteo_data["created"])
                for lat_lon in africa:
                    lat_lon = lat_lon["coordinates"]
                    url_coordinates = f"latitude={str(lat_lon['latitude']).strip()}&longitude={str(lat_lon['longitude']).strip()}"
                    url_dates = f"start_date={start_date}&end_date={end_date}"
                    url = f"{url_prefix}?{url_coordinates}&{url_dates}&{url_hourly}"
                    meteo_data = {
                        "meteo_url": url,
                        "created": datetime.now().isoformat()
                    }
                    africa_urls.append(meteo_data)
                    self.save_resources(meteo_url=meteo_data["meteo_url"], 
                                        created=meteo_data["created"])
        urls = {
            "us_urls": us_urls,
            "africa_urls": africa_urls,
        }
        return urls

import httpx
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks

class AtlanticHurricanesData:
    def __init__(self, 
                 max_hurricane: int, east_us_lat: str, east_us_lon: str, 
                 west_africa_lat: str, west_africa_lon: str, month: str):
        self.max_hurricane: int = max_hurricane
        self.east_us_lat: str = east_us_lat
        self.east_us_lon: str = east_us_lon
        self.west_africa_lat: str = west_africa_lat
        self.west_africa_lon: str = west_africa_lon
        self.month: str = month
        self.hurricanes = AtlanticHurricanesUrlSerializer(
            max_hurricane=max_hurricane, east_us_lat=east_us_lat, east_us_lon=east_us_lon, west_africa_lat=west_africa_lat, west_africa_lon=west_africa_lon, month=month)
        self.urls = self.hurricanes.serialize_meteo_url()
        self.us_urls = self.urls["us_urls"]
        self.africa_urls = self.urls["africa_urls"]
        self.weather_data = None # Add this line
    
    async def fetch_weather_data(self) -> dict:
        """Fetch weather data from Open-Meteo API"""  
        us_results = []  
        africa_results = []      

        for url in self.us_urls:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url['meteo_url']}")
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch weather data")
                us_results.append(response.json())
        for url in self.africa_urls:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url['meteo_url']}")
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch weather data")
                africa_results.append(response.json())
        results = {
            "us_results": us_results,
            "africa_results": africa_results,
        }
        self.weather_data = results # And this line
        return results

import json
import asyncio
from scipy import stats
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from analysis.llms.HurricanesAnalysisExplainer import HurricanesAnalysisExplainer
import base64
from io import BytesIO
from PIL import Image
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HurricaneWeatherAnalyzer:
    """
    A comprehensive class for analyzing hurricane and weather patterns using
    statistical distributions and meteorological concepts.
    """
    def __init__(self, 
                 max_hurricane: int, east_us_lat: str, east_us_lon: str, 
                 west_africa_lat: str, west_africa_lon: str, month: str):
        """
        Initialize the analyzer with weather data.
        
        Args:
            data_path: Path to JSON file with weather data
            data_dict: Dictionary containing weather data (alternative to file path)
        """
        self.max_hurricane: int = max_hurricane
        self.east_us_lat: str = east_us_lat
        self.east_us_lon: str = east_us_lon
        self.west_africa_lat: str = west_africa_lat
        self.west_africa_lon: str = west_africa_lon
        self.month: str = month
        self.hurricanes = AtlanticHurricanesData(
            max_hurricane=max_hurricane, east_us_lat=east_us_lat, east_us_lon=east_us_lon, west_africa_lat=west_africa_lat, west_africa_lon=west_africa_lon, month=month)
        # The API key will be provided at runtime by the environment if left as an empty string.
        self.API_KEY = ""
        self.MODEL_NAME = "gemini-2.0-flash"
        self.API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent?key={self.API_KEY}"

        # The mimeType must match the actual format of the image you encoded (e.g., 'image/png' or 'image/jpeg').
        self.MIME_TYPE = "image/png"
    
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
            "You are an expert at solving mathematical problems and coding python methods. "
            "When you receive a mathematical exercise, from a PNG image, first analyze it carefully. "
            "Then, think step by step how to best approach the problem. "
            "Finally, write a python method to solve the problem and return the solution."
            "You are an expert in data visualization and analysis, particularly with geographical and meteorological data. "
            "You are capable of interpreting various types of visualizations, including scatter plots, histograms, and time series graphs. "
            "When you receive a base64 encoded image of a visualization, analyze it carefully to understand the data patterns, trends, and relationships being presented."
            "Identify key features, anomalies, and potential insights that can be derived from the visualization."
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
        user_query = "Analyze the image and describe the key insights and patterns observed in the visualization. Focus on data trends, correlations, and anomalies."

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
    
    def extract_weather_events(self, results: List[Dict], 
                               precipitation_threshold: float = 1.0) -> List[int]:
        """
        Extract count of weather events (rainy days) from hourly data.
        
        Args:
            results: List of weather result dictionaries
            precipitation_threshold: Minimum precipitation (mm) to count as event
            
        Returns:
            List of event counts per observation period
        """
        event_counts = []
        
        for result in results:
            hourly = result.get('hourly', {})
            precip = hourly.get('precipitation', [])
            
            # Count days with precipitation above threshold
            events = sum(1 for p in precip if p >= precipitation_threshold)
            event_counts.append(events)
            
        return event_counts
    
    def poisson_analysis(self, event_counts: List[int]) -> Dict:
        """
        Perform Poisson distribution analysis on weather events.
        
        The Poisson distribution models discrete, random events like storms
        occurring within a specific time frame, assuming independence and
        constant average rate (λ).
        
        Args:
            event_counts: List of event counts
            
        Returns:
            Dictionary with lambda estimate, probabilities, and goodness-of-fit
        """
        lambda_est = np.mean(event_counts)
        n_observations = len(event_counts)
        max_count = max(event_counts)

        # Include all possible counts from 0 to max_count
        all_counts = np.arange(0, max_count + 1)
        observed_freq = np.array([np.sum(np.array(event_counts) == k) for k in all_counts])

        # Calculate expected frequencies for all counts
        expected_freq = np.array([n_observations * stats.poisson.pmf(k, lambda_est) for k in all_counts])

        # Chi-square goodness of fit (skip bins with expected frequency < 5)
        mask = expected_freq >= 5  # Only include bins with sufficient expected frequency
        if np.sum(mask) > 0:  # Ensure at least one bin is included
            chi2, p_value = stats.chisquare(observed_freq[mask], expected_freq[mask])
        else:
            chi2, p_value = 0.0, 0.0  # Handle cases with insufficient data

        return {
            'lambda': lambda_est,
            'variance': np.var(event_counts),
            'mean': lambda_est,
            'dispersion_index': np.var(event_counts) / lambda_est if lambda_est > 0 else None,
            'chi_square': chi2,
            'p_value': p_value,
            'fit_quality': 'good' if p_value > 0.05 else 'poor' if not np.isnan(p_value) else 'insufficient_data'
        }
    
    def negative_binomial_analysis(self, event_counts: List[int]) -> Dict:
        """
        Perform Negative Binomial distribution analysis.
        
        The Negative Binomial accounts for variable event rates and clustering
        effects, accommodating over-dispersion in extreme weather events.
        
        Args:
            event_counts: List of event counts
            
        Returns:
            Dictionary with r, p parameters and dispersion metrics
        """
        mean = np.mean(event_counts)
        variance = np.var(event_counts)
        
        # Method of moments estimation
        if variance > mean:
            # r = mean^2 / (variance - mean)
            r = (mean ** 2) / (variance - mean) if variance > mean else None
            # p = mean / variance
            p = mean / variance if variance > 0 else None
        else:
            r = None
            p = None
        
        return {
            'r': r,
            'p': p,
            'mean': mean,
            'variance': variance,
            'overdispersion': variance > mean,
            'dispersion_ratio': variance / mean if mean > 0 else None
        }
    
    def enso_phase_detection(self, results: List[Dict], 
                            temp_threshold: float = 0.5) -> Dict:
        """
        Detect potential ENSO phase based on temperature and pressure anomalies.
        
        ENSO encompasses El Niño (warmer) and La Niña (cooler) phases that
        disrupt normal weather patterns globally.
        
        Args:
            results: List of weather result dictionaries
            temp_threshold: Temperature deviation threshold for phase detection
            
        Returns:
            Dictionary with ENSO indicators
        """
        temps = []
        pressures = []
        
        for result in results:
            hourly = result.get('hourly', {})
            temps.extend([float(t) for t in hourly.get('temperature_2m', []) if t is not None])
            pressures.extend([float(p) for p in hourly.get('pressure_msl', []) if p is not None])
        
        temp_mean = np.mean(temps)
        temp_std = np.std(temps)
        pressure_mean = np.mean(pressures)
        
        # Simple anomaly detection
        temp_anomaly = (temp_mean - 27.5) / temp_std  # 27.5°C as baseline
        
        if temp_anomaly > temp_threshold:
            phase = 'El Niño-like'
        elif temp_anomaly < -temp_threshold:
            phase = 'La Niña-like'
        else:
            phase = 'Neutral'
        
        return {
            'detected_phase': phase,
            'temperature_anomaly': temp_anomaly,
            'mean_temperature': temp_mean,
            'mean_pressure': pressure_mean,
            'temperature_variability': temp_std
        }
    
    def nonhomogeneous_poisson_process(self, results: List[Dict]) -> Dict:
        """
        Analyze time-varying intensity of weather events.
        
        The nonhomogeneous Poisson process models events where the rate varies
        over time, accounting for seasonal patterns and temporal variability.
        
        Args:
            results: List of weather result dictionaries
            
        Returns:
            Dictionary with time-varying intensity metrics
        """
        all_times = []
        all_events = []
        
        for result in results:
            hourly = result.get('hourly', {})
            times = hourly.get('time', [])
            precip = hourly.get('precipitation', [])
            
            all_times.extend(times)
            all_events.extend([1 if p > 1.0 else 0 for p in precip])
        
        # Calculate intensity over time windows
        if len(all_events) > 0:
            early_period = all_events[:len(all_events)//2]
            late_period = all_events[len(all_events)//2:]
            
            lambda_early = np.mean(early_period) if early_period else 0
            lambda_late = np.mean(late_period) if late_period else 0
            
            # Test for homogeneity
            if lambda_early > 0 and lambda_late > 0:
                intensity_ratio = lambda_late / lambda_early
            else:
                intensity_ratio = None
        else:
            lambda_early = lambda_late = intensity_ratio = None
        
        return {
            'lambda_early_period': lambda_early,
            'lambda_late_period': lambda_late,
            'intensity_ratio': intensity_ratio,
            'is_homogeneous': abs(intensity_ratio - 1.0) < 0.2 if intensity_ratio else None,
            'temporal_variability': 'high' if intensity_ratio and abs(intensity_ratio - 1.0) > 0.5 else 'low'
        }
    
    def rainy_season_analysis(self, results: List[Dict], 
                              rain_threshold: float = 5.0) -> Dict:
        """
        Identify and characterize the typical rainy season.
        
        Args:
            results: List of weather result dictionaries
            rain_threshold: Daily precipitation threshold for rainy season
            
        Returns:
            Dictionary with rainy season characteristics
        """
        daily_precip = []
        
        for result in results:
            hourly = result.get('hourly', {})
            precip = hourly.get('precipitation', [])
            daily_precip.extend(precip)
        
        rainy_days = sum(1 for p in daily_precip if p >= rain_threshold)
        total_days = len(daily_precip)
        
        return {
            'rainy_days': rainy_days,
            'total_days': total_days,
            'rainy_day_frequency': rainy_days / total_days if total_days > 0 else 0,
            'mean_precipitation': np.mean(daily_precip),
            'max_precipitation': np.max(daily_precip) if daily_precip else 0,
            'precipitation_variability': np.std(daily_precip) if daily_precip else 0
        }
    
    def compare_regions(self, us_results: list, africa_results: list) -> Dict:
        """
        Compare weather patterns between US and Africa regions.
        
        Args:
            precip_threshold: Precipitation threshold for event counting
            
        Returns:
            Dictionary comparing both regions
        """
        precip_threshold: float = 1.0
        us_events = self.extract_weather_events(us_results, precip_threshold)
        africa_events = self.extract_weather_events(africa_results, precip_threshold)
        
        us_poisson = self.poisson_analysis(us_events)
        africa_poisson = self.poisson_analysis(africa_events)
        logger.info("us events...")
        logger.info(us_events)
        logger.info("africa events...")
        logger.info(africa_events)
        us_enso = self.enso_phase_detection(us_results)
        africa_enso = self.enso_phase_detection(africa_results)
        
        return {
            'us_analysis': {
                'poisson': us_poisson,
                'enso': us_enso,
                'event_counts': us_events
            },
            'africa_analysis': {
                'poisson': africa_poisson,
                'enso': africa_enso,
                'event_counts': africa_events
            },
            'comparison': {
                'lambda_ratio': us_poisson['lambda'] / africa_poisson['lambda'] 
                                if africa_poisson['lambda'] > 0 else None,
                'us_more_active': us_poisson['lambda'] > africa_poisson['lambda']
            }
        }
    
    def comprehensive_analysis(self, us_results: list, africa_results: list) -> Dict:
        """
        Perform comprehensive analysis using all available methods.
        
        Returns:
            Dictionary with all analysis results
        """
        
        results = {
            "metadata": {
                "max_hurricane": str(self.max_hurricane),
                "east_us_lat": str(self.east_us_lat),
                "east_us_lon": str(self.east_us_lon),
                "west_africa_lat": str(self.west_africa_lat),
                "west_africa_lon": str(self.west_africa_lon),
                "month": str(self.month),
            },
            'regional_comparison': self.compare_regions(us_results, africa_results),
            'us_detailed': {
                'negative_binomial': self.negative_binomial_analysis(
                    self.extract_weather_events(us_results)
                ),
                'nonhomogeneous_poisson': self.nonhomogeneous_poisson_process(
                    us_results
                ),
                'rainy_season': self.rainy_season_analysis(us_results)
            },
            'africa_detailed': {
                'negative_binomial': self.negative_binomial_analysis(
                    self.extract_weather_events(africa_results)
                ),
                'nonhomogeneous_poisson': self.nonhomogeneous_poisson_process(
                    africa_results
                ),
                'rainy_season': self.rainy_season_analysis(africa_results)
            },
            "visualization": {
                "precipitation": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
                "dispersion": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
                "temporal_intensity": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
                "enso_phase": {
                    "plot_encoding": "",
                    "plot_description": ""
                },
            },
            "explanation": ""
        }
        logger.info(results)
        precipitation = self.visualize_precipitation(results)
        dispersion = self.visualize_dispersion(results)
        temporal_intensity = self.visualize_temporal_intensity(results)
        enso_phase = self.visualize_enso_phase(results)
        # regional_comparison = self.visualize_regional_comparison(results)
        results["visualization"]["precipitation"]["plot_encoding"] = precipitation
        results["visualization"]["dispersion"]["plot_encoding"] = dispersion
        results["visualization"]["temporal_intensity"]["plot_encoding"] = temporal_intensity
        results["visualization"]["enso_phase"]["plot_encoding"] = enso_phase
        # results["visualization"]["regional_comparison"] = regional_comparison
        explainer = HurricanesAnalysisExplainer()
        analysis = self.serialize_analysis(results)
        explanation = explainer.generate_expert_explanation(analysis)
        results["explanation"] = explanation

        precipitation_desc = self.explain_plots(precipitation)
        dispersion_desc = self.explain_plots(dispersion)
        temporal_intensity_desc = self.explain_plots(temporal_intensity)
        enso_phase_desc = self.explain_plots(enso_phase)

        results["visualization"]["precipitation"]["plot_description"] = precipitation_desc
        results["visualization"]["dispersion"]["plot_description"] = dispersion_desc
        results["visualization"]["temporal_intensity"]["plot_description"] = temporal_intensity_desc
        results["visualization"]["enso_phase"]["plot_description"] = enso_phase_desc
        logger.info(results)
        return results
    
    def serialize_analysis(self, analysis: Dict) -> str:
        """
        Convert analysis dictionary to JSON-serializable format.
        Handles numpy types and special values like nan/inf.
        """
        import json
        import numpy as np

        def json_serializer(obj):
            """Custom JSON serializer for non-serializable objects"""
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif obj != obj:  # Check for NaN
                return None
            elif obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            return str(obj)  # Fallback for other types

        return json.dumps(analysis, default=json_serializer, indent=2)

    async def print_summary(self, analysis: Optional[Dict] = None) -> dict:
        """
        Print a human-readable summary of the analysis.
        
        Args:
            analysis: Analysis dictionary (if None, runs comprehensive_analysis)
        """
        data = await self.hurricanes.fetch_weather_data()
        us_results = data.get("us_results", [])
        africa_results = data.get("africa_results", [])
        logger.info(data)
        if analysis is None:
            analysis = self.comprehensive_analysis(us_results, africa_results)
        
        results = self.serialize_analysis(analysis=analysis)
        # # logger.info(results)
        results_dict: dict = json.loads(results)
        
        # return results_dict        
        return results_dict
    
    def visualize_precipitation(self, data: dict):
        plt.figure(figsize=(12, 6))

        us_rainy = data['us_detailed']['rainy_season']
        africa_rainy = data['africa_detailed']['rainy_season']

        metrics = ['Rainy Days', 'Total Days', 'Mean Precipitation', 'Max Precipitation']
        us_vals = [us_rainy['rainy_days'], us_rainy['total_days'],
                us_rainy['mean_precipitation'], us_rainy['max_precipitation']]
        africa_vals = [africa_rainy['rainy_days'], africa_rainy['total_days'],
                    africa_rainy['mean_precipitation'], africa_rainy['max_precipitation']]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width/2, us_vals, width, label='US')
        plt.bar(x + width/2, africa_vals, width, label='Africa')

        plt.xticks(x, metrics)
        plt.title('Rainy Season Characteristics Comparison')
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return encoded_image

    
    def visualize_dispersion(self, data: dict):
        plt.figure(figsize=(10, 6))

        regions = ['US', 'Africa']
        poisson_dispersion = [
            data['regional_comparison']['us_analysis']['poisson']['dispersion_index'],
            data['regional_comparison']['africa_analysis']['poisson']['dispersion_index']
        ]
        nb_dispersion = [
            data['us_detailed']['negative_binomial']['dispersion_ratio'],
            data['africa_detailed']['negative_binomial']['dispersion_ratio']
        ]

        x = np.arange(len(regions))
        width = 0.35

        plt.bar(x - width/2, poisson_dispersion, width, label='Poisson Dispersion')
        plt.bar(x + width/2, nb_dispersion, width, label='Negative Binomial Dispersion')

        plt.xticks(x, regions)
        plt.title('Dispersion Comparison by Region')
        plt.ylabel('Dispersion Index')
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return encoded_image

    
    def visualize_temporal_intensity(self, data: dict):
        plt.figure(figsize=(12, 6))

        us_np = data['us_detailed']['nonhomogeneous_poisson']
        africa_np = data['africa_detailed']['nonhomogeneous_poisson']

        periods = ['Early Period', 'Late Period']
        us_intensities = [us_np['lambda_early_period'], us_np['lambda_late_period']]
        africa_intensities = [africa_np['lambda_early_period'], africa_np['lambda_late_period']]

        plt.plot(periods, us_intensities, 'o-', label=f"US (Ratio: {us_np['intensity_ratio']:.2f})")
        plt.plot(periods, africa_intensities, 'o-', label=f"Africa (Ratio: {africa_np['intensity_ratio']:.2f})")

        plt.title('Temporal Variation in Hurricane Intensity (λ)')
        plt.ylabel('Intensity Parameter')
        plt.legend()
        plt.grid(True)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return encoded_image

    
    def visualize_enso_phase(self, data: dict):
        plt.figure(figsize=(10, 6))

        us_enso = data['regional_comparison']['us_analysis']['enso']
        africa_enso = data['regional_comparison']['africa_analysis']['enso']

        phases = [f"US: {us_enso['detected_phase']}", f"Africa: {africa_enso['detected_phase']}"]
        temps = [us_enso['mean_temperature'], africa_enso['mean_temperature']]
        anomalies = [us_enso['temperature_anomaly'], africa_enso['temperature_anomaly']]
        pressures = [us_enso['mean_pressure'], africa_enso['mean_pressure']]

        x = np.arange(len(phases))
        width = 0.25

        plt.bar(x - width, temps, width, label='Mean Temperature (°C)')
        plt.bar(x, anomalies, width, label='Temperature Anomaly')
        plt.bar(x + width, pressures, width, label='Mean Pressure (hPa)')

        plt.xticks(x, phases)
        plt.title('ENSO Conditions by Region')
        plt.legend()
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()
        return encoded_image

    
import oracledb
import json
from typing import Dict, Any, List

