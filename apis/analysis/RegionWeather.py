import oracledb
from datetime import datetime
import httpx
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
import logging
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.stats import norm
from io import BytesIO
import base64
from PIL import Image
import time
import oracledb
from typing import Dict, List, Any
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionWeather:
    def __init__(self, month: str):
        self.month = month
        self.user = "SYS"
        self.password = "oracle"
        self.dsn = "10.42.0.243:1521/FREE"
        self.connection = oracledb.connect(
                user=self.user, password=self.password, dsn=self.dsn, mode=oracledb.SYSDBA
            )
        self.urls = self.serialize_meteo_url()
        self.weather_data = None # Add this line
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
            "You are an expert in data visualization and statistical analysis, particularly with meteorological data. "
            "You are capable of interpreting Gaussian distribution plots. "
            "When you receive a base64 encoded image of a Gaussian distribution plot, analyze it carefully to understand the distribution of temperatures being presented."
            "Identify key features such as the mean, standard deviation, and the overall shape of the distribution. "
            "Relate these statistical measures to potential weather patterns or anomalies."
        )
        # Decode Base64 string back into bytes
        image_bytes = base64.b64decode(img_url)
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Re-encode to ensure consistent format
        buf = BytesIO()
        pil.save(buf, format="png")  # convert to JPEG for smaller size
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        # data_url = f"data:image/jpeg;base64,{img_b64}"

        # The user query is the instruction to execute for the current request.
        user_query = "Analyze the image and describe the key insights about the temperature distribution. " \
                     "Focus on interpreting the mean and standard deviation, and what they indicate about the typical temperatures and variability."
        
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

    async def fetch_weather_data(self) -> list:
        """Fetch weather data from Open-Meteo API"""  
        results = []      
        for url_data in self.urls:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url_data['meteo_url']}")
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch weather data")
                data = response.json()
                processed_data = self._process_weather_data(data, url_data)
                results.append(processed_data)
        self.weather_data = results # And this line
        return results
    
    async def fetch_weather_data_with_analysis(self) -> list:
        """Fetch weather data and perform temperature distribution analysis"""
        results = await self.fetch_weather_data()

        analyzed_results = []
        for data in results:
            analysis = self.analyze_temperature_distribution(data)
            analyzed_results.append({
                "original_data": data,
                "temperature_analysis": analysis
            })

        self.weather_data = analyzed_results
        data = json.dumps(analyzed_results, indent=4)
        results_dict: list = json.loads(data)
        return results_dict

    def analyze_temperature_distribution(self, processed_data: dict) -> dict:
        """Analyze temperature distribution with Gaussian fitting and visualization"""
        # Extract all temperature values from segmented data
        all_temps_c = []
        for segment in processed_data["segmented_data"]:
            all_temps_c.extend(segment["temperature_2m"]["values"] if "values" in segment["temperature_2m"]
                             else [segment["temperature_2m"]["avg"]])  # Fallback to avg if no raw values

        if not all_temps_c:
            return {"error": "No temperature data available"}

        # Convert to numpy array
        temps_c = np.array(all_temps_c)

        # a. Fit Gaussian distribution
        mu_c, sigma_c = norm.fit(temps_c)

        # b. Calculate Fahrenheit parameters without converting individual values
        # Conversion formulas: F = (C × 9/5) + 32
        mu_f = (mu_c * 9/5) + 32
        sigma_f = sigma_c * 9/5  # Only scale differs, not offset

        # c. Create histogram with density plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        n, bins, patches = ax.hist(temps_c, bins=30, density=True,
                                  alpha=0.6, color='g', edgecolor='black')

        # Plot fitted distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_c, sigma_c)
        ax.plot(x, p, 'k', linewidth=2)
        title = (f"Temperature Distribution: μ={mu_c:.2f}°C, σ={sigma_c:.2f}°C\n"
                f"(Fahrenheit: μ={mu_f:.2f}°F, σ={sigma_f:.2f}°F)")
        ax.set_title(title)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density')

        # Convert plot to base64 for easy storage/transmission
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plot_description = self.explain_plots(plot_data)
        return {
            "celsius_parameters": {"mu": mu_c, "sigma": sigma_c},
            "fahrenheit_parameters": {"mu": mu_f, "sigma": sigma_f},
            "plot": plot_data,
            "plot_description": plot_description,
            "sample_size": len(temps_c),
            "temperature_range_c": {"min": min(temps_c), "max": max(temps_c)}
        }

    
    def _process_weather_data(self, data: dict, url_data: dict) -> dict:
        """Process weather data by segmenting into 20 steps and calculating max/min values"""
        # Add original URL metadata to the result
        processed = {
            "latitude": url_data["latitude"],
            "longitude": url_data["longitude"],
            "city": url_data["city"],
            "state": url_data["state"],
            "county": url_data["county"],
            "start_date": url_data["start_date"],
            "end_date": url_data["end_date"],
            "hourly_units": data["hourly_units"],
            "segmented_data": []
        }

        # Calculate segment size (20 steps)
        total_hours = len(data["hourly"]["time"])
        segment_size = total_hours // 20

        # Process each segment
        for i in range(20):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 19 else total_hours

            segment = {
                "time_range": f"{data['hourly']['time'][start_idx]} to {data['hourly']['time'][end_idx-1]}",
                "temperature_2m": {
                    "max": max(data["hourly"]["temperature_2m"][start_idx:end_idx]),
                    "min": min(data["hourly"]["temperature_2m"][start_idx:end_idx]),
                    "avg": sum(data["hourly"]["temperature_2m"][start_idx:end_idx]) / (end_idx - start_idx)
                },
                "relative_humidity_2m": {
                    "max": max(data["hourly"]["relative_humidity_2m"][start_idx:end_idx]),
                    "min": min(data["hourly"]["relative_humidity_2m"][start_idx:end_idx]),
                    "avg": sum(data["hourly"]["relative_humidity_2m"][start_idx:end_idx]) / (end_idx - start_idx)
                },
                "wind_speed_10m": {
                    "max": max(data["hourly"]["wind_speed_10m"][start_idx:end_idx]),
                    "min": min(data["hourly"]["wind_speed_10m"][start_idx:end_idx]),
                    "avg": sum(data["hourly"]["wind_speed_10m"][start_idx:end_idx]) / (end_idx - start_idx)
                },
                "precipitation": {
                    "max": max(data["hourly"]["precipitation"][start_idx:end_idx]),
                    "min": min(data["hourly"]["precipitation"][start_idx:end_idx]),
                    "total": sum(data["hourly"]["precipitation"][start_idx:end_idx])
                },
                "weather_code": {
                    "most_common": max(set(data["hourly"]["weather_code"][start_idx:end_idx]),
                                      key=data["hourly"]["weather_code"][start_idx:end_idx].count)
                },
                "pressure_msl": {
                    "max": max(data["hourly"]["pressure_msl"][start_idx:end_idx]),
                    "min": min(data["hourly"]["pressure_msl"][start_idx:end_idx]),
                    "avg": sum(data["hourly"]["pressure_msl"][start_idx:end_idx]) / (end_idx - start_idx)
                },
                "cloud_cover": {
                    "max": max(data["hourly"]["cloud_cover"][start_idx:end_idx]),
                    "min": min(data["hourly"]["cloud_cover"][start_idx:end_idx]),
                    "avg": sum(data["hourly"]["cloud_cover"][start_idx:end_idx]) / (end_idx - start_idx)
                }
            }
            processed["segmented_data"].append(segment)

        return processed
    
    def serialize_meteo_url(self) -> list:
        urls = []
        # Get the current date and time
        current_datetime = datetime.now()
        url_prefix = "https://archive-api.open-meteo.com/v1/era5"
        regions = self.get_us_west_south_coordinates()
        for region in regions:
            url_coordinates = f"latitude={region['latitude']}&longitude={region['longitude']}"
            url_hourly = "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code,pressure_msl,cloud_cover"
            # Extract the year
            current_year = current_datetime.year
            winter = 0
            year: int = int(current_year) - winter
            start_date, end_date = self.get_start_end_dates(year=year)
            url_dates = f"start_date={start_date}&end_date={end_date}"
            url = f"{url_prefix}?{url_coordinates}&{url_dates}&{url_hourly}"
            meteo_data = {
                "meteo_url": url,
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "city": region["city"],
                "state": region["state"],
                "county": region["county"],
                "start_date": start_date,
                "end_date": end_date,
                "created": datetime.now().isoformat()
            }
            urls.append(meteo_data)
            self.save_resources(meteo_url=meteo_data["meteo_url"], 
                                created=meteo_data["created"])

        return urls
    
    def save_resources(self, meteo_url: str, created: str):
        """Save LLM interaction to database"""
        try:
            
            cursor = self.connection.cursor()
            sql = """
            INSERT INTO WeatherUrlTriggers 
            (meteo_url, created)
            VALUES (:meteo_url, :created)
            """
            cursor.execute(sql, meteo_url=meteo_url, created=created)            
            self.connection.commit()
            
                
        except Exception as e:
            logger.error(f"Error saving meteo url to database: {e}")

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

    def get_us_west_south_coordinates(self) -> list:
        """Filter News Stocks Prompts Oracle Database Table by Title or Tickers"""
        try:
            
            cursor = self.connection.cursor()
            sql = str(open('/home/labrigui/Software/microservices/python-software/conversationalai/apis/weather/repos/analysis/queries/regions.sql',
                           'r', encoding='utf-8').read())
            # SQL Query to filter table
            cursor.execute(
                sql
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

import json
import asyncio

# if __name__ == "__main__":
#     region = RegionWeather("june")
#     results = asyncio.run(region.fetch_weather_data_with_analysis())
#     print(json.dumps(results, indent=4))
    
#     # print(results)

