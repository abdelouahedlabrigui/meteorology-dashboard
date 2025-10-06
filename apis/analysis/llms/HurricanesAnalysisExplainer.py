from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json
from typing import Dict, Any, Optional


class HurricanesAnalysisExplainer:
    """
    A class to generate expert meteorological explanations using Hugging Face GPT-2 medium model.
    """
    
    def __init__(self, model_name: str = "gpt2-medium"):
        """
        Initialize the explainer with GPT-2 medium model.
        
        Args:
            model_name: The Hugging Face model to use (default: gpt2-medium)
        """
        print(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def serialize_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Serialize the analysis data to JSON string.
        
        Args:
            analysis: Dictionary containing hurricane analysis data
            
        Returns:
            JSON string representation of the analysis
        """
        return json.dumps(analysis, indent=2)
    
    def create_meteorology_prompt(self, results: str) -> str:
        """
        Create a structured prompt for meteorological analysis explanation.
        
        Args:
            results: Serialized analysis data
            
        Returns:
            Formatted prompt string
        """
        # Parse the results to extract key information
        data = json.loads(results)
        
        # Extract key metrics
        us_lambda = data['regional_comparison']['us_analysis']['poisson']['lambda']
        africa_lambda = data['regional_comparison']['africa_analysis']['poisson']['lambda']
        us_enso = data['regional_comparison']['us_analysis']['enso']['detected_phase']
        africa_enso = data['regional_comparison']['africa_analysis']['enso']['detected_phase']
        us_events = sum(data['regional_comparison']['us_analysis']['event_counts'])
        africa_events = sum(data['regional_comparison']['africa_analysis']['event_counts'])
        
        prompt = f"""As a meteorology expert, I will explain this hurricane analysis:

REGIONAL COMPARISON:
- US Region: {us_events} total events, lambda={us_lambda}, ENSO phase: {us_enso}
- Africa Region: {africa_events} total events, lambda={africa_lambda}, ENSO phase: {africa_enso}

STATISTICAL FINDINGS:
- US shows overdispersion (ratio: {data['us_detailed']['negative_binomial']['dispersion_ratio']:.2f})
- Africa shows overdispersion (ratio: {data['africa_detailed']['negative_binomial']['dispersion_ratio']:.2f})
- US is {data['regional_comparison']['comparison']['lambda_ratio']:.2f}x more active than Africa

EXPERT INTERPRETATION:
The hurricane activity patterns reveal that"""
        
        return prompt
    
    def generate_expert_explanation(
        self,
        results: str,
        max_length: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate an expert meteorological explanation using GPT-2 medium.
        
        Args:
            results: Serialized analysis data from serialize_analysis()
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of explanations to generate
            
        Returns:
            Generated expert explanation text
        """
        # Create the prompt
        prompt = self.create_meteorology_prompt(results)
        
        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate the explanation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs['attention_mask']
            )
        
        # Decode the generated text
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return explanation
    
    def explain_analysis(
        self,
        analysis: Dict[str, Any],
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Complete workflow: serialize analysis and generate explanation.
        
        Args:
            analysis: Hurricane analysis dictionary
            verbose: Whether to print the explanation
            
        Returns:
            Dictionary containing serialized results and explanation
        """
        # Serialize the analysis
        results = self.serialize_analysis(analysis)
        
        # Generate explanation
        explanation = self.generate_expert_explanation(results)
        
        if verbose:
            print("\n" + "="*80)
            print("METEOROLOGICAL EXPERT EXPLANATION")
            print("="*80)
            print(explanation)
            print("="*80 + "\n")
        
        return {
            "serialized_results": results,
            "expert_explanation": explanation
        }


# Example usage
# if __name__ == "__main__":
#     # Sample analysis data (your hurricanes.json data)
#     sample_analysis = {
#         "regional_comparison": {
#             "us_analysis": {
#                 "poisson": {
#                     "lambda": 37.0,
#                     "variance": 529.0,
#                     "mean": 37.0,
#                     "dispersion_index": 14.297297297297296,
#                     "chi_square": 0.0,
#                     "p_value": 0.0,
#                     "fit_quality": "poor"
#                 },
#                 "enso": {
#                     "detected_phase": "La Niña-like",
#                     "temperature_anomaly": -0.7172468306126768,
#                     "mean_temperature": 25.315625,
#                     "mean_pressure": 1014.7296527777779,
#                     "temperature_variability": 3.0454996896034974
#                 },
#                 "event_counts": [60, 14]
#             },
#             "africa_analysis": {
#                 "poisson": {
#                     "lambda": 11.0,
#                     "variance": 100.0,
#                     "mean": 11.0,
#                     "dispersion_index": 9.090909090909092,
#                     "chi_square": 0.0,
#                     "p_value": 0.0,
#                     "fit_quality": "poor"
#                 },
#                 "enso": {
#                     "detected_phase": "Neutral",
#                     "temperature_anomaly": 0.35032330447691884,
#                     "mean_temperature": 29.061458333333334,
#                     "mean_pressure": 1010.9658333333332,
#                     "temperature_variability": 4.457192294600006
#                 },
#                 "event_counts": [21, 1]
#             },
#             "comparison": {
#                 "lambda_ratio": 3.3636363636363638,
#                 "us_more_active": True
#             }
#         },
#         "us_detailed": {
#             "negative_binomial": {
#                 "r": 2.782520325203252,
#                 "p": 0.06994328922495274,
#                 "mean": 37.0,
#                 "variance": 529.0,
#                 "overdispersion": True,
#                 "dispersion_ratio": 14.297297297297296
#             },
#             "nonhomogeneous_poisson": {
#                 "lambda_early_period": 0.0763888888888889,
#                 "lambda_late_period": 0.015277777777777777,
#                 "intensity_ratio": 0.19999999999999998,
#                 "is_homogeneous": False,
#                 "temporal_variability": "high"
#             },
#             "rainy_season": {
#                 "rainy_days": 17,
#                 "total_days": 1440,
#                 "rainy_day_frequency": 0.011805555555555555,
#                 "mean_precipitation": 0.2182638888888889,
#                 "max_precipitation": 14.8,
#                 "precipitation_variability": 0.9561822451385561
#             }
#         },
#         "africa_detailed": {
#             "negative_binomial": {
#                 "r": 1.3595505617977528,
#                 "p": 0.11,
#                 "mean": 11.0,
#                 "variance": 100.0,
#                 "overdispersion": True,
#                 "dispersion_ratio": 9.090909090909092
#             },
#             "nonhomogeneous_poisson": {
#                 "lambda_early_period": 0.02638888888888889,
#                 "lambda_late_period": 0.001388888888888889,
#                 "intensity_ratio": 0.052631578947368425,
#                 "is_homogeneous": False,
#                 "temporal_variability": "high"
#             },
#             "rainy_season": {
#                 "rainy_days": 6,
#                 "total_days": 1440,
#                 "rainy_day_frequency": 0.004166666666666667,
#                 "mean_precipitation": 0.07375,
#                 "max_precipitation": 10.9,
#                 "precipitation_variability": 0.5535289250195814
#             }
#         }
#     }
    
#     # Create explainer instance
#     explainer = MeteorologicalAnalysisExplainer()
    
#     # Generate explanation
#     result = explainer.explain_analysis(sample_analysis)
    
#     # Access individual components
#     print("\nYou can also access components separately:")
#     print(f"Serialized length: {len(result['serialized_results'])} characters")
#     print(f"Explanation length: {len(result['expert_explanation'])} characters")