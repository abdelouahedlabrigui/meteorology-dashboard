from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import Dict, List
import json


class MeteoClustersExpertExplainer:
    """
    A class to generate meteorological explanations using GPT-2 medium model
    from weather clustering analysis results.
    """
    
    def __init__(self, model_name: str = "gpt2-medium"):
        """
        Initialize the explainer with GPT-2 medium model.
        
        Args:
            model_name: The Hugging Face model name (default: gpt2-medium)
        """
        print(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _format_cluster_info(self, cluster_id: str, cluster_data: Dict) -> str:
        """
        Format cluster information into a readable text.
        
        Args:
            cluster_id: The cluster identifier
            cluster_data: Dictionary containing cluster statistics
            
        Returns:
            Formatted string with cluster information
        """
        stats = cluster_data['statistics']
        
        info = f"Cluster {cluster_id} Analysis:\n"
        info += f"- Size: {cluster_data['size']} hours\n"
        info += f"- Temperature: {stats['temperature']['mean']:.1f}°C "
        info += f"(range: {stats['temperature']['min']:.1f}-{stats['temperature']['max']:.1f}°C)\n"
        info += f"- Humidity: {stats['humidity']['mean']:.1f}% "
        info += f"(range: {stats['humidity']['min']:.1f}-{stats['humidity']['max']:.1f}%)\n"
        info += f"- Wind Speed: {stats['wind_speed']['mean']:.1f} km/h "
        info += f"(range: {stats['wind_speed']['min']:.1f}-{stats['wind_speed']['max']:.1f} km/h)\n"
        info += f"- Precipitation: {stats['precipitation']['mean']:.2f} mm "
        info += f"(max: {stats['precipitation']['max']:.1f} mm)\n"
        info += f"- Pressure: {stats['pressure']['mean']:.1f} hPa\n"
        info += f"- Dominant Weather: {cluster_data['dominant_weather']}\n"
        
        return info
    
    def _create_expert_prompt(self, results: Dict, cluster_id: str = None) -> str:
        """
        Create a prompt for the model to generate expert meteorological analysis.
        
        Args:
            results: The complete analysis results dictionary
            cluster_id: Specific cluster to analyze (if None, analyzes all)
            
        Returns:
            Formatted prompt string
        """
        metadata = results['metadata']
        location = f"Location: Lat {metadata['latitude']}, Lon {metadata['longitude']}, Elevation {metadata['elevation']}m"
        period = f"Period: {metadata['analysis_period']['start']} to {metadata['analysis_period']['end']}"
        
        prompt = f"As a meteorology expert, I will analyze the weather patterns.\n\n"
        prompt += f"{location}\n{period}\n\n"
        
        if cluster_id:
            cluster_data = results['clusters'][str(cluster_id)]
            prompt += self._format_cluster_info(cluster_id, cluster_data)
            prompt += f"\nExpert meteorological analysis: This weather pattern represents"
        else:
            prompt += f"Found {len(results['clusters'])} distinct weather patterns:\n\n"
            for cid, cluster_data in results['clusters'].items():
                prompt += self._format_cluster_info(cid, cluster_data) + "\n"
            prompt += "\nExpert meteorological summary: The overall weather patterns show"
        
        return prompt
    
    def generate_explanation(
        self, 
        results: Dict, 
        cluster_id: str = None,
        max_length: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate meteorological explanations using GPT-2.
        
        Args:
            results: The complete analysis results from meteo.save_analysis()
            cluster_id: Specific cluster to explain (None for overall analysis)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of different explanations to generate
            
        Returns:
            List of generated explanations
        """
        prompt = self._create_expert_prompt(results, cluster_id)
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        # Decode outputs
        explanations = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the generated part (after the prompt)
            generated = text[len(prompt):].strip()
            explanations.append(generated)
        
        return explanations
    
    def explain_all_clusters(
        self, 
        results: Dict,
        max_length: int = 200,
        temperature: float = 0.8
    ) -> Dict[str, str]:
        """
        Generate explanations for all clusters in the analysis.
        
        Args:
            results: The complete analysis results
            max_length: Maximum length for each explanation
            temperature: Sampling temperature
            
        Returns:
            Dictionary mapping cluster_id to explanation
        """
        explanations = {}
        
        for cluster_id in results['clusters'].keys():
            print(f"Generating explanation for Cluster {cluster_id}...")
            explanation = self.generate_explanation(
                results, 
                cluster_id=cluster_id,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )[0]
            explanations[cluster_id] = explanation
        
        return explanations
    
    def generate_full_report(
        self,
        results: Dict,
        save_to_file: str = None
    ) -> str:
        """
        Generate a complete meteorological report with analysis of all clusters.
        
        Args:
            results: The complete analysis results
            save_to_file: Optional filepath to save the report
            
        Returns:
            Complete report as string
        """
        metadata = results['metadata']
        
        report = "=" * 80 + "\n"
        report += "METEOROLOGICAL ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Location: Latitude {metadata['latitude']}, Longitude {metadata['longitude']}\n"
        report += f"Elevation: {metadata['elevation']} meters\n"
        report += f"Analysis Period: {metadata['analysis_period']['start']} to {metadata['analysis_period']['end']}\n"
        report += f"Clustering Method: {metadata['clustering_method']}\n"
        report += f"Number of Clusters: {metadata['number_of_clusters']}\n\n"
        
        # Overall analysis
        print("Generating overall analysis...")
        overall_explanation = self.generate_explanation(
            results,
            cluster_id=None,
            max_length=250,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        
        report += "-" * 80 + "\n"
        report += "OVERALL WEATHER PATTERN ANALYSIS\n"
        report += "-" * 80 + "\n"
        report += overall_explanation + "\n\n"
        
        # Individual cluster analyses
        cluster_explanations = self.explain_all_clusters(results, max_length=200)
        
        for cluster_id, explanation in cluster_explanations.items():
            cluster_data = results['clusters'][cluster_id]
            report += "-" * 80 + "\n"
            report += f"CLUSTER {cluster_id} DETAILED ANALYSIS\n"
            report += "-" * 80 + "\n"
            report += self._format_cluster_info(cluster_id, cluster_data)
            report += f"\nExpert Analysis:\n{explanation}\n\n"
        
        if save_to_file:
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {save_to_file}")
        
        return report


# Example usage
# if __name__ == "__main__":
#     # Load your results (example)
#     # results = meteo.save_analysis()
    
#     # Or load from file
#     with open('meteo_1.json.txt', 'r') as f:
#         results = json.load(f)
    
#     # Initialize explainer
#     explainer = MeteoClustersExpertExplainer(model_name="gpt2-medium")
    
#     # Generate explanation for a specific cluster
#     explanation = explainer.generate_explanation(results, cluster_id="0")
#     print("Cluster 0 Explanation:")
#     print(explanation[0])
#     print("\n" + "="*80 + "\n")
    
#     # Generate full report
#     full_report = explainer.generate_full_report(results, save_to_file="meteo_report.txt")
#     print(full_report)