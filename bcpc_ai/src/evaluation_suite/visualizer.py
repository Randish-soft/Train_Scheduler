import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "artifacts/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_training_report(self, 
                                model_name: str,
                                config: Dict,
                                metrics: Dict,
                                training_time: float,
                                save_path: Optional[str] = None) -> str:
        
        report = f"""
# Training Report - {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
```yaml
{yaml.dump(config, default_flow_style=False)}
```

## Training Metrics
"""
        
        # Add metrics table
        metrics_df = pd.DataFrame([metrics])
        report += metrics_df.to_markdown(index=False) + "\n\n"
        
        report += f"""
## Training Information
- Total Training Time: {training_time:.2f} hours
- Device Used: {config.get('device', 'cpu')}
- Batch Size: {config.get('batch_size', 'N/A')}
- Learning Rate: {config.get('learning_rate', 'N/A')}
- Epochs Trained: {config.get('epochs', 'N/A')}

## Model Performance Summary
- Best Validation Loss: {metrics.get('best_val_loss', 'N/A')}
- Final R² Score: {metrics.get('r2', 'N/A')}
- Route Accuracy: {metrics.get('route_similarity', 'N/A')}%
"""
        
        if save_path:
            output_file = self.output_dir / save_path
            output_file.write_text(report)
            logger.info(f"Training report saved to {output_file}")
        
        return report
    
    def generate_evaluation_report(self,
                                  model_name: str,
                                  test_results: Dict[str, Dict],
                                  country_metrics: Dict[str, Dict],
                                  save_path: Optional[str] = None) -> str:
        
        report = f"""
# Evaluation Report - {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Test Performance
"""
        
        # Overall metrics
        overall_df = pd.DataFrame([test_results.get('overall', {})])
        report += overall_df.to_markdown(index=False) + "\n\n"
        
        report += "## Country-Specific Performance\n"
        
        for country, metrics in country_metrics.items():
            report += f"\n### {country.capitalize()}\n"
            
            if 'route_metrics' in metrics:
                report += "#### Route Prediction\n"
                route_df = pd.DataFrame([metrics['route_metrics']])
                report += route_df.to_markdown(index=False) + "\n"
            
            if 'cost_metrics' in metrics:
                report += "\n#### Cost Estimation\n"
                cost_df = pd.DataFrame([metrics['cost_metrics']])
                report += cost_df.to_markdown(index=False) + "\n"
            
            if 'timetable_metrics' in metrics:
                report += "\n#### Timetable Optimization\n"
                time_df = pd.DataFrame([metrics['timetable_metrics']])
                report += time_df.to_markdown(index=False) + "\n"
        
        # Add recommendations
        report += self._generate_recommendations(test_results, country_metrics)
        
        if save_path:
            output_file = self.output_dir / save_path
            output_file.write_text(report)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report
    
    def generate_inference_report(self,
                                country: str,
                                predictions: Dict,
                                confidence_scores: Dict,
                                save_path: Optional[str] = None) -> str:
        
        report = f"""
# Railway Infrastructure Plan - {country.capitalize()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents the AI-generated railway infrastructure plan for {country.capitalize()}, 
including route predictions, station placements, cost estimates, and implementation timeline.

## Predicted Railway Network

### Route Overview
- Total Network Length: {predictions.get('total_length_km', 0):.1f} km
- Number of Lines: {predictions.get('num_lines', 0)}
- Number of Stations: {predictions.get('num_stations', 0)}
- Estimated Total Cost: ${predictions.get('total_cost', 0)/1e9:.2f} billion
- Construction Timeline: {predictions.get('construction_years', 0):.1f} years

### Line Details
"""
        
        if 'lines' in predictions:
            for line_id, line_data in predictions['lines'].items():
                report += f"""
#### {line_data.get('name', line_id)}
- Length: {line_data.get('length_km', 0):.1f} km
- Type: {line_data.get('type', 'Standard')}
- Max Speed: {line_data.get('max_speed', 0)} km/h
- Stations: {', '.join(line_data.get('stations', []))}
- Estimated Cost: ${line_data.get('cost', 0)/1e6:.1f} million
- Track Configuration:
  - Tunnels: {line_data.get('tunnel_pct', 0):.1f}%
  - Bridges: {line_data.get('bridge_pct', 0):.1f}%
  - Ground Level: {line_data.get('ground_pct', 0):.1f}%
"""
        
        report += "\n### Station Information\n"
        
        if 'stations' in predictions:
            stations_df = pd.DataFrame(predictions['stations'])
            report += stations_df[['name', 'type', 'platforms', 'daily_passengers', 'lat', 'lon']].to_markdown(index=False) + "\n"
        
        report += "\n## Cost Breakdown\n"
        
        if 'cost_breakdown' in predictions:
            cost_df = pd.DataFrame([predictions['cost_breakdown']])
            report += cost_df.to_markdown(index=False) + "\n"
        
        report += "\n## Implementation Timeline\n"
        
        if 'timeline' in predictions:
            for phase, details in predictions['timeline'].items():
                report += f"""
### {phase.replace('_', ' ').title()}
- Duration: {details.get('duration_months', 0)} months
- Start Date: {details.get('start_date', 'TBD')}
- End Date: {details.get('end_date', 'TBD')}
- Key Milestones: {', '.join(details.get('milestones', []))}
"""
        
        report += f"\n## Confidence Scores\n"
        confidence_df = pd.DataFrame([confidence_scores])
        report += confidence_df.to_markdown(index=False) + "\n"
        
        report += "\n## Recommendations\n"
        report += self._generate_implementation_recommendations(predictions)
        
        if save_path:
            output_file = self.output_dir / save_path
            output_file.write_text(report)
            logger.info(f"Inference report saved to {output_file}")
        
        return report
    
    def _generate_recommendations(self, test_results: Dict, 
                                 country_metrics: Dict) -> str:
        recommendations = "\n## Recommendations\n\n"
        
        # Check overall performance
        if test_results.get('overall', {}).get('rmse', float('inf')) > 5.0:
            recommendations += "- Consider collecting more training data to improve model accuracy\n"
        
        if test_results.get('overall', {}).get('r2', 0) < 0.8:
            recommendations += "- Model performance is below optimal threshold. Review feature engineering\n"
        
        # Check country-specific issues
        for country, metrics in country_metrics.items():
            if metrics.get('route_metrics', {}).get('hausdorff_distance', 0) > 10:
                recommendations += f"- Route prediction for {country} needs improvement\n"
            
            if metrics.get('cost_metrics', {}).get('cost_mape', 0) > 20:
                recommendations += f"- Cost estimation for {country} has high error rate\n"
        
        return recommendations
    
    def _generate_implementation_recommendations(self, predictions: Dict) -> str:
        recommendations = []
        
        if predictions.get('total_cost', 0) > 10e9:
            recommendations.append("- Consider phased implementation due to high total cost")
        
        if predictions.get('construction_years', 0) > 10:
            recommendations.append("- Long construction timeline suggests need for interim solutions")
        
        if predictions.get('tunnel_pct', 0) > 20:
            recommendations.append("- High tunnel percentage requires specialized engineering expertise")
        
        if predictions.get('num_stations', 0) / predictions.get('total_length_km', 1) < 0.1:
            recommendations.append("- Consider adding more stations to improve accessibility")
        
        return '\n'.join(recommendations) if recommendations else "No specific concerns identified"
    
    def generate_comparison_report(self,
                                  models: List[str],
                                  results: Dict[str, Dict],
                                  save_path: Optional[str] = None) -> str:
        report = f"""
# Model Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Compared
{', '.join(models)}

## Performance Comparison
"""
        
        comparison_data = []
        for model in models:
            model_results = results.get(model, {})
            comparison_data.append({
                'Model': model,
                'RMSE': model_results.get('rmse', 'N/A'),
                'MAE': model_results.get('mae', 'N/A'),
                'R²': model_results.get('r2', 'N/A'),
                'Training Time (h)': model_results.get('training_time', 'N/A'),
                'Inference Time (s)': model_results.get('inference_time', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        report += comparison_df.to_markdown(index=False) + "\n"
        
        # Determine best model
        best_model = min(results.keys(), key=lambda x: results[x].get('rmse', float('inf')))
        report += f"\n## Best Performing Model: {best_model}\n"
        
        if save_path:
            output_file = self.output_dir / save_path
            output_file.write_text(report)
            logger.info(f"Comparison report saved to {output_file}")
        
        return report