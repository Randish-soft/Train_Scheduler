import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import yaml

logger = logging.getLogger(__name__)

class ModelServer:
    def __init__(self, model_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.preprocessor = None
        self.postprocessor = None
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> nn.Module:
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Dynamically create model based on config
        model_type = self.config['model']['type']
        
        if model_type == 'route_predictor':
            from ..model_architecture.route_predictor import RoutePredictor
            model = RoutePredictor(**self.config['model']['route_predictor'])
        elif model_type == 'cost_estimator':
            from ..model_architecture.cost_estimator import CostEstimator
            model = CostEstimator(**self.config['model']['cost_estimator'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Preprocess
            processed_input = self._preprocess(input_data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_input)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            result = self._postprocess(output)
            
            return {
                'status': 'success',
                'predictions': result
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _preprocess(self, input_data: Dict) -> np.ndarray:
        # Implement preprocessing logic
        features = []
        
        for feature_name in self.config['features']['required']:
            if feature_name not in input_data:
                raise ValueError(f"Missing required feature: {feature_name}")
            features.append(input_data[feature_name])
        
        return np.array(features)
    
    def _postprocess(self, output: torch.Tensor) -> Dict:
        # Convert tensor to numpy
        if isinstance(output, dict):
            result = {k: v.numpy().tolist() for k, v in output.items()}
        else:
            result = {'output': output.numpy().tolist()}
        
        return result

def create_app(model_path: str, config_path: str) -> Flask:
    app = Flask(__name__)
    server = ModelServer(model_path, config_path)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            result = server.predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
    
    @app.route('/model_info', methods=['GET'])
    def model_info():
        return jsonify({
            'model_type': server.config['model']['type'],
            'version': server.config.get('version', '1.0.0'),
            'features': server.config['features']['required']
        })
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server')
    
    args = parser.parse_args()
    
    app = create_app(args.model, args.config)
    app.run(host='0.0.0.0', port=args.port, debug=False)