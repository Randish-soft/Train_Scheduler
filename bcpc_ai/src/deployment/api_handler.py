from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
import json
import io
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class APIHandler:
    def __init__(self, model_server, preprocessor, postprocessor):
        self.model_server = model_server
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.request_history = []
        
    def handle_prediction_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Log request
            self._log_request(request_data)
            
            # Validate input
            is_valid, errors = self._validate_input(request_data)
            if not is_valid:
                return {
                    'status': 'error',
                    'errors': errors
                }
            
            # Extract prediction type
            prediction_type = request_data.get('type', 'route')
            
            # Preprocess
            processed_input = self.preprocessor.transform(
                pd.DataFrame([request_data['features']])
            )
            
            # Predict
            raw_predictions = self.model_server.predict({
                'inputs': processed_input,
                'type': prediction_type
            })
            
            # Postprocess
            final_output = self.postprocessor.process_predictions(
                raw_predictions['predictions'],
                prediction_type
            )
            
            return {
                'status': 'success',
                'predictions': final_output,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_version': self.model_server.config.get('version', '1.0.0'),
                    'prediction_type': prediction_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def handle_batch_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            batch_size = len(request_data.get('batch', []))
            results = []
            
            for item in request_data['batch']:
                result = self.handle_prediction_request(item)
                results.append(result)
            
            return {
                'status': 'success',
                'batch_size': batch_size,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def handle_file_upload(self, file_data: Any, file_type: str) -> Dict[str, Any]:
        try:
            if file_type == 'csv':
                df = pd.read_csv(io.StringIO(file_data.decode('utf-8')))
            elif file_type == 'json':
                df = pd.read_json(io.StringIO(file_data.decode('utf-8')))
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported file type: {file_type}'
                }
            
            # Process each row
            results = []
            for _, row in df.iterrows():
                result = self.handle_prediction_request({
                    'features': row.to_dict(),
                    'type': 'route'
                })
                results.append(result)
            
            return {
                'status': 'success',
                'processed_rows': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _validate_input(self, request_data: Dict) -> Tuple[bool, List[str]]:
        errors = []
        
        if 'features' not in request_data:
            errors.append("Missing 'features' field")
        
        required_features = self.model_server.config.get('features', {}).get('required', [])
        
        if 'features' in request_data:
            for feature in required_features:
                if feature not in request_data['features']:
                    errors.append(f"Missing required feature: {feature}")
        
        return len(errors) == 0, errors
    
    def _log_request(self, request_data: Dict):
        self.request_history.append({
            'timestamp': datetime.now().isoformat(),
            'request': request_data
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]

def create_api_app(model_path: str, config_path: str) -> Flask:
    app = Flask(__name__)
    CORS(app)
    
    # Initialize components
    from .model_server import ModelServer
    from .preprocessor import RailwayPreprocessor
    from .postprocessor import RailwayPostprocessor
    
    model_server = ModelServer(model_path, config_path)
    preprocessor = RailwayPreprocessor()
    postprocessor = RailwayPostprocessor()
    
    api_handler = APIHandler(model_server, preprocessor, postprocessor)
    
    @app.route('/api/v1/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            result = api_handler.handle_prediction_request(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/v1/batch_predict', methods=['POST'])
    def batch_predict():
        try:
            data = request.json
            result = api_handler.handle_batch_request(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/v1/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400
            
            file = request.files['file']
            file_type = request.form.get('type', 'csv')
            
            result = api_handler.handle_file_upload(
                file.read(),
                file_type
            )
            
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/api/v1/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/v1/model_info', methods=['GET'])
    def model_info():
        return jsonify({
            'model_type': model_server.config.get('model', {}).get('type'),
            'version': model_server.config.get('version', '1.0.0'),
            'features': model_server.config.get('features', {}).get('required', []),
            'capabilities': [
                'route_prediction',
                'cost_estimation',
                'timetable_optimization',
                'station_placement'
            ]
        })
    
    @app.route('/api/v1/stats', methods=['GET'])
    def stats():
        return jsonify({
            'total_requests': len(api_handler.request_history),
            'recent_requests': api_handler.request_history[-10:]
        })
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    app = create_api_app(args.model, args.config)
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)