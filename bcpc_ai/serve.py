#!/usr/bin/env python3
import argparse
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent))

from src.deployment.api_handler import create_api_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Serve BCPC Railway Model via API')
    parser.add_argument('--model', type=str, default='models/final/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model/architecture.yaml',
                       help='Path to model config')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to run server on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"Starting API server...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    
    # Create and run app
    app = create_api_app(args.model, args.config)
    
    if args.workers > 1:
        # Use gunicorn for production
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'{args.host}:{args.port}',
                'workers': args.workers,
                'worker_class': 'sync',
                'timeout': 120,
                'keepalive': 2,
                'accesslog': '-',
                'errorlog': '-'
            }
            
            StandaloneApplication(app, options).run()
            
        except ImportError:
            logger.warning("Gunicorn not installed. Running with Flask development server.")
            app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()