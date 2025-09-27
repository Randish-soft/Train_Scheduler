import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import folium
from folium import plugins
import logging

logger = logging.getLogger(__name__)

class RailwayVisualizer:
    def __init__(self, output_dir: str = "artifacts/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_route_comparison(self, predicted_route: np.ndarray, 
                            actual_route: np.ndarray,
                            title: str = "Route Comparison",
                            save_path: Optional[str] = None) -> go.Figure:
        fig = go.Figure()
        
        # Actual route
        if len(actual_route) > 0:
            fig.add_trace(go.Scatter3d(
                x=actual_route[:, 0],
                y=actual_route[:, 1],
                z=actual_route[:, 2] if actual_route.shape[1] > 2 else np.zeros(len(actual_route)),
                mode='lines+markers',
                name='Actual Route',
                line=dict(color='blue', width=4),
                marker=dict(size=3)
            ))
        
        # Predicted route
        if len(predicted_route) > 0:
            fig.add_trace(go.Scatter3d(
                x=predicted_route[:, 0],
                y=predicted_route[:, 1],
                z=predicted_route[:, 2] if predicted_route.shape[1] > 2 else np.zeros(len(predicted_route)),
                mode='lines+markers',
                name='Predicted Route',
                line=dict(color='red', width=4),
                marker=dict(size=3)
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)'
            ),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / save_path))
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy/R2 plot
        if 'r2' in history:
            axes[0, 1].plot(history['r2'], label='R² Score', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Model Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Custom metrics
        custom_metrics = [k for k in history.keys() 
                         if k not in ['train_loss', 'val_loss', 'r2', 'learning_rate']]
        if custom_metrics:
            for metric in custom_metrics[:3]:  # Plot up to 3 custom metrics
                axes[1, 1].plot(history[metric], label=metric)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Custom Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(str(self.output_dir / save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cost_breakdown(self, costs: Dict[str, float],
                          save_path: Optional[str] = None) -> go.Figure:
        categories = list(costs.keys())
        values = list(costs.values())
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values,
                  text=[f'${v/1e6:.1f}M' for v in values],
                  textposition='auto',
                  marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Railway Construction Cost Breakdown',
            xaxis_title='Cost Category',
            yaxis_title='Cost (USD)',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / save_path))
        
        return fig
    
    def create_interactive_map(self, gdf: gpd.GeoDataFrame,
                             stations_df: Optional[pd.DataFrame] = None,
                             save_path: Optional[str] = None) -> folium.Map:
        # Get bounds
        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create map
        m = folium.Map(location=center, zoom_start=8)
        
        # Add railway lines
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'LineString':
                coords = [[lat, lon] for lon, lat in row.geometry.coords]
                
                folium.PolyLine(
                    coords,
                    color='red' if 'predicted' in str(idx).lower() else 'blue',
                    weight=3,
                    opacity=0.8,
                    popup=f"Line {idx}"
                ).add_to(m)
        
        # Add stations
        if stations_df is not None:
            for _, station in stations_df.iterrows():
                folium.CircleMarker(
                    location=[station['lat'], station['lon']],
                    radius=5,
                    popup=f"{station.get('name', 'Station')}<br>Platforms: {station.get('platforms', 'N/A')}",
                    color='green',
                    fill=True,
                    fillColor='lightgreen'
                ).add_to(m)
        
        # Add plugins
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl().add_to(m)
        
        if save_path:
            m.save(str(self.output_dir / save_path))
        
        return m
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(str(self.output_dir / save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                              importance_scores: np.ndarray,
                              top_k: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        # Sort by importance
        indices = np.argsort(importance_scores)[-top_k:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(indices)), importance_scores[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Most Important Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(str(self.output_dir / save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_timetable_heatmap(self, timetable: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
        # Create hour x station matrix
        if 'departure_time' in timetable.columns and 'station' in timetable.columns:
            timetable['hour'] = pd.to_datetime(timetable['departure_time']).dt.hour
            
            pivot = timetable.pivot_table(
                values='train_id' if 'train_id' in timetable.columns else 'departure_time',
                index='station',
                columns='hour',
                aggfunc='count',
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='d', cbar_kws={'label': 'Number of Trains'})
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Station')
            ax.set_title('Train Frequency Heatmap')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(str(self.output_dir / save_path), dpi=300, bbox_inches='tight')
            
            return fig
    
    def plot_gradient_profile(self, route_coords: np.ndarray,
                            elevations: np.ndarray,
                            save_path: Optional[str] = None) -> go.Figure:
        distances = np.cumsum(np.sqrt(np.sum(np.diff(route_coords, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        
        gradients = np.diff(elevations) / np.diff(distances) * 100  # Percentage
        
        fig = go.Figure()
        
        # Elevation profile
        fig.add_trace(go.Scatter(
            x=distances,
            y=elevations,
            mode='lines',
            name='Elevation',
            yaxis='y',
            line=dict(color='blue', width=2)
        ))
        
        # Gradient
        fig.add_trace(go.Scatter(
            x=distances[:-1],
            y=gradients,
            mode='lines',
            name='Gradient (%)',
            yaxis='y2',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title='Route Elevation and Gradient Profile',
            xaxis_title='Distance (km)',
            yaxis=dict(title='Elevation (m)', side='left'),
            yaxis2=dict(title='Gradient (%)', side='right', overlaying='y'),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / save_path))
        
        return fig