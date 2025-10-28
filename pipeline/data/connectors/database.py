import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                self._create_tables(conn)
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create required database tables"""
        cursor = conn.cursor()
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                country TEXT NOT NULL,
                budget REAL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Routes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                name TEXT NOT NULL,
                data JSON,
                cost_estimation JSON,
                optimization_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Stations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER,
                name TEXT NOT NULL,
                station_data JSON,
                position_km REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (route_id) REFERENCES routes (id)
            )
        ''')
        
        # Timetables table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timetables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER,
                timetable_data JSON,
                optimization_results JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (route_id) REFERENCES routes (id)
            )
        ''')
        
        # Railyards table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS railyards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                name TEXT NOT NULL,
                railyard_data JSON,
                layout_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Cost data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                cost_analysis JSON,
                optimization_results JSON,
                financial_plan JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Epochs table for versioning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                epoch_id TEXT NOT NULL,
                epoch_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        conn.commit()
    
    def create_project(self, name: str, country: str, budget: float) -> int:
        """Create a new project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO projects (name, country, budget) VALUES (?, ?, ?)',
                    (name, country, budget)
                )
                project_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Created project: {name} with ID: {project_id}")
                return project_id
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            raise
    
    def save_route_data(self, project_id: int, route_name: str, route_data: Dict[str, Any]) -> int:
        """Save route data to database"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                # Check if route already exists
                cursor.execute(
                    'SELECT id FROM routes WHERE project_id = ? AND name = ?',
                    (project_id, route_name)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing route
                    cursor.execute(
                        'UPDATE routes SET data = ? WHERE id = ?',
                        (json.dumps(route_data), existing[0])
                    )
                    route_id = existing[0]
                    self.logger.debug(f"Updated existing route: {route_name}")
                else:
                    # Insert new route
                    cursor.execute(
                        'INSERT INTO routes (project_id, name, data) VALUES (?, ?, ?)',
                        (project_id, route_name, json.dumps(route_data))
                    )
                    route_id = cursor.lastrowid
                    self.logger.debug(f"Created new route: {route_name}")
                
                conn.commit()
                return route_id
        except Exception as e:
            self.logger.error(f"Failed to save route data: {e}")
            raise
    
    def save_station_data(self, route_id: int, stations: List[Dict[str, Any]]):
        """Save station data for a route"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                # Delete existing stations for this route
                cursor.execute('DELETE FROM stations WHERE route_id = ?', (route_id,))
                
                # Insert new stations
                for station in stations:
                    cursor.execute(
                        'INSERT INTO stations (route_id, name, station_data, position_km) VALUES (?, ?, ?, ?)',
                        (route_id, station.get('name', ''), json.dumps(station), station.get('position_km', 0))
                    )
                
                conn.commit()
                self.logger.debug(f"Saved {len(stations)} stations for route {route_id}")
        except Exception as e:
            self.logger.error(f"Failed to save station data: {e}")
            raise
    
    def save_timetable_data(self, route_id: int, timetable_data: Dict[str, Any]) -> int:
        """Save timetable data for a route"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO timetables (route_id, timetable_data) VALUES (?, ?)',
                    (route_id, json.dumps(timetable_data))
                )
                timetable_id = cursor.lastrowid
                
                conn.commit()
                self.logger.debug(f"Saved timetable for route {route_id}")
                return timetable_id
        except Exception as e:
            self.logger.error(f"Failed to save timetable data: {e}")
            raise
    
    def save_railyard_data(self, project_id: int, railyard_data: Dict[str, Any]) -> int:
        """Save railyard data for a project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO railyards (project_id, name, railyard_data) VALUES (?, ?, ?)',
                    (project_id, railyard_data.get('name', ''), json.dumps(railyard_data))
                )
                railyard_id = cursor.lastrowid
                
                conn.commit()
                self.logger.debug(f"Saved railyard data for project {project_id}")
                return railyard_id
        except Exception as e:
            self.logger.error(f"Failed to save railyard data: {e}")
            raise
    
    def save_cost_data(self, project_id: int, cost_data: Dict[str, Any]) -> int:
        """Save cost analysis data"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO cost_data (project_id, cost_analysis) VALUES (?, ?)',
                    (project_id, json.dumps(cost_data))
                )
                cost_id = cursor.lastrowid
                
                conn.commit()
                self.logger.debug(f"Saved cost data for project {project_id}")
                return cost_id
        except Exception as e:
            self.logger.error(f"Failed to save cost data: {e}")
            raise
    
    def save_epoch(self, project_id: int, epoch_id: str, epoch_data: Dict[str, Any]) -> int:
        """Save epoch data for versioning"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO epochs (project_id, epoch_id, epoch_data) VALUES (?, ?, ?)',
                    (project_id, epoch_id, json.dumps(epoch_data))
                )
                epoch_db_id = cursor.lastrowid
                
                conn.commit()
                self.logger.debug(f"Saved epoch {epoch_id} for project {project_id}")
                return epoch_db_id
        except Exception as e:
            self.logger.error(f"Failed to save epoch data: {e}")
            raise
    
    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project data by ID"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'country': row[2],
                        'budget': row[3],
                        'status': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Failed to get project: {e}")
            return None
    
    def get_routes(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all routes for a project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM routes WHERE project_id = ?', (project_id,))
                rows = cursor.fetchall()
                
                routes = []
                for row in rows:
                    route_data = json.loads(row[3]) if row[3] else {}
                    routes.append({
                        'id': row[0],
                        'project_id': row[1],
                        'name': row[2],
                        'data': route_data,
                        'cost_estimation': json.loads(row[4]) if row[4] else {},
                        'optimization_data': json.loads(row[5]) if row[5] else {},
                        'created_at': row[6]
                    })
                return routes
        except Exception as e:
            self.logger.error(f"Failed to get routes: {e}")
            return []
    
    def get_stations(self, route_id: int) -> List[Dict[str, Any]]:
        """Get all stations for a route"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM stations WHERE route_id = ?', (route_id,))
                rows = cursor.fetchall()
                
                stations = []
                for row in rows:
                    station_data = json.loads(row[3]) if row[3] else {}
                    stations.append({
                        'id': row[0],
                        'route_id': row[1],
                        'name': row[2],
                        'station_data': station_data,
                        'position_km': row[4],
                        'created_at': row[5]
                    })
                return stations
        except Exception as e:
            self.logger.error(f"Failed to get stations: {e}")
            return []
    
    def get_timetables(self, route_id: int) -> List[Dict[str, Any]]:
        """Get timetables for a route"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM timetables WHERE route_id = ?', (route_id,))
                rows = cursor.fetchall()
                
                timetables = []
                for row in rows:
                    timetable_data = json.loads(row[2]) if row[2] else {}
                    timetables.append({
                        'id': row[0],
                        'route_id': row[1],
                        'timetable_data': timetable_data,
                        'optimization_results': json.loads(row[3]) if row[3] else {},
                        'created_at': row[4]
                    })
                return timetables
        except Exception as e:
            self.logger.error(f"Failed to get timetables: {e}")
            return []
    
    def get_railyards(self, project_id: int) -> List[Dict[str, Any]]:
        """Get railyards for a project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM railyards WHERE project_id = ?', (project_id,))
                rows = cursor.fetchall()
                
                railyards = []
                for row in rows:
                    railyard_data = json.loads(row[3]) if row[3] else {}
                    layout_data = json.loads(row[4]) if row[4] else {}
                    railyards.append({
                        'id': row[0],
                        'project_id': row[1],
                        'name': row[2],
                        'railyard_data': railyard_data,
                        'layout_data': layout_data,
                        'created_at': row[5]
                    })
                return railyards
        except Exception as e:
            self.logger.error(f"Failed to get railyards: {e}")
            return []
    
    def get_cost_data(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get cost data for a project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM cost_data WHERE project_id = ? ORDER BY created_at DESC LIMIT 1', (project_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'project_id': row[1],
                        'cost_analysis': json.loads(row[2]) if row[2] else {},
                        'optimization_results': json.loads(row[3]) if row[3] else {},
                        'financial_plan': json.loads(row[4]) if row[4] else {},
                        'created_at': row[5]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Failed to get cost data: {e}")
            return None
    
    def get_epochs(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all epochs for a project"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM epochs WHERE project_id = ? ORDER BY created_at DESC', (project_id,))
                rows = cursor.fetchall()
                
                epochs = []
                for row in rows:
                    epoch_data = json.loads(row[3]) if row[3] else {}
                    epochs.append({
                        'id': row[0],
                        'project_id': row[1],
                        'epoch_id': row[2],
                        'epoch_data': epoch_data,
                        'created_at': row[4]
                    })
                return epochs
        except Exception as e:
            self.logger.error(f"Failed to get epochs: {e}")
            return []
    
    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all associated data"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                # Delete in correct order to respect foreign key constraints
                cursor.execute('DELETE FROM epochs WHERE project_id = ?', (project_id,))
                cursor.execute('DELETE FROM cost_data WHERE project_id = ?', (project_id,))
                cursor.execute('DELETE FROM railyards WHERE project_id = ?', (project_id,))
                
                # Get route IDs first
                cursor.execute('SELECT id FROM routes WHERE project_id = ?', (project_id,))
                route_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete stations and timetables for each route
                for route_id in route_ids:
                    cursor.execute('DELETE FROM stations WHERE route_id = ?', (route_id,))
                    cursor.execute('DELETE FROM timetables WHERE route_id = ?', (route_id,))
                
                # Delete routes
                cursor.execute('DELETE FROM routes WHERE project_id = ?', (project_id,))
                
                # Finally delete project
                cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
                
                conn.commit()
                self.logger.info(f"Deleted project {project_id} and all associated data")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete project: {e}")
            return False
    
    def export_project_data(self, project_id: int) -> Dict[str, Any]:
        """Export complete project data for backup or transfer"""
        try:
            project = self.get_project(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            routes = self.get_routes(project_id)
            for route in routes:
                route['stations'] = self.get_stations(route['id'])
                route['timetables'] = self.get_timetables(route['id'])
            
            export_data = {
                'project': project,
                'routes': routes,
                'railyards': self.get_railyards(project_id),
                'cost_data': self.get_cost_data(project_id),
                'epochs': self.get_epochs(project_id),
                'export_timestamp': str(sqlite3.datetime.datetime.now())
            }
            
            self.logger.info(f"Exported data for project {project_id}")
            return export_data
        except Exception as e:
            self.logger.error(f"Failed to export project data: {e}")
            raise
    
    def import_project_data(self, import_data: Dict[str, Any]) -> int:
        """Import project data from export format"""
        try:
            project_data = import_data['project']
            
            # Create new project
            project_id = self.create_project(
                project_data['name'],
                project_data['country'],
                project_data['budget']
            )
            
            # Import routes and associated data
            for route_data in import_data.get('routes', []):
                route_id = self.save_route_data(project_id, route_data['name'], route_data['data'])
                
                # Import stations
                for station in route_data.get('stations', []):
                    self.save_station_data(route_id, [station['station_data']])
                
                # Import timetables
                for timetable in route_data.get('timetables', []):
                    self.save_timetable_data(route_id, timetable['timetable_data'])
            
            # Import railyards
            for railyard in import_data.get('railyards', []):
                self.save_railyard_data(project_id, railyard['railyard_data'])
            
            # Import cost data
            if import_data.get('cost_data'):
                self.save_cost_data(project_id, import_data['cost_data'])
            
            self.logger.info(f"Imported project data as new project {project_id}")
            return project_id
        except Exception as e:
            self.logger.error(f"Failed to import project data: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.database_url) as conn:
                cursor = conn.cursor()
                
                stats = {}
                tables = ['projects', 'routes', 'stations', 'timetables', 'railyards', 'cost_data', 'epochs']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[table] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}