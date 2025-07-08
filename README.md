# Train Scheduler

A Python-based train scheduling application that processes city data to generate optimized train schedules, types, rail manouvering, etc. 

## 🚂 Overview

The Train Scheduler is a command-line application that reads city data from CSV files and generates train schedules. It's designed to help with route planning and optimization for train services between  cities of countries that do not have railway(s) or for countries that have existing networks but want to expand their network.

## 🔧 Features

- **CSV Data Processing**: Reads city information from CSV files
- **Flexible Input**: Supports custom CSV files for different city datasets
- **Command-line Interface**: Easy to use CLI for schedule generation
- **Lebanese Cities Support**: Includes pre-configured data for Lebanon cities (2024)

## 📋 Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/MiguelIbrahimE/Train_Scheduler.git
cd Train_Scheduler
```

```markdown
2. Install dependencies using Poetry:
```bash
poetry install
```

## 💻 Usage

### Basic Usage

You can use your own CSV file with city data:

```bash
poetry run python -m src --csv input/your_custom_cities.csv
```

### CSV File Format

The input CSV file should contain city information in the following format:

```csv
city_name,latitude,longitude,population
Beirut,33.8938,35.5018,361366
Tripoli,34.4346,35.8493,229398
Sidon,33.5571,35.3814,163554
```

*Note: The exact format may vary based on the implementation. Please check the sample files in the `input/` directory.*

## 📁 Project Structure

```
Train_Scheduler/
├── src/                    # Source code directory
│   ├── __init__.py
│   ├── __main__.py        # Main entry point
│   └── ...                # Additional modules
├── input/                  # Input data directory
│   └── lebanon_cities_2024.csv  # Sample city data
├── pyproject.toml         # Poetry configuration
├── poetry.lock            # Poetry lock file
└── README.md              # This file
```

## 🗺️ How It Works

1. **Data Loading**: The application reads city data from the specified CSV file
2. **Processing**: Analyzes the city information to determine optimal routes
3. **Schedule Generation**: Creates train schedules based on:
   - Distance between cities
   - Population density
   - Geographic constraints
4. **Output**: Displays the generated schedule with arrival/departure times

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 TODO

- [ ] Add support for multiple transportation modes
- [ ] Implement real-time schedule updates
- [X] Add visualization of train routes
- [ ] Support for international routes
- [ ] API endpoint for schedule queries
- [X] Integration with mapping services

## 🐛 Known Issues

- Please check the [Issues](https://github.com/MiguelIbrahimE/Train_Scheduler/issues) page for current known issues and bug reports.

## 📄 License

This project is licensed under private licencing

## 👤 Author

**Miguel Ibrahim E**

- GitHub: [@MiguelIbrahimE](https://github.com/MiguelIbrahimE)


---

For questions or support, please open an issue on GitHub.
```
