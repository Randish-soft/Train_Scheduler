# Train Scheduler

A Python-based train scheduling application that processes city data to generate optimized train schedules, types, rail manouvering, etc.

# Warning
Resource Hogger, it will bjork your resources, so use LEARN function carefully!!!

## 🚂 Overview

The Train Scheduler is a command-line application that reads city data from CSV files and generates train conenctions and approximate cost of these connections. It’s designed to help with route planning and optimization for train services between cities of countries that do not have railway(s) or for countries that have existing networks but want to expand their network or optimize them

## 🔧 Features

- **CSV Data Processing**: Reads city information from CSV files
- Produces an approximate cost for lines

## 📋 Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/Randish-soft/Train_Scheduler.git
cd Train_Scheduler
```

2. Install Poetry Dependencies:

```bash
poetry install
```

## 💻 Usage

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
2. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
3. Push to the branch you made
4. Open a Pull Request and we’ll review it :D 

## 📝 TODO

- [x]  Add support for multiple transportation modes
- [ ]  Implement real-time schedule updates
- [x]  Add visualization of train routes
- [ ]  Support for international routes
- [ ]  API endpoint for schedule queries
- [x]  Integration with mapping services
- [ ]  Create realistic lines instead of straight ones
- [ ]  Create a 3D model of stations
- [ ]  Create an interactable environment (similar to cities skylines)

## 🐛 Known Issues

- Please check the [Issues](https://github.com/MiguelIbrahimE/Train_Scheduler/issues) page for current known issues and bug reports.

## 📄 License

This project is licensed under private licencing

## 👤 Author

**Miguel Ibrahim E**

- GitHub: [@MiguelIbrahimE](https://github.com/MiguelIbrahimE)
- GitHub: [@linster_the_second](https://github.com/linster_the_second)

---

## Also Check the Documentations Folder! We will be updating Wiki in Parallel!
