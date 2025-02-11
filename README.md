# Energy System Levelized Cost Modeling

## Overview
This repository provides an energy system modeling framework to calculate the **Levelized Cost of Energy (LCOE)** and **Levelized Cost of Hydrogen (LCOH)**. The model integrates capacity expansion, capital expenditure (CAPEX), operational expenditure (OPEX), fuel costs, and other relevant factors to estimate the cost evolution of various energy sources.

## Features
- **LCOE Calculation**: Computes levelized costs for nuclear, coal, gas, solar, and wind.
- **LCOH Calculation**: Estimates hydrogen production costs using electrolysis with different energy sources.
- **Weighted CAPEX Calculation**: Handles capacity expansion over time.
- **Energy Mix Analysis**: Supports multiple energy sources including renewable and conventional power generation.
- **Degradation & Discounting**: Accounts for degradation, discount rates, and technology lifespans.

## Installation
To use this repository, clone it and install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/YOUR_GITHUB/energy-cost-model.git
cd energy-cost-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Model
To execute the script and compute LCOE and LCOH:

```sh
python levelisedcost_v0.1.py
```

### Configuration
The model reads from an **Excel file (`data/grid.xlsx`)**, which contains:
- **Capacity Projections** (`capacity` sheet)
- **Capital Expenditure (CAPEX)** (`capex` sheet)
- **Generation Projections** (`generation` sheet)
- **Fuel Costs** (`fuelcost` sheet)
- **Electrolysis Parameters** (`electrolyser` sheet)
- **Discount Rate & Lifespan Assumptions** (`assumption` sheet)

## Key Components

### `utils.py`
- **`calculate_lcoe()`**: Computes LCOE for different energy sources.
- **`calculate_lcoh()`**: Computes LCOH based on electrolysis parameters.
- **`calculate_weighted_average()`**: Calculates weighted CAPEX considering capacity additions.
- **`forecast_series()`**: Uses linear regression to predict cost trends.

### `levelisedcost_v0.1.py`
- **Reads energy system data from `grid.xlsx`**.
- **Computes LCOE for various energy sources**.
- **Computes LCOH using different electricity sources**.
- **Exports results to CSV files (`lcoe.csv`, `lcoh.csv`)**.

## Example Output
Upon successful execution, results are saved in the project directory:
- **`lcoe.csv`**: LCOE values for different energy sources.
- **`lcoh.csv`**: LCOH values for hydrogen production.

## Contributing
We welcome contributions to improve this repository! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-new`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your fork (`git push origin feature-new`).
5. Submit a pull request.

## License
This project is licensed under the **GNU General Public License v3.0**.

## Contact
For inquiries or collaboration, please contact: **[sanghyun@planit.institute](mailto:sanghyun@planit.institute)**
