import pandas as pd
import lib.utils as _utils

# import data

# Define start and end years
start_year = 2023
end_year = 2051

data_dt = _utils.load_excel('data/grid.xlsx', start_year, end_year)

# Extract specific DataFrames
capacity_df = (data_dt['capacity'] * 1e3).round(0)                  # MW to kW
capex_df = data_dt['capex'].round(0)                                # krw / kW
generation_df = (data_dt['generation'] * 1e6).round(0)              # GWh to kWh
opex_df = data_dt['opex'].round(0)                                  # krw / kW
fuel_df = data_dt['fuelcost'].round(2)                              # krw / kWh
# altfuelgeneration_df = data_dt['altfuelgeneration'] * 1e9         # TWh to kWh
# altfuelquant_df = data_dt['altfuelquant'] * 1e6                   # kT to kg
altfuelcost_df = data_dt['altfuelcost']                             # krw/kg-fuel
fuelemission_df = data_dt['fuelemission'].round(2)                  # kg-co2/kWh
landcost_df = data_dt['landcost'].round(0)                          # krw / kW
lifespan_df = data_dt['lifespan'].round(0)

electrolysis_df = data_dt['electrolyser'].round(2)                  # kwh/kgh2, krw/kw, krw/kw.yea
discount_rate = (data_dt['assumption'].loc['discount_rate'].values[0]/100).round(3) # percentage
# calculate the difference
delta_df = capacity_df.diff()
delta_df.iloc[0] = capacity_df.iloc[0]

# calculate weighted averaged capex
wcapex_df = _utils.calculate_weighted_average(delta_df, capex_df)
# wcapex_df = capex_df
# Create a new DataFrame for LCOE calculations
lcoe_df = wcapex_df.copy()
lcoe_df[:] = 0

# Loop through each row and column to calculate LCOE for each entry
for idx, row in wcapex_df.iterrows():
    for col, _ in row.items():
        # Extract values for the current entry
        capacity = capacity_df.loc[idx, col]            # in kW
        capex_per_mw = wcapex_df.loc[idx, col]           # in krw / kW
        generation = generation_df.loc[idx, col]        # in kWh per year
        fixed_opex_per_mw = opex_df.loc[idx, col]       # in krw / kW per year
        variable_opex = fuel_df.loc[idx, col]           # in krw / kWh (fuel cost per MWh equivalent)
        land_cost_per_mw = landcost_df.loc[idx, col]    # in krw / kW per year
        lifespan = int(lifespan_df.loc[idx, col])       # in years
        # Call the updated LCOE calculation function from _utils with the extracted parameters
        lcoe_df.loc[idx, col] = _utils.calculate_lcoe(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, discount_rate)


# Create a DataFrame to store grid LCOE calculations
gridlcoe_df = pd.DataFrame({'grid': (lcoe_df * generation_df).sum(axis=1, skipna=True) / generation_df.sum(axis=1, skipna=True),})

# Prepare the DataFrame for levelized cost of hydrogen (LCOH) calculations
slcoe_df = pd.concat([gridlcoe_df, lcoe_df[['solar', 'wind']]], axis=1)
lcoh_df = slcoe_df.copy()
lcoh_df[:] = 0  # Initialize all values to zero

# Iterate through each row and column to calculate LCOH
for idx, row in slcoe_df.iterrows():
    for col, _ in row.items():

        # Define system parameters
        capacity = 1  # Assuming a 1 kW system for consistency
        capacity_factor = 0.7 if col == 'grid' else 0.3 if col == 'wind' else 0.15 if col == 'solar' else 1.0

        # Retrieve specific values for each energy source
        capex_per_kw = electrolysis_df.loc[idx, 'capex']             # currency per kW
        fixed_opex_per_kw = electrolysis_df.loc[idx, 'opex']         # currency per kW per year
        electricity_cost = slcoe_df.loc[idx, col]                    # currency per kWh
        lifespan = int(electrolysis_df.loc[idx, 'lifespan'])         # years
        efficiency = electrolysis_df.loc[idx, 'efficiency']          # kWh per kg of H₂

        # Annual energy production (kWh/year), adjusted by capacity and capacity factor
        annual_energy_production = capacity * 8760 * capacity_factor  # kWh/year

        # Calculate LCOH using the calculate_lcoh function
        lcoh_df.loc[idx, col] = _utils.calculate_lcoh(
            capex_per_kw, fixed_opex_per_kw, efficiency, electricity_cost,
            capacity, capacity_factor, discount_rate, lifespan
        )

lcoh_df['planned'] = altfuelcost_df['hydrogen']

# _utils.plot_levelisedcost(lcoe_df[['nuclear', 'coal', 'gas', 'solar', 'wind']], figsize=(10, 9))
# _utils.plot_levelisedcost(lcoh_df, figsize = (7, 10))

lcoe_df.to_csv('lcoe.csv', index=False)
lcoh_df.to_csv('lcoh.csv', index=False)