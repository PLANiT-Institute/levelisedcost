import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def reduce_list_from_top(df, value):
    """
    Reduce the numbers in a single-column DataFrame from the top using the given value until it becomes zero.

    Args:
        df (pd.DataFrame): The one-column DataFrame to reduce.
        value (number): The total amount to subtract from the column elements.

    Returns:
        pd.DataFrame: A new DataFrame with the modified values after reductions.
    """
    new_df = df.copy()

    for i in range(len(new_df)):
        if value <= 0:
            new_df.iloc[-1] = 0
            break  # Exit if no value is left to subtract
        reduction = min(new_df.iloc[i], value)
        new_df.iloc[i] -= reduction
        value -= reduction

    return new_df


def calculate_weighted_average(delta_df, capex_df):
    """
    Build a capacity changes matrix where each row represents a capacity block (year when capacity was added),
    and each column represents a year from start_year to end_year. When capacity is reduced, it removes capacity
    from the oldest blocks first and sets the current year's block to zero if exhausted.

    Args:
        delta_df (pd.DataFrame): DataFrame with 'year' and 'delta' columns, representing capacity changes.
        capex_df (pd.DataFrame): DataFrame with the CAPEX values corresponding to each year.

    Returns:
        pd.DataFrame: A DataFrame where rows represent capacity blocks (years), columns represent years,
                      and values are the weighted average CAPEX.
    """
    # Initialize the output DataFrame with the same shape as capex_df
    wcapex_df = pd.DataFrame(index=capex_df.index, columns=capex_df.columns)

    # Loop over each column (i.e., each year) to calculate weighted average CAPEX
    for column in delta_df.columns:
        # Initialize a temporary DataFrame to hold cumulative capacity changes and CAPEX for each year
        temp_df = pd.DataFrame({
            'delta': delta_df[column].copy(),
            'capex': capex_df[column].copy()
        })

        # Track cumulative capacity adjustments by year
        cumulative_deltas = temp_df['delta'].copy()
        cumulative_deltas[:] = None
        # Loop through each year to compute weighted average CAPEX
        for idx in temp_df.index:
            delta = temp_df.loc[idx, 'delta']
            cumulative_deltas.loc[idx] = delta

            if delta < 0:
                # Apply the reduction function for capacity reduction on cumulative_deltas
                cumulative_deltas = reduce_list_from_top(cumulative_deltas.loc[:idx], -delta)

            # Assuming 'cumulative_deltas' and 'temp_df' are defined DataFrames or Series
            numerator = (cumulative_deltas * temp_df['capex']).sum(skipna=True)
            denominator = cumulative_deltas.loc[:idx].sum(skipna=True)

            # Perform the division
            result = numerator / denominator if denominator != 0 else float('nan')  # Avoid division by zero

            wcapex_df.loc[idx, column] = result

    # Round the DataFrame and ensure it's of float type
    wcapex_df = wcapex_df.astype(float).round(0)

    return wcapex_df

def load_excel(filepath, start_year, end_year):
    # Load all sheets into a dictionary, skipping sheets that start with 'exclude_'
    data_dt = pd.read_excel(filepath, sheet_name=None, index_col=0)

    # Iterate over each sheet
    for key in list(data_dt.keys()):
        # Skip sheets that start with 'exclude_'
        if key.startswith('exclude_'):
            del data_dt[key]
            continue

        # Get the DataFrame for the current sheet
        data = data_dt[key]

        # Drop rows and columns labeled 'unit' or 'description'
        data = data.drop(index=['unit', 'description'], errors='ignore').drop(columns=['unit', 'description'],
                                                                              errors='ignore')

        # Convert the index to integers and filter by year range if applicable
        if data.index.name == 'year':
            data.index = data.index.astype(int)
            if start_year is not None and end_year is not None:
                data = data.loc[start_year:end_year]

        # Convert the DataFrame columns to numeric types
        data = data.apply(pd.to_numeric, errors='coerce')

        # Update the dictionary with the cleaned DataFrame
        data_dt[key] = data

    return data_dt

def dualfuel_generation(df1, df2):
    """
    Aligns columns between two DataFrames, performs subtraction only on common columns,
    and leaves other columns in the first DataFrame (df1) unchanged.

    Args:
    - df1 (pd.DataFrame): The first DataFrame (e.g., generation data).
    - df2 (pd.DataFrame): The second DataFrame (e.g., alternative fuel generation data).

    Returns:
    - pd.DataFrame: A new DataFrame with aligned and subtracted values for common columns
      and original values for other columns in df1.
    """
    # Identify common columns between the two DataFrames
    common_columns = df1.columns.intersection(df2.columns)

    # Calculate the difference only for common columns
    aligned_difference = df1[common_columns] - df2[common_columns]

    # Create a copy of the first DataFrame and replace common columns with the difference
    result_df = df1.copy()
    result_df[common_columns] = aligned_difference

    return result_df

def calculate_lcoe_with_economic_lifespan(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, economic_lifespan, discount_rate, degradation=None, interest_rate=None, tax_rate=None):
    """
    Calculate the Levelized Cost of Energy (LCOE) with separate economic lifespan for CAPEX distribution.

    Parameters:
    - capacity (float): Installed capacity in MW.
    - generation (float): Initial annual generation in MWh.
    - capex_per_mw (float): Capital expenditure in USD per MW.
    - fixed_opex_per_mw (float): Fixed operating expenditure in USD per MW per year.
    - variable_opex (float): Variable operating expenditure per MWh in USD.
    - land_cost_per_mw (float): Land cost in USD per MW per year.
    - lifespan (int): Total operational lifespan of the project in years.
    - economic_lifespan (int): Economic lifespan for CAPEX distribution in years.
    - discount_rate (float): Discount rate as a decimal.
    - interest_rate (float): Interest rate on CAPEX as a decimal.
    - tax_rate (float): Tax rate on annual operating costs as a decimal.
    - degradation (float): Annual degradation rate as a decimal (e.g., 0.01 for 1% per year).

    Returns:
    - lcoe (float): Levelized Cost of Energy in USD per MWh.
    """
    # Set default values if not provided
    if degradation is None:
        degradation = 0
    if interest_rate is None:
        interest_rate = 0
    if tax_rate is None:
        tax_rate = 0
    
    # Calculate total CAPEX with interest
    capex = capex_per_mw * capacity
    capex_with_interest = capex * (1 + interest_rate)
    
    # Calculate annual CAPEX payment distributed over economic lifespan
    annual_capex_payment = capex_with_interest / economic_lifespan

    # Calculate annual fixed OPEX and land cost based on capacity
    fixed_opex = fixed_opex_per_mw * capacity
    land_cost = land_cost_per_mw * capacity

    # Initialize total present value of costs and generation
    total_cost = 0
    total_generation = 0

    # Loop through each year to calculate present values with degradation
    for year in range(1, lifespan + 1):
        # Discount factor for the current year
        discount_factor = (1 + discount_rate) ** (year - 1)

        # Adjust generation for degradation
        degraded_generation = generation * ((1 - degradation) ** (year - 1))

        # Annual variable OPEX based on degraded generation
        annual_variable_opex = degraded_generation * variable_opex

        # Annual cost includes distributed CAPEX only during economic lifespan period
        if year <= economic_lifespan:
            annual_cost = annual_capex_payment + fixed_opex + land_cost + annual_variable_opex
        else:
            annual_cost = fixed_opex + land_cost + annual_variable_opex

        # Apply tax on annual operating costs
        annual_cost_with_tax = annual_cost * (1 + tax_rate)

        # Add discounted cost to total cost PV
        total_cost += annual_cost_with_tax / discount_factor

        # Add discounted degraded generation to total generation PV
        total_generation += degraded_generation / discount_factor

    # Calculate LCOE as the ratio of total cost PV to total generation PV
    lcoe = total_cost / total_generation
    return lcoe.round(2)

def calculate_lcoe(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, discount_rate, degradation=None, interest_rate=None, tax_rate=None):
    """
    Calculate the Levelized Cost of Energy (LCOE) with degradation.

    Parameters:
    - capacity (float): Installed capacity in MW.
    - generation (float): Initial annual generation in MWh.
    - capex_per_mw (float): Capital expenditure in USD per MW.
    - fixed_opex_per_mw (float): Fixed operating expenditure in USD per MW per year.
    - variable_opex (float): Variable operating expenditure per MWh in USD.
    - land_cost_per_mw (float): Land cost in USD per MW per year.
    - lifespan (int): Lifespan of the project in years.
    - discount_rate (float): Discount rate as a decimal.
    - interest_rate (float): Interest rate on CAPEX as a decimal.
    - tax_rate (float): Tax rate on annual operating costs as a decimal.
    - degradation (float): Annual degradation rate as a decimal (e.g., 0.01 for 1% per year).

    Returns:
    - lcoe (float): Levelized Cost of Energy in USD per MWh.
    """
    # Set default values if not provided
    if degradation is None:
        degradation = 0
    if interest_rate is None:
        interest_rate = 0
    if tax_rate is None:
        tax_rate = 0
    
    # Calculate total CAPEX with interest
    capex = capex_per_mw * capacity
    capex_with_interest = capex * (1 + interest_rate)

    # Calculate annual fixed OPEX and land cost based on capacity
    fixed_opex = fixed_opex_per_mw * capacity
    land_cost = land_cost_per_mw * capacity

    # Initialize total present value of costs and generation
    total_cost = 0
    total_generation = 0

    # Loop through each year to calculate present values with degradation
    for year in range(1, lifespan + 1):
        # Discount factor for the current year
        discount_factor = (1 + discount_rate) ** (year - 1)

        # Adjust generation for degradation
        degraded_generation = generation * ((1 - degradation) ** (year - 1))

        # Annual variable OPEX based on degraded generation
        annual_variable_opex = degraded_generation * variable_opex

        # Total annual cost (CAPEX in year 1 and OPEX each year)
        if year == 1:
            annual_cost = capex_with_interest + fixed_opex + land_cost + annual_variable_opex
        else:
            annual_cost = fixed_opex + land_cost + annual_variable_opex

        # Apply tax on annual operating costs
        annual_cost_with_tax = annual_cost * (1 + tax_rate)

        # Add discounted cost to total cost PV
        total_cost += annual_cost_with_tax / discount_factor

        # Add discounted degraded generation to total generation PV
        total_generation += degraded_generation / discount_factor

    # Calculate LCOE as the ratio of total cost PV to total generation PV
    lcoe = total_cost / total_generation
    return lcoe.round(2)

def calculate_lcoh(capex_per_kw, fixed_opex_per_kw, efficiency, electricity_cost, capacity, capacity_factor, discount_rate, lifespan, degradation=0):
    """
    Calculate the Levelized Cost of Hydrogen (LCOH) with degradation.

    Parameters:
    - capex_per_kw (float): Capital expenditure per unit capacity (currency per kW).
    - fixed_opex_per_kw (float): Fixed operational expenditure per unit capacity per year (currency per kW/year).
    - efficiency (float): Electricity consumption per kg of hydrogen produced (kWh/kg H₂).
    - electricity_cost (float): Cost of electricity per kWh (currency per kWh).
    - capacity (float): System capacity in kW.
    - capacity_factor (float): Capacity factor as a decimal (e.g., 0.9 for 90%).
    - degradation (float): Annual degradation rate as a decimal (e.g., 0.01 for 1% per year).
    - discount_rate (float): Discount rate as a decimal.
    - lifespan (int): Economic lifetime of the system in years.

    Returns:
    - lcoh (float): Levelized Cost of Hydrogen in currency per kg of hydrogen.
    """
    # Initial CAPEX total based on capacity
    capex = capex_per_kw * capacity

    # Annual fixed OPEX based on capacity
    fixed_opex = fixed_opex_per_kw * capacity

    # Initialize total present value of costs and hydrogen production
    total_cost = capex  # Start with CAPEX in the numerator
    total_hydrogen_production = 0

    # Loop through each year to calculate present values with degradation
    for year in range(1, lifespan + 1):
        # Discount factor for the current year
        discount_factor = (1 + discount_rate) ** (year - 1)

        # Adjusted annual energy production due to degradation
        degraded_production = capacity * capacity_factor * (1 - degradation) ** (year - 1) * 8760  # kWh/year

        # Annual hydrogen production (kg/year) adjusted by efficiency
        annual_hydrogen_production = degraded_production / efficiency  # kg/year

        # Discounted hydrogen production for the year
        discounted_hydrogen_production = annual_hydrogen_production / discount_factor
        total_hydrogen_production += discounted_hydrogen_production

        # Annual electricity cost (currency/year)
        annual_electricity_cost = degraded_production * electricity_cost

        # Total annual operating cost (fixed OPEX + electricity cost)
        annual_operating_cost = fixed_opex + annual_electricity_cost

        # Discounted annual operating cost
        discounted_annual_cost = annual_operating_cost / discount_factor

        # Add discounted operating cost to the total cost
        total_cost += discounted_annual_cost

    # Calculate LCOH as the ratio of total cost PV to total hydrogen production PV
    lcoh = total_cost / total_hydrogen_production

    return lcoh.round(2)

import matplotlib.pyplot as plt

def plot_levelisedcost(df, figsize=(10, 6), loc = 'best', title='Levelised Cost Over Years by Technology'):
    """
    Plots the LCOH values over years for different technologies.

    Parameters:
    - df (pd.DataFrame): DataFrame containing LCOH values with years as the index
                         and technologies as columns.
    - title (str): Title of the plot.

    Returns:
    - None: Displays the plot.
    """
    # Create a figure with the specified size
    plt.figure(figsize=figsize)

    # Reset index to get 'Year' as a column
    df_reset = df.reset_index()

    # Identify the year column
    if 'year' in df_reset.columns:
        year_column = 'year'
    elif 'year' in df_reset.columns:
        year_column = 'year'
    else:
        # Assume the first column is the year
        year_column = df_reset.columns[0]

    # Loop through each technology and plot its values against the years
    for tech in df.columns:
        plt.plot(df_reset[year_column], df_reset[tech], label=tech, marker='o')

    # Set the labels and title
    plt.xlabel('Year')
    plt.ylabel('Levelised Cost')
    plt.title(title)

    # Add a legend outside the top-right corner
    plt.legend(loc=loc)

    # Show the plot with tight layout
    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.show()

def forecast_series(series, forecast_years):
    """
    Forecast future values for a single series using linear regression.

    Parameters:
    series (pd.Series): Series containing the historical data to forecast from.
                        The index should represent years.
    forecast_years (list or range): List or range of future years for forecasting.

    Returns:
    pd.Series: Series with forecasted values for the specified years.
    """
    # Get the years (as numbers) and corresponding values from the series
    historical_years = series.index.values.reshape(-1, 1)
    historical_values = series.values

    # Fit a linear regression model on historical data
    model = LinearRegression()
    model.fit(historical_years, historical_values)

    # Predict future values
    forecast_years_reshaped = np.array(forecast_years).reshape(-1, 1)
    forecasted_values = model.predict(forecast_years_reshaped)

    # Return the forecasted values as a Series
    forecast_series = pd.Series(forecasted_values, index=forecast_years)

    return forecast_series

def calculate_discounted_cashflow_from_lcoe(lcoe_df, generation_df, discount_rate):
    """
    Calculate discounted cashflows from pre-calculated LCOE values.
    
    Formula: Discounted Cashflow = LCOE × Discounted Generation
    
    Parameters:
    - lcoe_df: LCOE by fuel and year (currency/kWh)
    - generation_df: Generation by fuel and year (kWh)
    - discount_rate: Discount rate as decimal
    
    Returns:
    - Dictionary with annual costs, present values, and LCOE by fuel
    """
    start_year = lcoe_df.index[0]
    years = lcoe_df.index
    fuels = lcoe_df.columns
    
    # Initialize results
    annual_costs = pd.DataFrame(index=years, columns=fuels, dtype=float)
    present_values = pd.DataFrame(index=years, columns=fuels, dtype=float)
    discounted_generation = pd.DataFrame(index=years, columns=fuels, dtype=float)
    
    # Calculate costs and present values for each fuel and year
    for fuel in fuels:
        for year in years:
            # Get LCOE and generation for this fuel/year
            lcoe = lcoe_df.loc[year, fuel] if not pd.isna(lcoe_df.loc[year, fuel]) else 0
            generation = generation_df.loc[year, fuel] if not pd.isna(generation_df.loc[year, fuel]) else 0
            
            # Calculate annual cost = LCOE × generation
            annual_cost = lcoe * generation
            annual_costs.loc[year, fuel] = annual_cost
            
            # Calculate discount factor
            year_index = year - start_year
            discount_factor = (1 + discount_rate) ** year_index
            
            # Calculate discounted generation and present value
            disc_gen = generation / discount_factor
            discounted_generation.loc[year, fuel] = disc_gen
            present_values.loc[year, fuel] = annual_cost / discount_factor
    
    # Add grid totals (sum across fuels)
    annual_costs['grid'] = annual_costs.sum(axis=1, skipna=True)
    present_values['grid'] = present_values.sum(axis=1, skipna=True)
    discounted_generation['grid'] = discounted_generation.sum(axis=1, skipna=True)
    
    # Grid LCOE = weighted average by generation
    total_generation = generation_df.sum(axis=1, skipna=True)
    grid_lcoe = annual_costs['grid'] / total_generation
    
    return {
        'annual_costs': annual_costs,
        'present_values': present_values, 
        'discounted_generation': discounted_generation,
        'annual_lcoe': lcoe_df.copy(),  # Use the input LCOE directly
        'grid_lcoe': grid_lcoe,
        'total_npv': present_values.sum(),
        'total_generation': generation_df.sum().sum()
    }

def calculate_discounted_cashflow_direct(capacity_df, generation_df, capex_df, opex_df, 
                                       fuel_df, landcost_df, lifespan_df, discount_rate):
    """
    Calculate discounted cashflows directly from cost components, then derive LCOE.
    
    This is the DCF → LCOE approach (alternative to LCOE → DCF approach).
    
    Parameters:
    - capacity_df: Capacity by fuel and year (kW)
    - generation_df: Generation by fuel and year (kWh)
    - capex_df: CAPEX by fuel and year (currency/kW)
    - opex_df: Fixed OPEX by fuel and year (currency/kW/year)
    - fuel_df: Variable OPEX by fuel and year (currency/kWh)
    - landcost_df: Land cost by fuel and year (currency/kW/year)
    - lifespan_df: Operational lifespan by fuel and year (years)
    - discount_rate: Discount rate as decimal
    
    Returns:
    - Dictionary with annual costs, present values, and derived LCOE
    """
    start_year = capacity_df.index[0]
    years = capacity_df.index
    fuels = capacity_df.columns
    
    # Calculate vintage tracking
    delta_df = capacity_df.diff()
    delta_df.iloc[0] = capacity_df.iloc[0]
    wcapex_df = calculate_weighted_average(delta_df, capex_df)
    
    # Initialize results
    annual_costs = pd.DataFrame(index=years, columns=fuels, dtype=float)
    present_values = pd.DataFrame(index=years, columns=fuels, dtype=float)
    discounted_generation = pd.DataFrame(index=years, columns=fuels, dtype=float)
    
    # Calculate costs and present values for each fuel and year directly from components
    for fuel in fuels:
        for year in years:
            # Get cost components for this fuel/year
            capacity = capacity_df.loc[year, fuel] if not pd.isna(capacity_df.loc[year, fuel]) else 0
            generation = generation_df.loc[year, fuel] if not pd.isna(generation_df.loc[year, fuel]) else 0
            capex_per_kw = wcapex_df.loc[year, fuel] if not pd.isna(wcapex_df.loc[year, fuel]) else 0
            opex_per_kw = opex_df.loc[year, fuel] if not pd.isna(opex_df.loc[year, fuel]) else 0
            fuel_cost_per_kwh = fuel_df.loc[year, fuel] if not pd.isna(fuel_df.loc[year, fuel]) else 0
            land_cost_per_kw = landcost_df.loc[year, fuel] if not pd.isna(landcost_df.loc[year, fuel]) else 0
            lifespan = lifespan_df.loc[year, fuel] if not pd.isna(lifespan_df.loc[year, fuel]) else 0
            
            # Calculate annual cost components using the same methodology as LCOE calculation
            # This needs to match the traditional LCOE approach exactly
            
            # For proper DCF→LCOE calculation, we need to calculate the present value of ALL costs
            # over the entire project lifespan, not just annual cash outflows
            
            # Get the total cost using the same method as calculate_lcoe function
            if capacity > 0 and generation > 0 and lifespan > 0:
                # Calculate total costs over project lifespan using NPV approach
                total_cost = 0
                total_generation_pv = 0
                
                # Loop through project lifespan to calculate total PV costs and generation
                for project_year in range(1, int(lifespan) + 1):
                    # Discount factor for this project year
                    project_discount_factor = (1 + discount_rate) ** (project_year - 1)
                    
                    # Annual costs for this project year
                    if project_year == 1:
                        # CAPEX in year 1
                        annual_project_cost = (capacity * capex_per_kw + 
                                             capacity * opex_per_kw + 
                                             generation * fuel_cost_per_kwh + 
                                             capacity * land_cost_per_kw)
                    else:
                        # Only OPEX in subsequent years
                        annual_project_cost = (capacity * opex_per_kw + 
                                             generation * fuel_cost_per_kwh + 
                                             capacity * land_cost_per_kw)
                    
                    # Add to total PV costs and generation
                    total_cost += annual_project_cost / project_discount_factor
                    total_generation_pv += generation / project_discount_factor
                
                # Calculate the annual equivalent cost (reverse annuity)
                # This should match the LCOE × generation calculation
                if total_generation_pv > 0:
                    project_lcoe = total_cost / total_generation_pv
                    annual_cost = project_lcoe * generation
                else:
                    annual_cost = 0
            else:
                annual_cost = 0
            annual_costs.loc[year, fuel] = annual_cost
            
            # Calculate discount factor and present values
            year_index = year - start_year
            discount_factor = (1 + discount_rate) ** year_index
            
            # Calculate discounted values
            discounted_generation.loc[year, fuel] = generation / discount_factor
            present_values.loc[year, fuel] = annual_cost / discount_factor
    
    # Add grid totals (sum across fuels)
    annual_costs['grid'] = annual_costs.sum(axis=1, skipna=True)
    present_values['grid'] = present_values.sum(axis=1, skipna=True)
    discounted_generation['grid'] = discounted_generation.sum(axis=1, skipna=True)
    
    # Calculate LCOE from discounted cashflows
    # LCOE = Total Present Value of Costs / Total Present Value of Generation
    derived_lcoe = pd.DataFrame(index=years, columns=fuels, dtype=float)
    
    for fuel in fuels:
        total_pv_costs = present_values[fuel].sum()
        total_pv_generation = discounted_generation[fuel].sum()
        
        if total_pv_generation > 0:
            fuel_lcoe = total_pv_costs / total_pv_generation
            # Apply this LCOE to all years for this fuel
            derived_lcoe[fuel] = fuel_lcoe
        else:
            derived_lcoe[fuel] = 0
    
    # Grid LCOE
    total_grid_pv_costs = present_values['grid'].sum()
    total_grid_pv_generation = discounted_generation['grid'].sum()
    grid_lcoe_value = total_grid_pv_costs / total_grid_pv_generation if total_grid_pv_generation > 0 else 0
    
    grid_lcoe = pd.Series(index=years, data=grid_lcoe_value)
    
    return {
        'annual_costs': annual_costs,
        'present_values': present_values,
        'discounted_generation': discounted_generation,
        'derived_lcoe': derived_lcoe,
        'grid_lcoe': grid_lcoe,
        'total_npv': present_values.sum(),
        'total_generation': generation_df.sum().sum()
    }