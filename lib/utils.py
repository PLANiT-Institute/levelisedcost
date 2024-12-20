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

def calculate_lcoe(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, discount_rate, degradation=0, interest_rate=0, tax_rate=0):
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