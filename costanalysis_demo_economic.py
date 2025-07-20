import pandas as pd
import lib.utils as _utils

# Cost Analysis with Economic Lifespan - DEMONSTRATION VERSION
# Shows the impact of economic lifespan vs operational lifespan with modified values
# Economic lifespan = shorter period for CAPEX recovery
# Operational lifespan = full plant operating life

print("="*60)
print("ECONOMIC LIFESPAN DEMONSTRATION - MODIFIED VALUES")
print("="*60)

# Define analysis period
start_year = 2023
end_year = 2051

print(f"Analysis period: {start_year}-{end_year}")
print("Loading data...")

# Import data
data_dt = _utils.load_excel('data/grid.xlsx', start_year, end_year)

# Extract specific DataFrames
capacity_df = (data_dt['capacity'] * 1e3).round(0)                  # MW to kW
capex_df = data_dt['capex'].round(0)                                # krw / kW
generation_df = (data_dt['generation'] * 1e6).round(0)              # GWh to kWh
opex_df = data_dt['opex'].round(0)                                  # krw / kW
fuel_df = data_dt['fuelcost'].round(2)                              # krw / kWh
landcost_df = data_dt['landcost'].round(0)                          # krw / kW
lifespan_df = data_dt['lifespan'].round(0)                          # Operational lifespan
fuelemission_df = data_dt['fuelemission'].round(2)                  # kg-co2/kWh

# Create modified economic lifespan data for demonstration
# Economic lifespan = typical financing/depreciation period (shorter than operational life)
economic_lifespan_df = lifespan_df.copy()
economic_lifespan_df['nuclear'] = 10    # 10 years vs 60 years operational
economic_lifespan_df['coal'] = 15       # 15 years vs 30 years operational  
economic_lifespan_df['gas'] = 15        # 15 years vs 30 years operational
economic_lifespan_df['solar'] = 10      # 10 years vs 15 years operational
economic_lifespan_df['wind'] = 10       # 10 years vs 20 years operational

discount_rate = (data_dt['assumption'].loc['discount_rate'].values[0]/100).round(3) # percentage

print(f"Discount rate: {discount_rate*100:.1f}%")
print(f"Fuels analyzed: {list(capacity_df.columns)}")
print()

# Show economic vs operational lifespan comparison (MODIFIED VALUES FOR DEMO)
print("LIFESPAN COMPARISON - DEMONSTRATION VALUES (Years):")
print("Technology     Economic  Operational  Difference")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind']:
    if fuel in economic_lifespan_df.columns and fuel in lifespan_df.columns:
        econ_life = economic_lifespan_df[fuel].iloc[0]
        oper_life = lifespan_df[fuel].iloc[0]
        diff = oper_life - econ_life
        print(f"{fuel:10}     {econ_life:8.0f}     {oper_life:8.0f}       {diff:+3.0f}")
print()
print("Economic Lifespan = CAPEX recovery period (typical financing term)")
print("Operational Lifespan = Full plant operating life")
print("Economic approach = Lower LCOE (CAPEX spread over shorter period)")
print()

# =====================================================================
# 1. CALCULATE ECONOMIC LIFESPAN LCOE
# =====================================================================
print("1. CALCULATING ECONOMIC LIFESPAN LCOE...")

# Calculate the difference for vintage tracking
delta_df = capacity_df.diff()
delta_df.iloc[0] = capacity_df.iloc[0]

# Calculate weighted averaged capex (accounts for vintages)
wcapex_df = _utils.calculate_weighted_average(delta_df, capex_df)

# Create a new DataFrame for LCOE calculations
lcoe_economic_df = wcapex_df.copy()
lcoe_economic_df[:] = 0

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
        economic_lifespan = int(economic_lifespan_df.loc[idx, col])  # in years
        
        # Use economic lifespan LCOE calculation
        lcoe_economic_df.loc[idx, col] = _utils.calculate_lcoe_with_economic_lifespan(
            capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
            land_cost_per_mw, lifespan, economic_lifespan, discount_rate
        )

# Create a DataFrame to store grid LCOE calculations (weighted by generation)
gridlcoe_economic_df = pd.DataFrame({
    'grid': (lcoe_economic_df * generation_df).sum(axis=1, skipna=True) / generation_df.sum(axis=1, skipna=True),
})

print("✓ Economic lifespan LCOE calculation completed")

# =====================================================================
# 2. CALCULATE TRADITIONAL LCOE FOR COMPARISON
# =====================================================================
print("2. CALCULATING TRADITIONAL LCOE FOR COMPARISON...")

# Create a new DataFrame for traditional LCOE calculations
lcoe_traditional_df = wcapex_df.copy()
lcoe_traditional_df[:] = 0

# Loop through each row and column to calculate traditional LCOE
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
        
        # Use traditional LCOE calculation (CAPEX over full operational lifespan)
        lcoe_traditional_df.loc[idx, col] = _utils.calculate_lcoe(
            capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
            land_cost_per_mw, lifespan, discount_rate
        )

# Create a DataFrame to store grid LCOE calculations (weighted by generation)
gridlcoe_traditional_df = pd.DataFrame({
    'grid': (lcoe_traditional_df * generation_df).sum(axis=1, skipna=True) / generation_df.sum(axis=1, skipna=True),
})

print("✓ Traditional LCOE calculation completed")

# =====================================================================
# 3. CALCULATE ALIGNED DISCOUNTED CASHFLOWS FOR BOTH APPROACHES
# =====================================================================
print("3. CALCULATING ALIGNED DISCOUNTED CASHFLOWS...")

# Calculate discounted cashflows for economic lifespan approach
dcf_economic = _utils.calculate_discounted_cashflow_from_lcoe(
    lcoe_economic_df, generation_df, discount_rate
)

# Calculate discounted cashflows for traditional approach
dcf_traditional = _utils.calculate_discounted_cashflow_from_lcoe(
    lcoe_traditional_df, generation_df, discount_rate
)

print("✓ Discounted cashflow analysis completed for both approaches")

# =====================================================================
# 4. CALCULATE SUMMARY STATISTICS
# =====================================================================
print("4. CALCULATING SUMMARY STATISTICS...")

# Calculate system-wide weighted LCOE for both approaches
system_lcoe_economic = {}
system_lcoe_traditional = {}

for fuel in capacity_df.columns:
    fuel_generation = generation_df[fuel].sum()
    
    # Economic approach
    fuel_costs_economic = dcf_economic['annual_costs'][fuel].sum()
    if fuel_generation > 0:
        system_lcoe_economic[fuel] = fuel_costs_economic / fuel_generation
    else:
        system_lcoe_economic[fuel] = 0
    
    # Traditional approach
    fuel_costs_traditional = dcf_traditional['annual_costs'][fuel].sum()
    if fuel_generation > 0:
        system_lcoe_traditional[fuel] = fuel_costs_traditional / fuel_generation
    else:
        system_lcoe_traditional[fuel] = 0

# Grid system weighted LCOE
grid_generation = generation_df.sum().sum()
system_lcoe_economic['grid'] = dcf_economic['annual_costs']['grid'].sum() / grid_generation
system_lcoe_traditional['grid'] = dcf_traditional['annual_costs']['grid'].sum() / grid_generation

print("✓ Summary statistics completed")

# =====================================================================
# 5. GENERATE OUTPUTS
# =====================================================================
print("5. GENERATING OUTPUT FILES...")

# Save LCOE results for both approaches
lcoe_economic_df.to_csv('output/lcoe_demo_economic.csv', index=False)
lcoe_traditional_df.to_csv('output/lcoe_demo_traditional.csv', index=False)

# Save comprehensive Excel workbook
with pd.ExcelWriter('output/costanalysis_demo_economic.xlsx', engine='openpyxl') as writer:
    # Economic Lifespan Analysis
    lcoe_economic_df.to_excel(writer, sheet_name='LCOE Economic Lifespan', index=True)
    gridlcoe_economic_df.to_excel(writer, sheet_name='Grid LCOE Economic', index=True)
    dcf_economic['annual_costs'].to_excel(writer, sheet_name='Annual Costs Economic', index=True)
    dcf_economic['present_values'].to_excel(writer, sheet_name='Present Values Economic', index=True)
    
    # Traditional Analysis
    lcoe_traditional_df.to_excel(writer, sheet_name='LCOE Traditional', index=True)
    gridlcoe_traditional_df.to_excel(writer, sheet_name='Grid LCOE Traditional', index=True)
    dcf_traditional['annual_costs'].to_excel(writer, sheet_name='Annual Costs Traditional', index=True)
    dcf_traditional['present_values'].to_excel(writer, sheet_name='Present Values Traditional', index=True)
    
    # Comparison Analysis
    comparison_df = pd.DataFrame({
        'Economic Lifespan': system_lcoe_economic,
        'Traditional': system_lcoe_traditional
    })
    comparison_df['Difference (%)'] = ((comparison_df['Economic Lifespan'] - comparison_df['Traditional']) / comparison_df['Traditional'] * 100).round(1)
    comparison_df.to_excel(writer, sheet_name='LCOE Comparison', index=True)
    
    # Modified lifespan data for reference
    modified_comparison = pd.DataFrame({
        'Economic Lifespan (Years)': economic_lifespan_df.iloc[0],
        'Operational Lifespan (Years)': lifespan_df.iloc[0]
    })
    modified_comparison['Difference (Years)'] = modified_comparison['Operational Lifespan (Years)'] - modified_comparison['Economic Lifespan (Years)']
    modified_comparison.to_excel(writer, sheet_name='Lifespan Comparison', index=True)
    
    # Input Data
    capacity_df.to_excel(writer, sheet_name='Capacity (kW)', index=True)
    generation_df.to_excel(writer, sheet_name='Generation (kWh)', index=True)
    capex_df.to_excel(writer, sheet_name='CAPEX (KRW per kW)', index=True)

print("✓ Output files generated")

# =====================================================================
# 6. VERIFICATION: Check LCOE-Cashflow Alignment for Both Approaches
# =====================================================================
print("6. VERIFYING LCOE-CASHFLOW ALIGNMENT...")

# Verify economic lifespan approach
verification_costs_economic = lcoe_economic_df * generation_df
max_difference_economic = (dcf_economic['annual_costs'] - verification_costs_economic).abs().max().max()

# Verify traditional approach
verification_costs_traditional = lcoe_traditional_df * generation_df
max_difference_traditional = (dcf_traditional['annual_costs'] - verification_costs_traditional).abs().max().max()

print(f"Economic Lifespan - Max difference: {max_difference_economic:.2e}")
print(f"Traditional       - Max difference: {max_difference_traditional:.2e}")

if max_difference_economic < 1e-10 and max_difference_traditional < 1e-10:
    print("✓ Both approaches are perfectly aligned with their cashflows")
else:
    print("⚠ Warning: Alignment issues detected")

# =====================================================================
# 7. PRINT SUMMARY REPORT
# =====================================================================
print("\n" + "="*60)
print("ECONOMIC LIFESPAN DEMONSTRATION RESULTS")
print("="*60)

print("\nSYSTEM-WIDE LCOE COMPARISON (KRW/kWh) - Planning Period Average:")
print("Technology     Economic  Traditional  Difference")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind', 'grid']:
    if fuel in system_lcoe_economic and fuel in system_lcoe_traditional:
        econ = system_lcoe_economic[fuel]
        trad = system_lcoe_traditional[fuel]
        diff_pct = ((econ - trad) / trad * 100) if trad > 0 else 0
        print(f"{fuel:10}     {econ:8.1f}     {trad:8.1f}     {diff_pct:+6.1f}%")

print(f"\nTOTAL SYSTEM NPV COMPARISON:")
print(f"Economic Lifespan: {dcf_economic['total_npv']['grid']:,.0f} KRW")
print(f"Traditional:       {dcf_traditional['total_npv']['grid']:,.0f} KRW")
npv_diff = ((dcf_economic['total_npv']['grid'] - dcf_traditional['total_npv']['grid']) / dcf_traditional['total_npv']['grid'] * 100)
print(f"Difference:        {npv_diff:+.1f}%")

print(f"\nGRID LCOE COMPARISON (2023 vs 2050):")
print("Year    Economic  Traditional  Economic Advantage")
first_year_econ = gridlcoe_economic_df['grid'].iloc[0]
first_year_trad = gridlcoe_traditional_df['grid'].iloc[0]
last_year_econ = gridlcoe_economic_df['grid'].iloc[-1]
last_year_trad = gridlcoe_traditional_df['grid'].iloc[-1]

first_adv = ((first_year_trad - first_year_econ) / first_year_trad * 100)
last_adv = ((last_year_trad - last_year_econ) / last_year_trad * 100)

print(f"2023    {first_year_econ:8.1f}     {first_year_trad:8.1f}       {first_adv:+6.1f}%")
print(f"2050    {last_year_econ:8.1f}     {last_year_trad:8.1f}       {last_adv:+6.1f}%")

print("\nKEY INSIGHTS:")
print(f"• Economic lifespan approach results in {abs(npv_diff):.1f}% {'lower' if npv_diff < 0 else 'higher'} total system costs")
print(f"• Nuclear shows the largest difference due to 10-year vs 60-year CAPEX recovery")
print(f"• Wind and solar also show significant differences with 10-year CAPEX recovery")
print(f"• Economic approach favors technologies with long operational but short economic lives")

print("\n" + "="*60)
print("FILES GENERATED:")
print(f"  • output/lcoe_demo_economic.csv           - Economic lifespan LCOE")
print(f"  • output/lcoe_demo_traditional.csv        - Traditional LCOE")  
print(f"  • output/costanalysis_demo_economic.xlsx  - Comprehensive comparison")
print("="*60)
print("ECONOMIC LIFESPAN DEMONSTRATION COMPLETE")
print("Note: This uses modified economic lifespan values for demonstration")
print("="*60)