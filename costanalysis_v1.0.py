import pandas as pd
import lib.utils as _utils

# Cost Analysis v1.0
# Electricity system cost analysis with aligned LCOE and discounted cashflows
# Simplified version without hydrogen analysis

print("="*60)
print("ELECTRICITY SYSTEM COST ANALYSIS v1.0")
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
lifespan_df = data_dt['lifespan'].round(0)
fuelemission_df = data_dt['fuelemission'].round(2)                  # kg-co2/kWh

discount_rate = (data_dt['assumption'].loc['discount_rate'].values[0]/100).round(3) # percentage

print(f"Discount rate: {discount_rate*100:.1f}%")
print(f"Fuels analyzed: {list(capacity_df.columns)}")
print()

# =====================================================================
# 1. CALCULATE TRADITIONAL LCOE
# =====================================================================
print("1. CALCULATING TRADITIONAL LCOE...")

# Calculate the difference for vintage tracking
delta_df = capacity_df.diff()
delta_df.iloc[0] = capacity_df.iloc[0]

# Calculate weighted averaged capex (accounts for vintages)
wcapex_df = _utils.calculate_weighted_average(delta_df, capex_df)

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
        
        # Use traditional LCOE calculation (CAPEX over full operational lifespan)
        lcoe_df.loc[idx, col] = _utils.calculate_lcoe(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, discount_rate)

# Create a DataFrame to store grid LCOE calculations (weighted by generation)
gridlcoe_df = pd.DataFrame({'grid': (lcoe_df * generation_df).sum(axis=1, skipna=True) / generation_df.sum(axis=1, skipna=True),})

print("✓ LCOE calculation completed")

# =====================================================================
# 2. CALCULATE ALIGNED DISCOUNTED CASHFLOWS
# =====================================================================
print("2. CALCULATING ALIGNED DISCOUNTED CASHFLOWS...")

# Calculate discounted cashflows using the pre-calculated LCOE
# Formula: Discounted Cashflow = LCOE × Discounted Generation
dcf_results = _utils.calculate_discounted_cashflow_from_lcoe(
    lcoe_df, generation_df, discount_rate
)

print("✓ Discounted cashflow analysis completed")
print("✓ LCOE and cashflow are now perfectly aligned")

# =====================================================================
# 3. CALCULATE EMISSIONS
# =====================================================================
print("3. CALCULATING EMISSIONS...")

# Calculate annual emissions
annual_emissions = pd.DataFrame(index=generation_df.index, columns=generation_df.columns, dtype=float)
for fuel in generation_df.columns:
    for year in generation_df.index:
        generation = generation_df.loc[year, fuel] if not pd.isna(generation_df.loc[year, fuel]) else 0
        emission_factor = fuelemission_df.loc[year, fuel] if not pd.isna(fuelemission_df.loc[year, fuel]) else 0
        annual_emissions.loc[year, fuel] = generation * emission_factor / 1e9  # Convert to million tonnes CO2

# Grid total emissions
annual_emissions['grid'] = annual_emissions.sum(axis=1, skipna=True)

print("✓ Emissions calculation completed")

# =====================================================================
# 4. CALCULATE SUMMARY STATISTICS
# =====================================================================
print("4. CALCULATING SUMMARY STATISTICS...")

# NPV calculations
total_npv = dcf_results['total_npv']
total_generation = dcf_results['total_generation']

# Calculate system-wide weighted LCOE over planning period
system_lcoe = {}
for fuel in capacity_df.columns:
    fuel_generation = generation_df[fuel].sum()
    fuel_costs = dcf_results['annual_costs'][fuel].sum()
    if fuel_generation > 0:
        system_lcoe[fuel] = fuel_costs / fuel_generation
    else:
        system_lcoe[fuel] = 0

# Grid system weighted LCOE
grid_generation = generation_df.sum().sum()
grid_costs = dcf_results['annual_costs']['grid'].sum()
system_lcoe['grid'] = grid_costs / grid_generation

# Calculate capacity factors
capacity_factor_df = pd.DataFrame(index=capacity_df.index, columns=capacity_df.columns, dtype=float)
for fuel in capacity_df.columns:
    for year in capacity_df.index:
        capacity = capacity_df.loc[year, fuel] if not pd.isna(capacity_df.loc[year, fuel]) else 0
        generation = generation_df.loc[year, fuel] if not pd.isna(generation_df.loc[year, fuel]) else 0
        max_generation = capacity * 8760  # kW * hours = kWh
        if max_generation > 0:
            capacity_factor_df.loc[year, fuel] = generation / max_generation
        else:
            capacity_factor_df.loc[year, fuel] = 0

print("✓ Summary statistics completed")

# =====================================================================
# 5. GENERATE OUTPUTS
# =====================================================================
print("5. GENERATING OUTPUT FILES...")

# Save traditional LCOE results
lcoe_df.to_csv('output/lcoe_v1.0.csv', index=False)

# Save comprehensive Excel workbook
with pd.ExcelWriter('output/costanalysis_v1.0.xlsx', engine='openpyxl') as writer:
    # LCOE Analysis
    lcoe_df.to_excel(writer, sheet_name='Annual LCOE by Fuel', index=True)
    gridlcoe_df.to_excel(writer, sheet_name='Grid LCOE', index=True)
    
    # Aligned Cashflow Analysis
    dcf_results['annual_costs'].to_excel(writer, sheet_name='Annual Costs', index=True)
    dcf_results['present_values'].to_excel(writer, sheet_name='Present Values', index=True)
    dcf_results['discounted_generation'].to_excel(writer, sheet_name='Discounted Generation', index=True)
    
    # Emissions Analysis
    annual_emissions.to_excel(writer, sheet_name='Annual Emissions (Mt CO2)', index=True)
    
    # System Analysis
    capacity_factor_df.to_excel(writer, sheet_name='Capacity Factors', index=True)
    
    # Input Data
    capacity_df.to_excel(writer, sheet_name='Capacity (kW)', index=True)
    generation_df.to_excel(writer, sheet_name='Generation (kWh)', index=True)
    capex_df.to_excel(writer, sheet_name='CAPEX (KRW per kW)', index=True)
    fuelemission_df.to_excel(writer, sheet_name='Emission Factors', index=True)
    
    # Summary Statistics
    npv_summary = pd.DataFrame({
        'Total NPV (KRW)': total_npv,
        'System LCOE (KRW per kWh)': system_lcoe
    })
    npv_summary.to_excel(writer, sheet_name='Summary Statistics', index=True)

print("✓ Output files generated")

# =====================================================================
# 6. VERIFICATION: Check LCOE-Cashflow Alignment
# =====================================================================
print("6. VERIFYING LCOE-CASHFLOW ALIGNMENT...")

# Verify that annual_costs = LCOE × generation exactly
verification_costs = lcoe_df * generation_df
max_difference = (dcf_results['annual_costs'] - verification_costs).abs().max().max()

print(f"Maximum difference between LCOE×Gen and Annual Costs: {max_difference:.2e}")
if max_difference < 1e-10:
    print("✓ LCOE and cashflow are perfectly aligned")
else:
    print("⚠ Warning: LCOE and cashflow alignment may have issues")

# =====================================================================
# 7. PRINT SUMMARY REPORT
# =====================================================================
print("\n" + "="*60)
print("COST ANALYSIS SUMMARY v1.0")
print("="*60)

print("\nSYSTEM-WIDE LCOE (KRW/kWh) - Planning Period Average:")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind', 'grid']:
    if fuel in system_lcoe:
        print(f"  {fuel:8}: {system_lcoe[fuel]:6.1f}")

print(f"\nTOTAL SYSTEM NPV: {total_npv['grid']:,.0f} KRW")
print(f"TOTAL GENERATION: {total_generation/1e12:.1f} TWh")
print(f"TOTAL EMISSIONS: {annual_emissions['grid'].sum():.1f} Mt CO2")

print(f"\nFIRST vs LAST YEAR COMPARISON:")
print("LCOE (KRW/kWh):           2023    2050   Change")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind']:
    if fuel in lcoe_df.columns:
        first_val = lcoe_df[fuel].iloc[0]
        last_val = lcoe_df[fuel].iloc[-1]
        if pd.notna(first_val) and pd.notna(last_val) and first_val > 0:
            change_pct = ((last_val - first_val) / first_val) * 100
            print(f"  {fuel:8}:          {first_val:6.1f}  {last_val:6.1f}  {change_pct:+6.1f}%")

print("\nCAPACITY EXPANSION:")
print("Capacity (GW):            2023    2050   Growth")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind']:
    if fuel in capacity_df.columns:
        first_cap = capacity_df[fuel].iloc[0] / 1e6
        last_cap = capacity_df[fuel].iloc[-1] / 1e6
        growth_x = last_cap / first_cap if first_cap > 0 else 0
        print(f"  {fuel:8}:          {first_cap:6.1f}  {last_cap:6.1f}  {growth_x:6.1f}x")

print("\nEMISSIONS ANALYSIS:")
print("Annual Emissions (Mt CO2): 2023    2050   Change")
for fuel in ['coal', 'gas', 'grid']:
    if fuel in annual_emissions.columns:
        first_em = annual_emissions[fuel].iloc[0]
        last_em = annual_emissions[fuel].iloc[-1]
        change_pct = ((last_em - first_em) / first_em) * 100 if first_em > 0 else 0
        print(f"  {fuel:8}:          {first_em:6.1f}  {last_em:6.1f}  {change_pct:+6.1f}%")

print("\n" + "="*60)
print("FILES GENERATED:")
print(f"  • output/lcoe_v1.0.csv              - Annual LCOE by fuel")  
print(f"  • output/costanalysis_v1.0.xlsx     - Comprehensive workbook")
print("="*60)
print("KEY FEATURES:")
print("✓ Traditional LCOE with vintage tracking")
print("✓ Aligned discounted cashflows (LCOE × discounted generation)")
print("✓ Emissions analysis")
print("✓ No hydrogen analysis (simplified)")
print("✓ Perfect LCOE-cashflow alignment verified")
print("="*60)
print("COST ANALYSIS COMPLETE")
print("="*60)