import pandas as pd
import lib.utils as _utils

# Cost Analysis - LCOE vs DCF Approach Comparison
# Compares two calculation methodologies:
# 1. LCOE → DCF: Calculate LCOE first, then derive discounted cashflows
# 2. DCF → LCOE: Calculate discounted cashflows directly, then derive LCOE

print("="*60)
print("LCOE vs DCF APPROACH COMPARISON")
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
# APPROACH 1: LCOE → DCF (Calculate LCOE first, then derive cashflows)
# =====================================================================
print("APPROACH 1: LCOE → DCF")
print("1a. CALCULATING LCOE FIRST...")

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
        
        # Use traditional LCOE calculation
        lcoe_df.loc[idx, col] = _utils.calculate_lcoe(capacity, generation, capex_per_mw, fixed_opex_per_mw, variable_opex,
                   land_cost_per_mw, lifespan, discount_rate)

print("✓ LCOE calculation completed")

print("1b. DERIVING DISCOUNTED CASHFLOWS FROM LCOE...")

# Calculate discounted cashflows using the pre-calculated LCOE
# Formula: Discounted Cashflow = LCOE × Discounted Generation
dcf_from_lcoe = _utils.calculate_discounted_cashflow_from_lcoe(
    lcoe_df, generation_df, discount_rate
)

print("✓ DCF derived from LCOE completed")

# =====================================================================
# APPROACH 2: DCF → LCOE (Calculate DCF directly, then derive LCOE)
# =====================================================================
print("\nAPPROACH 2: DCF → LCOE")
print("2a. CALCULATING DISCOUNTED CASHFLOWS DIRECTLY...")

# Calculate discounted cashflows directly from cost components
dcf_direct = _utils.calculate_discounted_cashflow_direct(
    capacity_df, generation_df, capex_df, opex_df, 
    fuel_df, landcost_df, lifespan_df, discount_rate
)

print("✓ DCF calculation completed")
print("✓ LCOE derived from DCF completed")

# =====================================================================
# APPROACH COMPARISON AND VERIFICATION
# =====================================================================
print("\nCOMPARING BOTH APPROACHES...")

# Compare annual costs
annual_costs_diff = (dcf_from_lcoe['annual_costs'] - dcf_direct['annual_costs']).abs()
max_annual_diff = annual_costs_diff.max().max()

# Compare present values  
pv_diff = (dcf_from_lcoe['present_values'] - dcf_direct['present_values']).abs()
max_pv_diff = pv_diff.max().max()

# Compare discounted generation
gen_diff = (dcf_from_lcoe['discounted_generation'] - dcf_direct['discounted_generation']).abs()
max_gen_diff = gen_diff.max().max()

# Compare LCOE values
# For approach 1: Use original LCOE calculation
# For approach 2: Use derived LCOE from DCF
lcoe_approach1 = lcoe_df
lcoe_approach2 = dcf_direct['derived_lcoe']
lcoe_diff = (lcoe_approach1 - lcoe_approach2).abs()
max_lcoe_diff = lcoe_diff.max().max()

print(f"Maximum difference in annual costs: {max_annual_diff:.2e}")
print(f"Maximum difference in present values: {max_pv_diff:.2e}")
print(f"Maximum difference in discounted generation: {max_gen_diff:.2e}")
print(f"Maximum difference in LCOE values: {max_lcoe_diff:.2e}")

# =====================================================================
# CALCULATE SUMMARY STATISTICS
# =====================================================================
print("\nCALCULATING SUMMARY STATISTICS...")

# System-wide weighted LCOE for both approaches
system_lcoe_approach1 = {}
system_lcoe_approach2 = {}

for fuel in capacity_df.columns:
    fuel_generation = generation_df[fuel].sum()
    
    # Approach 1: LCOE → DCF
    fuel_costs_1 = dcf_from_lcoe['annual_costs'][fuel].sum()
    if fuel_generation > 0:
        system_lcoe_approach1[fuel] = fuel_costs_1 / fuel_generation
    else:
        system_lcoe_approach1[fuel] = 0
    
    # Approach 2: DCF → LCOE
    fuel_costs_2 = dcf_direct['annual_costs'][fuel].sum()
    if fuel_generation > 0:
        system_lcoe_approach2[fuel] = fuel_costs_2 / fuel_generation
    else:
        system_lcoe_approach2[fuel] = 0

# Grid system weighted LCOE
grid_generation = generation_df.sum().sum()
system_lcoe_approach1['grid'] = dcf_from_lcoe['annual_costs']['grid'].sum() / grid_generation
system_lcoe_approach2['grid'] = dcf_direct['annual_costs']['grid'].sum() / grid_generation

print("✓ Summary statistics completed")

# =====================================================================
# GENERATE OUTPUTS
# =====================================================================
print("\nGENERATING OUTPUT FILES...")

# Save comprehensive Excel workbook
with pd.ExcelWriter('output/costanalysis_comparison.xlsx', engine='openpyxl') as writer:
    # Approach 1: LCOE → DCF
    lcoe_df.to_excel(writer, sheet_name='Approach1_LCOE', index=True)
    dcf_from_lcoe['annual_costs'].to_excel(writer, sheet_name='Approach1_Annual_Costs', index=True)
    dcf_from_lcoe['present_values'].to_excel(writer, sheet_name='Approach1_Present_Values', index=True)
    dcf_from_lcoe['discounted_generation'].to_excel(writer, sheet_name='Approach1_Disc_Generation', index=True)
    
    # Approach 2: DCF → LCOE
    dcf_direct['derived_lcoe'].to_excel(writer, sheet_name='Approach2_LCOE', index=True)
    dcf_direct['annual_costs'].to_excel(writer, sheet_name='Approach2_Annual_Costs', index=True)
    dcf_direct['present_values'].to_excel(writer, sheet_name='Approach2_Present_Values', index=True)
    dcf_direct['discounted_generation'].to_excel(writer, sheet_name='Approach2_Disc_Generation', index=True)
    
    # Comparison Analysis
    annual_costs_diff.to_excel(writer, sheet_name='Diff_Annual_Costs', index=True)
    pv_diff.to_excel(writer, sheet_name='Diff_Present_Values', index=True)
    lcoe_diff.to_excel(writer, sheet_name='Diff_LCOE', index=True)
    
    # Summary Comparison
    comparison_summary = pd.DataFrame({
        'LCOE→DCF Approach': system_lcoe_approach1,
        'DCF→LCOE Approach': system_lcoe_approach2
    })
    comparison_summary['Absolute Difference'] = (comparison_summary['LCOE→DCF Approach'] - comparison_summary['DCF→LCOE Approach']).abs()
    comparison_summary['Relative Difference (%)'] = (comparison_summary['Absolute Difference'] / comparison_summary['LCOE→DCF Approach'] * 100).round(6)
    comparison_summary.to_excel(writer, sheet_name='Summary_Comparison', index=True)
    
    # Input Data
    capacity_df.to_excel(writer, sheet_name='Input_Capacity', index=True)
    generation_df.to_excel(writer, sheet_name='Input_Generation', index=True)
    capex_df.to_excel(writer, sheet_name='Input_CAPEX', index=True)

# Save individual CSV files for easy comparison
lcoe_df.to_csv('output/lcoe_approach1.csv', index=False)
dcf_direct['derived_lcoe'].to_csv('output/lcoe_approach2.csv', index=False)
dcf_from_lcoe['annual_costs'].to_csv('output/annual_costs_approach1.csv', index=False)
dcf_direct['annual_costs'].to_csv('output/annual_costs_approach2.csv', index=False)

print("✓ Output files generated")

# =====================================================================
# VERIFICATION AND FINAL REPORT
# =====================================================================
print("\n" + "="*60)
print("APPROACH COMPARISON RESULTS")
print("="*60)

print("\nMETHODOLOGY VERIFICATION:")
print("Approach 1 (LCOE→DCF): Calculate LCOE using individual cost components,")
print("                       then derive cashflows as LCOE × Generation")
print("Approach 2 (DCF→LCOE): Calculate annual costs directly from components,")
print("                       then derive LCOE as Total PV Costs / Total PV Generation")

print(f"\nNUMERICAL ALIGNMENT CHECK:")
print(f"Max difference in annual costs:      {max_annual_diff:.2e}")
print(f"Max difference in present values:    {max_pv_diff:.2e}")
print(f"Max difference in LCOE values:       {max_lcoe_diff:.2e}")

if max_annual_diff < 1e-6 and max_pv_diff < 1e-6 and max_lcoe_diff < 1e-6:
    print("✓ Both approaches are numerically equivalent")
    alignment_status = "PERFECTLY ALIGNED"
else:
    print("⚠ Approaches show numerical differences")
    alignment_status = "DIFFERENCES DETECTED"

print(f"\nSYSTEM-WIDE LCOE COMPARISON (KRW/kWh):")
print("Technology     LCOE→DCF  DCF→LCOE  Abs.Diff  Rel.Diff(%)")
for fuel in ['nuclear', 'coal', 'gas', 'solar', 'wind', 'grid']:
    if fuel in system_lcoe_approach1 and fuel in system_lcoe_approach2:
        lcoe1 = system_lcoe_approach1[fuel]
        lcoe2 = system_lcoe_approach2[fuel]
        abs_diff = abs(lcoe1 - lcoe2)
        rel_diff = (abs_diff / lcoe1 * 100) if lcoe1 > 0 else 0
        print(f"{fuel:10}     {lcoe1:8.1f}  {lcoe2:8.1f}  {abs_diff:8.3f}  {rel_diff:9.6f}")

print(f"\nTOTAL SYSTEM NPV COMPARISON:")
print(f"LCOE→DCF approach: {dcf_from_lcoe['total_npv']['grid']:,.0f} KRW")
print(f"DCF→LCOE approach: {dcf_direct['total_npv']['grid']:,.0f} KRW")
npv_diff_abs = abs(dcf_from_lcoe['total_npv']['grid'] - dcf_direct['total_npv']['grid'])
npv_diff_rel = (npv_diff_abs / dcf_from_lcoe['total_npv']['grid'] * 100)
print(f"Absolute difference: {npv_diff_abs:,.0f} KRW")
print(f"Relative difference: {npv_diff_rel:.6f}%")

print("\n" + "="*60)
print("FILES GENERATED:")
print(f"  • output/costanalysis_comparison.xlsx  - Comprehensive comparison workbook")
print(f"  • output/lcoe_approach1.csv            - LCOE from approach 1")
print(f"  • output/lcoe_approach2.csv            - LCOE from approach 2")
print(f"  • output/annual_costs_approach1.csv    - Annual costs from approach 1")
print(f"  • output/annual_costs_approach2.csv    - Annual costs from approach 2")
print("="*60)
print("KEY FINDINGS:")
print(f"✓ Approach 1: LCOE → DCF (traditional method)")
print(f"✓ Approach 2: DCF → LCOE (alternative method)")
print(f"✓ Alignment Status: {alignment_status}")
if max_lcoe_diff < 1e-6:
    print(f"✓ Both approaches yield identical LCOE results")
else:
    print(f"⚠ LCOE differences detected - review methodology")
print("="*60)
print("APPROACH COMPARISON COMPLETE")
print("="*60)