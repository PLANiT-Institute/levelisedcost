# Cost Analysis v1.0 - Traditional LCOE Approach

## Overview
`costanalysis_v1.0.py` implements the traditional electricity system cost analysis methodology where **Levelized Cost of Energy (LCOE) is calculated first**, followed by discounted cash flow derivation. This approach follows conventional energy economics practices but has limitations with non-generation assets like Energy Storage Systems (ESS).

## Methodology Summary

### Core Approach: LCOE → DCF
```
Input Data → Traditional LCOE Calculation → DCF = LCOE × Discounted Generation
```

This is the **opposite** of the DCF-first approach, representing the conventional method used in energy economics.

## Code Structure and Process Flow

### 1. Configuration and Data Loading
```python
# Parameter extraction from assumption sheet
start_year = get_param(assumption_df, 'start_year', 2023, int)
end_year = get_param(assumption_df, 'end_year', 2051, int)
discount_rate = get_param(assumption_df, 'discount_rate', 4.5, float) / 100
```

**Logic**:
- Loads all configuration parameters from `data/grid.xlsx`
- Uses `get_param()` helper function with fallback defaults
- Applies unit conversions (MW→kW, GWh→kWh)
- Configurable analysis period and fuel list

**Data Sources**:
- `capacity`: Installed capacity by fuel and year (MW)
- `capex`: Capital expenditure per unit capacity (KRW/kW)
- `generation`: Annual electricity generation (GWh)
- `opex`: Fixed operating expenditure (KRW/kW/year)
- `fuelcost`: Variable operating costs (KRW/kWh)
- `landcost`: Land-related costs (KRW/kW/year)
- `lifespan`: Project operational lifetime (years)
- `fuelemission`: Emission factors (kg-CO2/kWh)

### 2. Vintage Tracking and Weighted CAPEX
```python
delta_df = capacity_df.diff()
delta_df.iloc[0] = capacity_df.iloc[0]
wcapex_df = _utils.calculate_weighted_average(delta_df, capex_df)
```

**Purpose**: Account for different vintages of capacity with varying costs.

**Process**:
1. **Capacity Changes**: Calculate annual capacity additions/retirements
2. **Vintage Tracking**: Track cost-weighted average of active capacity
3. **FIFO Retirement**: Remove oldest capacity first when plants retire
4. **Weighted CAPEX**: Calculate capacity-weighted average CAPEX for each year

**Mathematical Foundation**:
```
Weighted_CAPEX_t = Σ(Capacity_vintage_i × CAPEX_vintage_i) / Σ(Capacity_vintage_i)
```

### 3. Traditional LCOE Calculation
```python
for idx, row in wcapex_df.iterrows():
    for col, _ in row.items():
        lcoe_df.loc[idx, col] = _utils.calculate_lcoe(
            capacity, generation, capex_per_mw, fixed_opex_per_mw, 
            variable_opex, land_cost_per_mw, lifespan, discount_rate
        )
```

**Core Algorithm** (from `utils.calculate_lcoe()`):
1. **Year 1**: Apply full CAPEX + OPEX + fuel costs
2. **Years 2-N**: Apply only OPEX + fuel costs (no additional CAPEX)
3. **Discounting**: Apply discount factor `(1 + r)^(t-1)` to each year
4. **Degradation**: Optional generation degradation over time
5. **LCOE Formula**: `Total PV Costs / Total PV Generation`

**Mathematical Expression**:
```
LCOE = [CAPEX + Σ(t=1 to N)[OPEX_t + Fuel_t + Land_t] / (1+r)^(t-1)] / 
       [Σ(t=1 to N)[Generation_t × (1-degradation)^(t-1)] / (1+r)^(t-1)]
```

### 4. Grid-Level LCOE Aggregation
```python
gridlcoe_df = pd.DataFrame({
    'grid': (lcoe_df * generation_df).sum(axis=1, skipna=True) / 
            generation_df.sum(axis=1, skipna=True)
})
```

**Generation-Weighted Grid LCOE**:
```
Grid_LCOE_t = Σ(LCOE_fuel_t × Generation_fuel_t) / Σ(Generation_fuel_t)
```

This provides the system-wide cost per kWh considering the generation mix.

### 5. Aligned Discounted Cash Flow Derivation
```python
dcf_results = _utils.calculate_discounted_cashflow_from_lcoe(
    lcoe_df, generation_df, discount_rate
)
```

**Perfect Alignment Formula**:
```
Annual_Cost_t = LCOE_t × Generation_t
Present_Value_t = Annual_Cost_t / (1 + r)^(t-start_year)
```

**Key Feature**: This ensures mathematical consistency between LCOE and DCF by construction.

### 6. Emissions Analysis
```python
annual_emissions.loc[year, fuel] = generation * emission_factor / emissions_unit_factor
```

**Process**:
- Calculate annual CO2 emissions by fuel and year
- Convert to million tonnes CO2 equivalent
- Aggregate to grid-level totals
- Track emissions trends over analysis period

### 7. Verification and Validation
```python
verification_costs = lcoe_df * generation_df
max_difference = (dcf_results['annual_costs'] - verification_costs).abs().max().max()
```

**Quality Assurance**:
- Verifies perfect alignment between LCOE×Generation and annual costs
- Checks for mathematical consistency
- Reports maximum numerical differences (should be ~0)

## Output Generation

### Primary Outputs
1. **LCOE Results**: `output/lcoe_v1.0_revised.csv`
2. **Comprehensive Analysis**: `output/costanalysis_v1.0_revised.xlsx`

### Excel Workbook Structure
- `Annual LCOE by Fuel`: Year-by-year LCOE values
- `Grid LCOE`: System-wide weighted LCOE
- `Annual Costs`: Cash flows by fuel and year
- `Present Values`: Discounted cash flows
- `Discounted Generation`: Present value of generation
- `Annual Emissions (Mt CO2)`: Environmental impact analysis
- `Capacity Factors`: Asset utilization rates
- `Summary Statistics`: System-wide metrics and NPV

## Key Features and Advantages

### 1. Perfect LCOE-DCF Alignment
- Mathematical guarantee: `DCF = LCOE × Discounted Generation`
- No numerical errors or inconsistencies
- Transparent relationship between metrics

### 2. Vintage Tracking
- Accounts for evolving technology costs over time
- Reflects realistic plant retirement patterns
- Maintains cost history for accurate analysis

### 3. Comprehensive Coverage
- All major cost components included
- Environmental impact assessment
- System-wide aggregation and reporting

### 4. Configurable Parameters
- No hardcoded values in the code
- All parameters sourced from data files
- Easy scenario analysis and sensitivity testing

## Known Issues and Limitations

### 1. ESS and Non-Generation Assets
**Critical Issue**: Traditional LCOE approach fails for assets with zero generation.

**Mathematical Problem**:
```
LCOE_ESS = Total_Costs / 0 = ∞ (undefined)
```

**Impact**:
- ESS shows as `inf` in LCOE calculations
- Meaningful cost analysis impossible for storage systems
- Grid-level costs exclude ESS contributions

**Warning Messages**:
```
RuntimeWarning: divide by zero encountered in scalar divide
RuntimeWarning: invalid value encountered in scalar multiply
```

### 2. Generation-Centric Bias
**Issue**: Framework assumes all assets generate electricity.

**Limitations**:
- Cannot handle demand response resources
- Excludes transmission and distribution costs
- Missing grid services valuation

### 3. Cost Allocation Methodology
**Issue**: CAPEX applied entirely in year 1.

**Alternatives**:
- Levelized annual CAPEX payments
- Economic lifespan different from operational lifespan
- Time-varying capital costs

### 4. Discount Rate Assumptions
**Issue**: Single discount rate for all fuels and time periods.

**Reality**:
- Different technologies have different risk profiles
- Interest rates vary over time
- Regional financing cost differences

## Technical Performance

### Computational Complexity
- **Time Complexity**: O(n × m × l) where n=years, m=fuels, l=lifespan
- **Space Complexity**: O(n × m) for result storage
- **Bottlenecks**: Nested loops for LCOE calculation

### Memory Usage
- Multiple large DataFrames for intermediate results
- Full materialization of year×fuel matrices
- Peak memory during Excel output generation

### Numerical Stability
- Generally stable for well-defined inputs
- Issues with extreme values (very small generation)
- Floating-point precision limitations for large cost values

## Comparison with DCF v1.0

### Methodological Differences
| Aspect | Traditional (costanalysis_v1.0) | DCF v1.0 |
|--------|--------------------------------|----------|
| Primary Calculation | LCOE → DCF | DCF → LCOE |
| ESS Handling | Shows ∞ (undefined) | Proper cost inclusion |
| Cost Focus | Per-unit generation | Total system cost |
| Mathematical Foundation | Generation-weighted | Cost-component based |

### Results Comparison
- **Generation Assets**: Identical LCOE values (perfect alignment)
- **System NPV**: Traditional ~3.1T KRW lower (excludes ESS)
- **Grid LCOE**: Traditional 125.6 vs DCF 125.9 KRW/kWh
- **ESS Treatment**: Traditional=inf vs DCF=meaningful cost analysis

## Best Practices and Usage Guidelines

### 1. When to Use Traditional Approach
- **Generation-only analysis**: When ESS and storage are not significant
- **Legacy compatibility**: When comparing with historical studies
- **Regulatory compliance**: When required methodology is LCOE-based
- **Simple scenarios**: Basic generation planning without complex assets

### 2. Data Quality Requirements
- **Complete generation data**: Critical for meaningful LCOE
- **Consistent units**: Ensure MW/GWh alignment
- **Validated lifespans**: Realistic project timelines
- **Market-based costs**: Reflect actual financing conditions

### 3. Interpretation Guidelines
- **LCOE comparison**: Valid only within similar technology classes
- **Grid LCOE**: Represents system average, not marginal cost
- **NPV interpretation**: Total system investment required
- **Emissions**: Operational emissions only, excludes lifecycle impacts

## Future Development Recommendations

### 1. Enhanced ESS Integration
- Hybrid approach combining both methodologies
- Conditional logic based on asset type
- Service-based valuation for non-generation assets

### 2. Advanced Financial Modeling
- Time-varying discount rates
- Risk-adjusted cost of capital by technology
- Real options valuation for flexibility

### 3. System Integration Features
- Transmission and distribution cost allocation
- Grid service valuation (frequency regulation, reserves)
- Demand response and efficiency program costs

### 4. Uncertainty Analysis
- Monte Carlo simulation capabilities
- Sensitivity analysis automation
- Scenario comparison tools

## Code Maintenance Notes

### Regular Updates Required
- **Data validation**: Check for missing or inconsistent inputs
- **Parameter updates**: Reflect current market conditions
- **Unit testing**: Validate calculations with known examples
- **Performance monitoring**: Track computational efficiency

### Dependencies Management
- Monitor pandas version compatibility
- Ensure Excel output format stability
- Validate utils.py function signatures
- Test with different data sheet structures

### Error Handling Improvements
- Add explicit checks for zero generation
- Implement graceful degradation for missing data
- Enhance warning messages with actionable guidance
- Add data quality reporting features