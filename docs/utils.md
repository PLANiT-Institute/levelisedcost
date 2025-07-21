# Utils Module Documentation

## Overview
The `lib/utils.py` module provides core utility functions for levelized cost of energy (LCOE) and discounted cash flow (DCF) calculations. It contains financial modeling functions, data processing utilities, and helper functions used across the analysis framework.

## Module Structure

### Core Functions

#### 1. `calculate_lcoe()`
**Purpose**: Calculate traditional Levelized Cost of Energy using NPV methodology.

**Parameters**:
- `capacity` (float): Installed capacity in kW
- `generation` (float): Annual generation in kWh
- `capex_per_kw` (float): Capital expenditure per kW
- `fixed_opex_per_kw` (float): Fixed OPEX per kW per year
- `variable_opex` (float): Variable OPEX per kWh
- `land_cost_per_kw` (float): Land cost per kW per year
- `lifespan` (int): Project lifespan in years
- `discount_rate` (float): Discount rate as decimal
- `degradation` (float, optional): Annual degradation rate
- `interest_rate` (float, optional): Interest rate on CAPEX
- `tax_rate` (float, optional): Tax rate on operating costs

**Logic**:
1. Calculate total CAPEX with interest (applied in year 1)
2. Loop through project lifespan:
   - Apply discount factor: `(1 + discount_rate) ** (year - 1)`
   - Calculate degraded generation: `generation * (1 - degradation) ** (year - 1)`
   - Sum annual costs with taxes
   - Accumulate present values of costs and generation
3. Return LCOE = Total PV Costs / Total PV Generation

**Mathematical Formula**:
```
LCOE = Σ[Annual_Cost_t / (1 + r)^t] / Σ[Generation_t / (1 + r)^t]
```

**Known Issues**:
- Returns `inf` when total generation is zero (ESS scenario)
- Generates RuntimeWarning for division by zero
- Does not handle negative generation values

#### 2. `calculate_lcoe_with_economic_lifespan()`
**Purpose**: Enhanced LCOE calculation with separate economic lifespan for CAPEX distribution.

**Key Difference**: Distributes CAPEX over `economic_lifespan` years instead of applying all in year 1.

**Logic**:
- Annual CAPEX payment = `capex_with_interest / economic_lifespan`
- CAPEX payments only occur during economic lifespan period
- Operational costs continue for full project lifespan

#### 3. `calculate_discounted_cashflow_from_lcoe()`
**Purpose**: Calculate DCF from pre-calculated LCOE values (traditional approach).

**Parameters**:
- `lcoe_df`: DataFrame of LCOE values by fuel and year
- `generation_df`: DataFrame of generation by fuel and year
- `discount_rate`: Discount rate as decimal

**Process**:
1. For each fuel and year:
   - `annual_cost = lcoe × generation`
   - `present_value = annual_cost / discount_factor`
   - `discounted_generation = generation / discount_factor`
2. Calculate grid totals as sum across fuels
3. Return comprehensive results dictionary

**Output Structure**:
```python
{
    'annual_costs': DataFrame,
    'present_values': DataFrame,
    'discounted_generation': DataFrame,
    'annual_lcoe': DataFrame,
    'grid_lcoe': Series,
    'total_npv': Series,
    'total_generation': float
}
```

#### 4. `calculate_discounted_cashflow_direct()`
**Purpose**: Calculate DCF directly from cost components (DCF-first approach).

**Key Innovation**: Calculates present value of all costs over project lifespan, then derives equivalent annual cost.

**Logic**:
1. Use vintage tracking with `calculate_weighted_average()`
2. For each fuel/year with capacity > 0:
   - Calculate total PV costs over entire project lifespan
   - Account for CAPEX in year 1, OPEX in all years
   - Calculate annual equivalent cost
3. Derive LCOE from total PV costs / total PV generation

**Advantage**: Provides consistent cost methodology regardless of generation profile.

### Data Processing Functions

#### 5. `load_excel()`
**Purpose**: Load and clean Excel data with year filtering.

**Process**:
1. Load all sheets except those starting with 'exclude_'
2. Drop 'unit' and 'description' rows/columns
3. Convert index to integers for year-based sheets
4. Filter by year range if specified
5. Convert all data to numeric types

**Data Cleaning**:
- Handles missing values with `pd.to_numeric(errors='coerce')`
- Filters out metadata rows/columns
- Ensures consistent data types across sheets

#### 6. `calculate_weighted_average()`
**Purpose**: Calculate weighted average CAPEX accounting for capacity vintages.

**Vintage Tracking Logic**:
1. Track capacity additions/reductions by year
2. When capacity is reduced, remove from oldest vintages first
3. Calculate weighted average CAPEX based on remaining capacity mix
4. Handle negative deltas with `reduce_list_from_top()`

**Mathematical Approach**:
```
Weighted_CAPEX = Σ(Capacity_vintage × CAPEX_vintage) / Σ(Capacity_vintage)
```

#### 7. `reduce_list_from_top()`
**Purpose**: Helper function for capacity reduction logic.

**Process**:
- Reduces values from top of list until reduction amount is exhausted
- Used in vintage tracking when capacity is retired
- Implements FIFO (First-In-First-Out) retirement logic

### Auxiliary Functions

#### 8. `calculate_lcoh()`
**Purpose**: Calculate Levelized Cost of Hydrogen for electrolyzer analysis.

**Specialized Parameters**:
- `efficiency`: Electricity consumption per kg H2 (kWh/kg)
- `electricity_cost`: Cost per kWh
- `capacity_factor`: Utilization rate

**Adaptation**: Similar to LCOE but for hydrogen production economics.

#### 9. `dualfuel_generation()`
**Purpose**: Handle dual-fuel scenarios by subtracting alternative fuel generation.

**Use Case**: When assets can operate on multiple fuels, calculates net generation.

#### 10. `forecast_series()`
**Purpose**: Linear regression forecasting for time series data.

**Implementation**:
- Uses sklearn.linear_model.LinearRegression
- Fits historical data to predict future values
- Returns forecasted series with specified years

#### 11. `plot_levelisedcost()`
**Purpose**: Visualization utility for LCOE trends.

**Features**:
- Plots multiple technologies over time
- Configurable figure size and legend position
- Uses matplotlib for publication-quality plots

## Data Flow Architecture

### Traditional LCOE Approach (costanalysis_v1.0.py)
```
Input Data → calculate_lcoe() → LCOE Values → calculate_discounted_cashflow_from_lcoe() → DCF Results
```

### DCF-First Approach (dcf_v1.0.py)
```
Input Data → calculate_discounted_cashflow_direct() → Annual Costs → Derive LCOE from DCF
```

## Error Handling

### Common Warnings
1. **RuntimeWarning: divide by zero**
   - Occurs when generation = 0 (ESS scenarios)
   - Function returns `inf` or `NaN`
   - Handled downstream with conditional logic

2. **RuntimeWarning: invalid value encountered**
   - Results from operations with `inf` or `NaN` values
   - Typically propagates from zero-generation scenarios

### Defensive Programming
- Parameter validation with default values
- Graceful handling of missing data
- Consistent return types across functions

## Performance Considerations

### Optimization Opportunities
1. **Vectorization**: Some loops could be vectorized with pandas operations
2. **Memory Usage**: Large DataFrames created for intermediate calculations
3. **Redundant Calculations**: LCOE calculated multiple times in some workflows

### Computational Complexity
- Most functions: O(n×m) where n=years, m=fuels
- Vintage tracking: O(n²) due to cumulative calculations
- Overall framework: Scales linearly with data size

## Known Issues and Limitations

### 1. Zero Generation Handling
**Issue**: Division by zero when generation = 0
**Impact**: ESS and other non-generation assets show `inf` LCOE
**Mitigation**: DCF-first approach resolves this limitation

### 2. Vintage Tracking Complexity
**Issue**: Complex logic in `calculate_weighted_average()`
**Risk**: Potential errors in capacity retirement calculations
**Testing**: Requires comprehensive validation with known scenarios

### 3. Missing Data Handling
**Issue**: `pd.to_numeric(errors='coerce')` converts errors to NaN
**Risk**: Silent data quality issues
**Recommendation**: Add explicit data validation steps

### 4. Memory Efficiency
**Issue**: Multiple large DataFrames created simultaneously
**Impact**: High memory usage for large datasets
**Optimization**: Implement lazy evaluation or chunked processing

### 5. Error Propagation
**Issue**: Mathematical errors (inf, NaN) propagate through calculations
**Risk**: Invalid results in downstream analysis
**Solution**: Add robust error checking at key calculation points

## Dependencies

### Required Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `sklearn`: Linear regression for forecasting
- `matplotlib`: Plotting and visualization

### Version Compatibility
- Developed with pandas 1.x series
- Compatible with Python 3.7+
- No breaking changes expected with current dependencies

## Future Enhancements

### Proposed Improvements
1. **Enhanced Error Handling**: Comprehensive validation and error reporting
2. **Performance Optimization**: Vectorized calculations where possible
3. **Extended Functionality**: Support for more complex financial scenarios
4. **Documentation**: Inline docstrings following numpy/scipy conventions
5. **Unit Testing**: Comprehensive test suite for all functions
6. **Type Hints**: Add static typing for better code clarity and IDE support

### Architectural Considerations
- Separate financial calculations from data processing
- Implement abstract base classes for different calculation methodologies
- Add configuration management for default parameters
- Consider implementing caching for expensive calculations