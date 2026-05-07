import pandas as pd
import numpy as np
import pulp
import statsmodels.api as sm
from datetime import datetime

# ============================================================
# Load data and fit the price estimation model (from previous work)
# ============================================================
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
min_date = df['Dates'].min()
df['days'] = (df['Dates'] - min_date).dt.days.astype(float)
df['month'] = df['Dates'].dt.month
dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(float)
X = pd.concat([df[['days']], dummies], axis=1)
X = sm.add_constant(X)
y = df['Prices'].astype(float)
price_model = sm.OLS(y, X).fit()

def get_price_estimate(date_input):
    """Estimate price on any date using the fitted trend + seasonality model."""
    date = pd.to_datetime(date_input)
    days = (date - min_date).days
    month = date.month
    pred_X = pd.DataFrame({'const': [1.0], 'days': [float(days)]})
    for m in range(2, 13):
        pred_X[f'month_{m}'] = 1.0 if month == m else 0.0
    return round(price_model.predict(pred_X)[0], 2)

# ============================================================
# Main pricing function for the storage contract
# ============================================================
def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    inj_rate=10000.0,
    wd_rate=10000.0,
    max_volume=100000.0,
    storage_cost_per_unit_per_day=0.005
):
    """
    Prototype pricing model for a natural gas storage contract.
    
    Parameters
    ----------
    injection_dates : list of str or datetime
        Dates on which gas can be injected (bought).
    withdrawal_dates : list of str or datetime
        Dates on which gas can be withdrawn (sold).
    inj_rate : float
        Maximum volume that can be injected on any single injection date (daily rate).
    wd_rate : float
        Maximum volume that can be withdrawn on any single withdrawal date.
    max_volume : float
        Maximum storage capacity (inventory limit at any time).
    storage_cost_per_unit_per_day : float
        Daily storage cost per unit of gas held in inventory.
    
    Returns
    -------
    float
        Maximum net value (profit) of the contract in the same currency units as prices.
        This is the optimized value considering all cash flows:
        - Revenue from withdrawals
        - Cost of injections
        - Storage costs over holding periods
        Subject to physical constraints (capacity, injection/withdrawal rates, 
        start and end with zero inventory).
    """
    if not injection_dates or not withdrawal_dates:
        return 0.0
    
    # Convert and prepare dates
    inj_dates = sorted([pd.to_datetime(d) for d in injection_dates])
    wd_dates = sorted([pd.to_datetime(d) for d in withdrawal_dates])
    all_dates = sorted(set(inj_dates + wd_dates))
    
    # Estimate market prices on the chosen dates using the model
    inj_price_dict = {d: get_price_estimate(d) for d in inj_dates}
    wd_price_dict = {d: get_price_estimate(d) for d in wd_dates}
    
    # Create PuLP linear programming problem (maximize profit)
    prob = pulp.LpProblem("Natural_Gas_Storage_Contract", pulp.LpMaximize)
    
    # Decision variables: volumes to inject/withdraw on each allowed date
    inj_vars = {}
    for d in inj_dates:
        name = f"inj_{d.strftime('%Y%m%d')}"
        inj_vars[d] = pulp.LpVariable(name, lowBound=0, upBound=inj_rate)
    
    wd_vars = {}
    for d in wd_dates:
        name = f"wd_{d.strftime('%Y%m%d')}"
        wd_vars[d] = pulp.LpVariable(name, lowBound=0, upBound=wd_rate)
    
    # Inventory level AFTER actions on each event date (K time points)
    K = len(all_dates)
    inv_vars = [pulp.LpVariable(f"inv_{k}", lowBound=0, upBound=max_volume) for k in range(K)]
    
    # Objective function: maximize (withdrawal revenue - injection costs - storage costs)
    revenue = pulp.lpSum(wd_price_dict[d] * wd_vars[d] for d in wd_dates)
    buy_cost = pulp.lpSum(inj_price_dict[d] * inj_vars[d] for d in inj_dates)
    
    storage_cost_expr = 0
    for k in range(K - 1):
        delta_days = (all_dates[k + 1] - all_dates[k]).days
        storage_cost_expr += storage_cost_per_unit_per_day * inv_vars[k] * delta_days
    prob += revenue - buy_cost - storage_cost_expr, "Net_Profit"
    
    # Inventory balance constraints (start empty, actions at each date, end empty)
    for k, tk in enumerate(all_dates):
        inj_add = inj_vars.get(tk, 0)   # 0 if this date is not an injection date
        wd_sub  = wd_vars.get(tk, 0)    # 0 if this date is not a withdrawal date
        
        if k == 0:
            prob += inv_vars[k] == inj_add - wd_sub, f"Balance_0"
        else:
            prob += inv_vars[k] == inv_vars[k-1] + inj_add - wd_sub, f"Balance_{k}"
    
    # Must end with zero inventory (contract cycle complete)
    prob += inv_vars[-1] == 0, "Final_Empty"
    
    # Solve the LP (quietly)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
    
    if pulp.LpStatus[status] != "Optimal":
        return 0.0  # No feasible profitable solution or solver issue
    
    optimal_value = pulp.value(prob.objective)
    return round(optimal_value, 2)

# ============================================================
# Test the prototype with sample inputs
# ============================================================
if __name__ == "__main__":
    print("=== Natural Gas Storage Contract Pricing Prototype ===\n")
    print("Using fitted price model (R² = 0.955) for indicative market prices.\n")
    
    # Sample 1: Simple cycle - inject in low-price summer, withdraw in high-price winter
    print("Sample 1: Single injection (2023-06-30) + Single withdrawal (2023-12-31)")
    val1 = price_storage_contract(
        injection_dates=['2023-06-30'],
        withdrawal_dates=['2023-12-31'],
        inj_rate=10000.0,
        wd_rate=10000.0,
        max_volume=10000.0,           # limits to one full injection
        storage_cost_per_unit_per_day=0.005
    )
    print(f"  Contract Value: ${val1:,.2f}\n")
    
    # Sample 2: Multiple dates - two injections in summer, two withdrawals in winter
    print("Sample 2: Multiple dates (two injections in 2023 summer + two withdrawals in winter 2023/24)")
    val2 = price_storage_contract(
        injection_dates=['2023-05-31', '2023-06-30'],
        withdrawal_dates=['2023-12-31', '2024-01-31'],
        inj_rate=8000.0,
        wd_rate=8000.0,
        max_volume=50000.0,
        storage_cost_per_unit_per_day=0.004
    )
    print(f"  Contract Value: ${val2:,.2f}\n")
    
    # Sample 3: Short-term cycle with lower storage cost (more profitable)
    print("Sample 3: Short-term - inject May 2024, withdraw Dec 2024 (extrapolated)")
    val3 = price_storage_contract(
        injection_dates=['2024-05-31'],
        withdrawal_dates=['2024-12-31'],
        inj_rate=15000.0,
        wd_rate=15000.0,
        max_volume=15000.0,
        storage_cost_per_unit_per_day=0.003
    )
    print(f"  Contract Value: ${val3:,.2f}\n")
    
    # Sample 4: Unprofitable case (withdrawal before injection or poor spread)
    print("Sample 4: Reversed timing (should optimally do nothing)")
    val4 = price_storage_contract(
        injection_dates=['2024-12-31'],
        withdrawal_dates=['2024-05-31'],
        inj_rate=10000.0,
        wd_rate=10000.0,
        max_volume=10000.0,
        storage_cost_per_unit_per_day=0.005
    )
    print(f"  Contract Value: ${val4:,.2f} (expected near zero)\n")
    
    print("Prototype ready for further validation and testing.")
    print("The model considers all cash flows and physical constraints via linear programming.")