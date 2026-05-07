import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# Load the monthly natural gas price data
# Assumes the CSV file 'Nat_Gas.csv' is in the current working directory after download
df = pd.read_csv('Nat_Gas.csv')

# Prepare the data
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
min_date = df['Dates'].min()
df['days'] = (df['Dates'] - min_date).dt.days.astype(float)
df['month'] = df['Dates'].dt.month

# Create monthly dummy variables (drop first to avoid multicollinearity, January as baseline)
dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(float)
X = pd.concat([df[['days']], dummies], axis=1)
X = sm.add_constant(X)
y = df['Prices'].astype(float)

# Fit the OLS regression model: Price ~ time_trend + monthly_seasonal_dummies
model = sm.OLS(y, X).fit()

def get_price_estimate(date_input):
    """
    Estimate the natural gas purchase price for any given date.
    
    The model uses a linear time trend combined with monthly seasonal factors
    fitted on historical end-of-month prices (Oct 2020 - Sep 2024).
    This captures both the long-term upward trend and strong seasonal patterns
    (higher prices in winter months due to heating demand, lower in summer).
    
    Parameters:
    -----------
    date_input : str or datetime-like
        Input date in various formats, e.g. '2025-06-15', '06/15/2025', '2023-12-31'
    
    Returns:
    --------
    float
        Estimated price (rounded to 2 decimal places)
    """
    date = pd.to_datetime(date_input)
    days = (date - min_date).days
    month = date.month
    
    # Build prediction DataFrame matching training features
    pred_X = pd.DataFrame({'const': [1.0], 'days': [float(days)]})
    for m in range(2, 13):
        pred_X[f'month_{m}'] = 1.0 if month == m else 0.0
    
    price = model.predict(pred_X)[0]
    return round(price, 2)

# Example usage and testing
if __name__ == '__main__':
    print('=== Natural Gas Price Estimator ===')
    print(f'Model R-squared: {model.rsquared:.4f} (excellent fit)')
    print(f'Long-term trend: +{model.params["days"]*365:.2f} per year')
    print()
    
    # Test cases
    test_dates = [
        '2020-10-31',   # First data point
        '2021-06-15',   # Mid summer 2021
        '2023-12-31',   # Winter peak
        '2024-09-30',   # Last known
        '2025-01-31',   # Future winter
        '2025-06-30',   # Future summer
        '2025-09-30',   # One year extrapolation
        '2026-03-15'    # Further future
    ]
    
    print('Price Estimates:')
    for d in test_dates:
        est = get_price_estimate(d)
        print(f'  {d}: ${est:.2f}')
    
    print()
    
    # Generate visualization: historical data + model fit + 1-year extrapolation
    # Future monthly points (end-of-month for next 12 months)
    future_start = pd.Timestamp('2024-10-31')
    future_dates = pd.date_range(start=future_start, periods=12, freq='ME')
    
    future_days_list = [(d - min_date).days for d in future_dates]
    future_months = [d.month for d in future_dates]
    
    future_rows = []
    for days_val, month_val in zip(future_days_list, future_months):
        row = {'const': 1.0, 'days': float(days_val)}
        for m in range(2, 13):
            row[f'month_{m}'] = 1.0 if month_val == m else 0.0
        future_rows.append(row)
    
    future_X = pd.DataFrame(future_rows)
    future_prices = model.predict(future_X)
    
    # Historical fitted values for smooth curve
    hist_fitted = model.fittedvalues.values
    
    # Create the plot
    plt.figure(figsize=(14, 7))
    
    # Actual historical prices (monthly snapshots)
    plt.plot(df['Dates'], df['Prices'], 'o', markersize=8, label='Actual End-of-Month Prices', 
             color='blue', alpha=0.8, zorder=3)
    
    # Model fitted line (captures trend + seasonality)
    plt.plot(df['Dates'], hist_fitted, '-', linewidth=2.5, 
             label='Model Fit (Trend + Seasonal)', color='darkgreen', zorder=2)
    
    # Extrapolation for next year
    plt.plot(future_dates, future_prices, '--', linewidth=2.5, 
             label='Extrapolation (+1 Year)', color='red', zorder=2)
    
    # Mark the boundary between historical and forecast
    plt.axvline(x=pd.Timestamp('2024-09-30'), color='gray', linestyle=':', linewidth=1.5, 
                label='Last Known Data Point')
    
    # Add seasonal annotation
    plt.annotate('Higher Winter\nPrices (Heating Demand)', xy=(pd.Timestamp('2023-01-15'), 12.5),
                xytext=(pd.Timestamp('2022-03-01'), 9.5),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=10, color='purple', ha='center')
    plt.annotate('Lower Summer\nPrices', xy=(pd.Timestamp('2024-06-15'), 11.0),
                xytext=(pd.Timestamp('2023-09-01'), 9.0),
                arrowprops=dict(arrowstyle='->', color='orange'),
                fontsize=10, color='orange', ha='center')
    
    plt.title('Natural Gas Price Analysis & Extrapolation\n(Linear Trend + Monthly Seasonality Model)', 
              fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($ per unit)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the visualization
    output_path = 'nat_gas_price_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nVisualization saved to: {output_path}')
    
    # Print extrapolated monthly prices for the next year
    print('\n=== Extrapolated Prices for Next 12 Months (End of Month) ===')
    for d, p in zip(future_dates, future_prices):
        print(f"{d.strftime('%b %Y')}: ${p:.2f}")
    
    print('\nCode execution complete. Use get_price_estimate("YYYY-MM-DD") for any date estimate.')