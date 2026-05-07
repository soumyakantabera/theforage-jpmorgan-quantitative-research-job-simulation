# JPMorgan Chase Quantitative Research Job Simulation

This repository contains my end-to-end solutions for the **JPMorgan Quantitative Research Virtual Experience Program**.  
The work covers commodity price modeling, storage contract valuation, credit default modeling, and FICO score bucketing.

## Repository Structure

- `task1_nat_gas_analysis/` — Natural gas price estimation and visualization
- `task2_storage_contract_pricing/` — Linear-programming-based storage contract pricer
- `task3_credit_risk_modeling/` — Probability of Default (PD) and Expected Loss (EL)
- `task4_fico_rating_bucketing/` — Dynamic-programming-based FICO rating map

---

## Task 1: Investigate and Analyze Natural Gas Price Data

**Goal**  
Estimate natural gas prices for arbitrary dates and extrapolate up to one year beyond the observed data.

**Method Used**
- Monthly natural gas prices are loaded from `Nat_Gas.csv`.
- Features include:
  - a linear time trend (`days`)
  - monthly seasonal dummy variables
- A statsmodels OLS model is fit: **Price ~ trend + seasonality**.

**Results Produced in Code**
- Model reports strong fit (R² shown in script output).
- Script generates:
  - `nat_gas_price_analysis.png` (historical + fit + forecast)
  - `seasonal_pattern.png` (monthly seasonal profile)
- A reusable function `get_price_estimate(date_input)` returns estimated prices.

**Practical Value**
- Provides a transparent baseline curve for pricing and scenario analysis.
- Enables quick indicative prices when market quotes are unavailable.

---

## Task 2: Price a Commodity Storage Contract

**Goal**  
Build a prototype engine to value natural gas storage strategies under operational constraints.

**Method Used**
- Uses the Task 1 estimated price function to price injection/withdrawal dates.
- Formulates optimization as a linear program (PuLP):
  - maximize withdrawal revenue
  - minus purchase cost
  - minus inventory holding/storage costs
- Enforces contract constraints:
  - injection/withdrawal rate limits
  - storage capacity limits
  - inventory balance across event dates
  - zero terminal inventory

**Results Produced in Code**
- Script evaluates multiple sample strategies and prints contract values.
- Includes an unprofitable/reversed timing case to show behavior under poor spread conditions.
- Supporting plot file: `storage_strategy_example.png`.

**Practical Value**
- Demonstrates a scalable structure for storage valuation.
- Can be extended with richer market assumptions (fees, constraints, stochastic prices).

---

## Task 3: Credit Risk Analysis — PD Model and Expected Loss

**Goal**  
Estimate default probability and compute expected loss for individual loans.

**Method Used**
- Logistic regression (`statsmodels.Logit`) with borrower-level features:
  - credit lines outstanding
  - loan amount outstanding
  - total debt outstanding
  - income
  - years employed
  - FICO score
- Expected Loss computed as:

\[
EL = PD \times LGD \times EAD
\]

with LGD = 0.90 (10% recovery assumption).

**Results Produced in Code**
- Script prints model summary, pseudo R², AIC, and sample predictions.
- Provides reusable `expected_loss(...)` function for new borrowers.
- Visualization artifact included: `default_rate_by_rating.png`.

**Practical Value**
- Converts borrower attributes into a risk-adjusted expected loss estimate.
- Gives an interpretable baseline model for lending decision support.

---

## Task 4/5: Bucket FICO Scores into Credit Ratings

**Goal**  
Create a 10-level rating map from FICO scores (1 = best, 10 = worst).

**Method Used**
- Groups observations by FICO score and default count.
- Uses dynamic programming to maximize binomial log-likelihood across 10 buckets.
- Produces boundary-based mapping implemented in `fico_to_rating(fico_score)`.

**Results Produced in Code**
- Script prints learned bucket boundaries and per-rating default statistics.
- Visualization artifact included: `fico_distribution_with_boundaries.png`.

**Practical Value**
- Provides an interpretable discretization useful for scorecards and monitoring.
- Supports downstream models that benefit from ordinal rating bands.

---

## Consolidated Outcome

Across all tasks, this repository delivers:
- a reproducible commodity price estimation baseline,
- an optimization-based storage valuation prototype,
- an interpretable PD + expected-loss workflow,
- and an evidence-based FICO-to-rating mapping.

These components together illustrate a practical quant workflow from raw data to model outputs that can support trading/risk/lending decisions.

---

## Notes on Reproducibility

- Python dependencies used in scripts include: `pandas`, `numpy`, `statsmodels`, `matplotlib`, and `pulp`.
- Some scripts currently reference local absolute CSV paths for loan datasets. If running locally, update those paths to files under this repository (e.g., `task3_credit_risk_modeling/Loan_Data.csv`) before execution.

---

## Conclusion

This simulation demonstrates a feasible and end-to-end quantitative workflow: forecast relevant market variables, optimize decisions under constraints, estimate borrower default risk, and translate continuous credit metrics into operational rating bands.  

The current implementation is suitable as a strong prototype baseline and can be production-hardened through stronger data pipelines, path/config standardization, and additional out-of-sample validation.
