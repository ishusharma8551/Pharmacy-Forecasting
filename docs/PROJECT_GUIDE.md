# Project guide: demo script, interview preparation, algorithms, formulas

> All data is SYNTHETIC. Scores demonstrate that the pipeline works, not real-world accuracy.

## 1. Abstract
Pharmacies must balance two risks: stockouts and expiry write-offs. This project builds an inventory decision-support system that
forecasts 7-, 15- and 30-day medicine demand from daily sales history using time-based, lag and rolling features, and compares
baselines, Linear Regression, Random Forest and XGBoost under a chronological, leakage-controlled evaluation. Forecasts feed an
expiry-risk classifier that estimates, for each batch, the share likely to remain unsold at expiry (Low / Medium / High) using
Logistic Regression, Decision Tree and Random Forest, with emphasis on recall for high-risk batches. An inventory engine computes
lead-time demand, safety stock, reorder points, order quantities, overstock and potential expiry loss, and recommends FEFO dispensing.
Results are stored in SQLite and presented through a six-page Streamlit dashboard. The system is validated on a documented synthetic dataset.

## 2. Demo script (about 4 minutes)
1. **(20s) Problem.** "Pharmacies lose money from stockouts and from medicines expiring unsold. I built a system that predicts demand and turns it into stock actions. All data is synthetic - no real pharmacy data."
2. **(30s) Executive Dashboard.** Point at total inventory value, high-risk medicines, reorder-now count, potential expiry loss. "The table below lists what needs attention first."
3. **(50s) Demand Forecasting.** Pick a seasonal medicine (e.g. Cetirizine). Show history, the 7/15/30-day forecasts, then actual vs predicted on the held-out last 90 days. "Split is chronological and purged so no future information leaks."
4. **(40s) Expiry Risk.** Show distribution, high-risk table, expected unsold = stock - expected sales before expiry. Read the disclaimer: decision support, not a regulatory determination.
5. **(40s) Inventory Optimization.** Reorder list with quantities; overstock alerts; FEFO tab; show formulas tab.
6. **(40s) Model Performance.** Baselines vs models, validation-based selection, confusion matrix, and the honest note that the rule baseline matches the classifier on synthetic data.
7. **(30s) Data Management.** Upload the sample CSV, validation passes, score it, download the report; edit a batch stock and press Recompute.
8. **(10s) Close.** Limitations (synthetic data) and next steps (real data, prediction intervals).

## 3. Technical interview explanation (about 1 minute)
"I built a pipeline with four layers. First, a synthetic data generator produces 60 medicines, two years of daily sales with seasonality and noise, and batches with expiry dates;
raw files include deliberate faults that my cleaning step removes. Second, a demand model predicts total units over the next 7, 15 and 30 days using lag and rolling features, evaluated chronologically against moving-average baselines.
Third, an expiry classifier labels batches Low/Medium/High using the share that actually remained unsold at expiry in historical snapshots - so labels reflect outcomes, not the features. Fourth, inventory formulas (safety stock, reorder point, FEFO) produce actions.
Everything is saved in SQLite and shown in Streamlit, with 38 automated tests including a leakage test."

## 4. Algorithms used
- **Baselines**: naive (last day x h), 7- and 28-day moving average x h. Any ML model must beat them.
- **Linear Regression**: fits a weighted sum of features; fast, interpretable, can't capture non-linear effects.
- **Random Forest**: many decision trees on bootstrap samples, averaged; handles non-linearity and interactions, robust to scale; cannot extrapolate beyond training range.
- **XGBoost**: gradient boosting - trees built sequentially, each correcting previous errors; strong on tabular data; more hyper-parameters.
- **Logistic Regression** (classifier): linear model producing class probabilities; uses scaled features, balanced class weights.
- **Decision Tree**: if-then splits; very explainable; prone to overfit (limited depth here).
- **Random Forest Classifier**: ensemble of trees voting; reduces variance.

## 5. Evaluation metrics
- **MAE** mean absolute error, in units - easy to explain. **RMSE** penalises large errors more. **MAPE** percentage error, undefined when actual = 0, so zeros are excluded; **WAPE** = total error / total demand, stable when many small values.
- **Accuracy** share correct. **Precision** (of predicted High, how many are High). **Recall** (of real High, how many we caught) - most important for High risk because a missed high-risk batch is an avoidable loss. **F1** harmonic mean of precision and recall; *macro* averages the three classes equally.
- **Confusion matrix** rows = actual, columns = predicted; shows which classes get confused (usually Medium).

## 6. Business logic and formulas
| Quantity | Formula |
|---|---|
| Lead-time demand | forecast per day x supplier lead time |
| Safety stock | Z x sigma_daily x sqrt(lead time); Z = 1.65 (95% service) |
| Reorder point | lead-time demand + safety stock |
| Target stock | forecast/day x (lead time + 30) + safety stock, capped at max stock level, never below ROP |
| Order quantity | max(target - usable stock, 0), only when usable stock <= ROP |
| Overstock | max(usable stock - max stock level, 0) |
| Expected unsold | stock - min(stock, forecast/day x days to expiry), shared FEFO between batches |
| Potential expiry loss | expected unsold units x purchase price |
| Risk label | unsold share at expiry: <15% Low, 15-50% Medium, >=50% High |

## 7. Limitations and future scope
Synthetic data; stationary demand assumed over lead time; one supplier; no on-order stock; sigma from history; piece-wise constant rate beyond 30 days;
selection on a short validation window is noisy; the risk model's high scores partly reflect that stock was synthetically tied to demand.
Future: real data, prediction intervals, time-series cross-validation, store-level hierarchical models, price/promotion/weather features, retraining automation, ERP integration.

## 8. Twenty-five interview questions and answers
1. **Why a chronological split?** Random splits let the model see the future (e.g. train on day 100, test on day 99), inflating accuracy. Time order mirrors deployment.
2. **How did you prevent leakage?** Features use data up to the origin day only (tested by altering future sales); training targets must end before validation/test begins (purged split); expiry-risk training snapshots need outcomes known before the test cutoff; the demand feature for snapshots comes from out-of-time rolling-origin models.
3. **Why direct forecasting instead of recursive?** One model per horizon avoids compounding errors from feeding predictions back as inputs.
4. **Why baselines?** To prove ML adds value. Here ML beats the 28-day average only modestly at 7 days, more at 30 days.
5. **Why one global model for 60 medicines?** Shares patterns, needs less data per medicine, scales; medicine/category codes and lag scale let it adapt.
6. **Why was Linear Regression selected though XGBoost won on test?** Selection uses validation to avoid choosing on the test set; the short validation window favoured Linear Regression. I report both honestly and would use longer/rolling validation next.
7. **How do you handle MAPE with zeros?** Exclude zero actuals and also report WAPE.
8. **Why recall for High risk?** Missing a batch that will expire costs money; extra false alarms only cost a check.
9. **How are risk labels not circular?** Labels come from actual future sales (outcome), features from information at snapshot time; a rule baseline is reported to show what ML adds.
10. **Did the ML classifier beat the rule?** No, on synthetic data they are about equal; I report that.
11. **What is safety stock?** Buffer for demand variability during lead time: Z x sigma x sqrt(L).
12. **Why sqrt(lead time)?** Independent daily variances add; standard deviation of the sum grows with the square root.
13. **What does Z = 1.65 mean?** About 95% probability of not stocking out in a cycle under normal demand.
14. **What is FEFO and why?** First-expiry-first-out dispenses earliest-expiring stock first, reducing expired waste; also used to allocate forecast demand between batches.
15. **Why cap order quantity at max stock?** Shelf/capital limits; but never below the reorder point.
16. **How did you clean outliers?** Values above median + 10 robust SD (MAD-based) are treated as entry errors and replaced by a 7-day rolling median; normal promo spikes stay.
17. **Why interpolate missing sales?** Daily series are continuous; interpolation preserves level without biasing lags. Duplicates were dropped first.
18. **What is class imbalance and how handled?** Unequal class counts; used class_weight='balanced' and macro metrics.
19. **Random Forest vs XGBoost?** RF averages independent trees (variance reduction); XGBoost sequentially fits residuals (bias reduction) and often wins on tabular data but needs tuning.
20. **Why a 3-table database?** Medicine, batch and daily sales are different grains; one flat table would duplicate data. Foreign keys with cascade keep integrity.
21. **How is SQL injection avoided?** Parameterised queries; update functions whitelist column names.
22. **How would you deploy it?** Schedule `train.py` for retraining, host Streamlit on a server/Streamlit Cloud, switch SQLite to Postgres for multi-user.
23. **How would you monitor in production?** Track rolling MAE/WAPE vs baseline, drift in demand levels, stockout and expiry rates; retrain on thresholds.
24. **What would you improve first?** Real data, prediction intervals for safety stock from forecast error, time-series cross-validation, hyper-parameter tuning.
25. **Is this medical advice?** No. Outputs are inventory decision-support estimates; expiry handling follows SOPs and regulations.
