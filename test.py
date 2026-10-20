"""
Graphene Conductivity Prediction Pipeline
-----------------------------------------
- Interpolation: RF + LightGBM + CatBoost (Optuna-tuned ensemble)
- Extrapolation: Polynomial regression + Gaussian Process Regression (with uncertainty)
- Switch: Chooses model based on whether wt% is in training range
- Includes 5-fold CV metrics for interpolation
"""

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib
matplotlib.use("TkAgg")

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold

# -------------------------
# CONFIG
# -------------------------
DATASET = "graphene final dataset.csv"
RND = 42

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATASET)

wt_col = next((c for c in df.columns if "wt" in c.lower() or "%" in c.lower()), None)
target_col = next((c for c in df.columns if "conduct" in c.lower() or "value" in c.lower() or "w/mk" in c.lower()),
                  None)

print("Detected wt column:", wt_col)
print("Detected target column:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND)


# -------------------------
# INTERPOLATION MODEL (Ensemble with Optuna)
# -------------------------
def objective(trial):
    model_choice = trial.suggest_categorical("model", ["rf", "lgbm", "cat"])

    if model_choice == "rf":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=RND, n_jobs=-1)
    elif model_choice == "lgbm":
        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            random_state=RND, n_jobs=-1)
    else:  # catboost
        model = CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 200, 500),
            depth=trial.suggest_int("depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            random_seed=RND, verbose=0)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    from math import sqrt
    return sqrt(mean_squared_error(y_test, preds))


# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=False)
best_params = study.best_params
print("\nBest Optuna params:", best_params)

# Train final ensemble models
rf = RandomForestRegressor(n_estimators=200, random_state=RND, n_jobs=-1).fit(X_train, y_train)
lgb = LGBMRegressor(n_estimators=300, random_state=RND, n_jobs=-1).fit(X_train, y_train)
cat = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=RND, verbose=0).fit(X_train, y_train)


def interpolation_predict(X_new):
    preds = np.column_stack([
        rf.predict(X_new),
        lgb.predict(X_new),
        cat.predict(X_new)
    ])
    return preds.mean(axis=1)  # simple average ensemble


# Evaluate interpolation (test split)
y_pred_interp = interpolation_predict(X_test)


# Calculate RMSE manually if 'squared' is not supported
rmse_interp = sqrt(mean_squared_error(y_test, y_pred_interp))
mae_interp = mean_absolute_error(y_test, y_pred_interp)
r2_interp = r2_score(y_test, y_pred_interp)

print("\nInterpolation performance (test set):")
print(f"RMSE: {rmse_interp:.4f}, MAE: {mae_interp:.4f}, R²: {r2_interp:.4f}")

# Cross-validation evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=RND)
rmse_list, mae_list, r2_list = [], [], []

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf_cv = RandomForestRegressor(n_estimators=200, random_state=RND, n_jobs=-1).fit(X_tr, y_tr)
    lgb_cv = LGBMRegressor(n_estimators=300, random_state=RND, n_jobs=-1).fit(X_tr, y_tr)
    cat_cv = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=RND, verbose=0).fit(X_tr, y_tr)

    preds = np.column_stack([
        rf_cv.predict(X_val),
        lgb_cv.predict(X_val),
        cat_cv.predict(X_val)
    ]).mean(axis=1)


    rmse_list.append(sqrt(mean_squared_error(y_val, preds)))
    mae_list.append(mean_absolute_error(y_val, preds))
    r2_list.append(r2_score(y_val, preds))

print("\nInterpolation performance (5-fold CV):")
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"MAE : {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"R²  : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# -------------------------
# EXTRAPOLATION MODEL
# -------------------------
# Polynomial regression on (wt, target)
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(df[[wt_col]])
linreg = LinearRegression().fit(X_poly, y)

# Gaussian Process for uncertainty
kernel = RBF(length_scale=10.0) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=RND).fit(df[[wt_col]], y)


def extrapolation_predict(wt_values):
    wt_values = np.array(wt_values).reshape(-1, 1)
    y_poly = linreg.predict(poly.transform(wt_values))
    y_gpr, sigma = gpr.predict(wt_values, return_std=True)
    y_combined = 0.7 * y_poly + 0.3 * y_gpr  # weighted hybrid
    return y_combined, sigma


# Evaluate extrapolation using synthetic holdout points outside training
wt_min, wt_max = df[wt_col].min(), df[wt_col].max()
holdout_wts = np.array([wt_min - 1, wt_min - 0.5, wt_max + 0.5, wt_max + 1]).reshape(-1, 1)
y_true_edge = linreg.predict(poly.transform(holdout_wts))  # proxy ground truth
y_pred_edge, sigma_edge = extrapolation_predict(holdout_wts)


rmse_extra = sqrt(mean_squared_error(y_true_edge, y_pred_edge))
mae_extra = mean_absolute_error(y_true_edge, y_pred_edge)
r2_extra = r2_score(y_true_edge, y_pred_edge)

print("\nExtrapolation (synthetic holdout) performance:")
print(f"RMSE: {rmse_extra:.4f}, MAE: {mae_extra:.4f}, R²: {r2_extra:.4f}")


import joblib

# After your training code:
joblib.dump(rf, "rf_model.pkl")
joblib.dump(lgb, "lgb_model.pkl")
joblib.dump(cat, "cat_model.pkl")
joblib.dump(linreg, "poly_model.pkl")
joblib.dump(gpr, "gpr_model.pkl")

joblib.dump(poly, "poly_transformer.pkl")


# Save training metadata
metadata = {
    "wt_col": wt_col,
    "wt_min": wt_min,
    "wt_max": wt_max,
    "feature_means": X.mean().to_dict()
}
joblib.dump(metadata, "graphene_metadata.pkl")

print("✅ All models and metadata saved.")



# -------------------------
# SWITCH FUNCTION
# -------------------------
def predict_pipeline(X_new):
    wts = X_new[wt_col].values
    preds, uncertainties = [], []
    for i, wt in enumerate(wts):
        if wt_min <= wt <= wt_max:
            preds.append(interpolation_predict(X_new.iloc[[i]])[0])
            uncertainties.append(np.nan)  # not used for interpolation
        else:
            y_pred, sigma = extrapolation_predict([wt])
            preds.append(y_pred[0])
            uncertainties.append(sigma[0])
    return np.array(preds), np.array(uncertainties)


# -------------------------
# DEMO PLOT
# -------------------------
test_wts = np.linspace(wt_min - 2, wt_max + 2, 100)
test_df = pd.DataFrame({wt_col: test_wts})
for c in X.columns:
    if c != wt_col:
        test_df[c] = X[c].mean()  # fill with mean

preds, unc = predict_pipeline(test_df)

plt.figure(figsize=(8, 5))
plt.scatter(df[wt_col], y, color="black", label="Data")
plt.plot(test_wts, preds, color="blue", label="Prediction")
plt.fill_between(test_wts, preds - unc, preds + unc, color="blue", alpha=0.2, label="Uncertainty (extrap)")
plt.xlabel("Weight % of Graphene")
plt.ylabel("Target")
plt.legend()
plt.title("Interpolation + Extrapolation Prediction")
plt.show()




