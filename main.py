

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from math import sqrt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
DATA = "graphene finall dataset.csv"
N_TRIALS = 30
RND = 42

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(DATA)
wt_col, target_col = df.columns[0], df.columns[1]
X = df[[wt_col]].copy()
y = df[target_col].astype(float).copy()


mask_ok = X[wt_col].notnull() & y.notnull()
X = X.loc[mask_ok].reset_index(drop=True)
y = y.loc[mask_ok].reset_index(drop=True)

# -----------------------------
# SPLITTING
# -----------------------------
n_bins = min(6, max(2, int(len(df) / 20)))
bins = pd.cut(X[wt_col], bins=n_bins, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=RND, stratify=bins)

# -----------------------------
# OPTUNA
# -----------------------------
def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": RND, "n_jobs": -1
    }
    m = RandomForestRegressor(**params)
    m.fit(X_train, y_train)
    return 1.0 - r2_score(y_test, m.predict(X_test))

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": RND, "n_jobs": -1
    }
    m = LGBMRegressor(**params)
    m.fit(X_train, y_train)
    return 1.0 - r2_score(y_test, m.predict(X_test))

def objective_cat(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 12),
        "iterations": trial.suggest_int("iterations", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": RND, "verbose": 0
    }
    m = CatBoostRegressor(**params)
    m.fit(X_train, y_train)
    return 1.0 - r2_score(y_test, m.predict(X_test))

print(" Running Optuna tuning...")
study_rf = optuna.create_study(direction="minimize"); study_rf.optimize(objective_rf, n_trials=N_TRIALS)
study_lgb = optuna.create_study(direction="minimize"); study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)
study_cat = optuna.create_study(direction="minimize"); study_cat.optimize(objective_cat, n_trials=N_TRIALS)

best_rf = study_rf.best_params
best_lgb = study_lgb.best_params
best_cat = study_cat.best_params

print("\n Best params")
print("RF:", best_rf)
print("LGB:", best_lgb)
print("CAT:", best_cat)

# -----------------------------
# TRAIN final tuned models
# -----------------------------
rf = RandomForestRegressor(**best_rf, random_state=RND, n_jobs=-1).fit(X_train, y_train)
lgb = LGBMRegressor(**best_lgb, random_state=RND, n_jobs=-1).fit(X_train, y_train)
cat = CatBoostRegressor(**best_cat, random_seed=RND, verbose=0).fit(X_train, y_train)

# -----------------------------
# compute validation R2 and weights
# -----------------------------
r2s = np.array([
    r2_score(y_test, rf.predict(X_test)),
    r2_score(y_test, lgb.predict(X_test)),
    r2_score(y_test, cat.predict(X_test))
])

r2s = np.clip(r2s, 0.0, None)
if r2s.sum() == 0:
    weights = np.ones_like(r2s) / len(r2s)
else:
    weights = r2s / r2s.sum()

print(f"\n Validation R² → RF={r2s[0]:.4f}, LGB={r2s[1]:.4f}, CAT={r2s[2]:.4f}")
print(" Ensemble weights →", np.round(weights, 3))

def interp_predict(X_new_df):

    preds = np.column_stack([
        rf.predict(X_new_df),
        lgb.predict(X_new_df),
        cat.predict(X_new_df)
    ])
    return preds @ weights

# -----------------------------
# INTERPOLATION EVALUATION
# -----------------------------
y_pred_test = interp_predict(X_test.reset_index(drop=True))
rmse = sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2v = r2_score(y_test, y_pred_test)
print(f"\n Interpolation (test): RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2v:.4f}")


# -------------------------------------------------------
# 5. Extrapolation model
# -------------------------------------------------------
print("\n Fitting improved extrapolation model...")

# Use polynomial + Gaussian Process combo
wt_vals = df[[wt_col]].values
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(wt_vals)
linreg = LinearRegression().fit(X_poly, y)

# Gaussian Process for residual fine-tuning
kernel = (
    C(1.0, (1e-3, 1e3)) *
    RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1.0)
)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    random_state=42,
    normalize_y=True
).fit(wt_vals, y)

def extrapolation_predict(wt_values):
    wt_values = np.array(wt_values).reshape(-1, 1)
    y_poly = linreg.predict(poly.transform(wt_values))
    y_gpr, sigma = gpr.predict(wt_values, return_std=True)


# Weighted combination – 85% polynomial trend + 15% GPR correction
    y_combined = 0.85 * y_poly + 0.15 * y_gpr
    return y_combined, sigma


# -----------------------------
# UNIFIED PREDICT
# -----------------------------
wt_min, wt_max = df[wt_col].min(), df[wt_col].max()
def predict_pipeline(df_in):
    preds = []
    sigs = []
    for wt in df_in[wt_col].values:
        if wt_min <= wt <= wt_max:
            preds.append(interp_predict(pd.DataFrame({wt_col: [wt]}))[0])
            sigs.append(np.nan)
        else:
            p, s = extrapolation_predict([wt])
            preds.append(p[0]); sigs.append(float(s[0]))
    return np.array(preds), np.array(sigs)


# -------------------------------------------------------
# 6. Extrapolation performance evaluation
# -------------------------------------------------------

# Generate extrapolation test data slightly beyond the known range
wt_extra = np.linspace(wt_max + 0.5, wt_max + 3, 20).reshape(-1, 1)

y_true_extra = linreg.predict(poly.transform(wt_extra))

# Model predictions & uncertainty
y_pred_extra, sigma_extra = extrapolation_predict(wt_extra)

# Compute metrics
rmse_extra = sqrt(mean_squared_error(y_true_extra, y_pred_extra))
mae_extra = mean_absolute_error(y_true_extra, y_pred_extra)
r2_extra = r2_score(y_true_extra, y_pred_extra)

print("\n Extrapolation Evaluation Metrics:")
print(f"RMSE: {rmse_extra:.4f}")
print(f"MAE : {mae_extra:.4f}")
print(f"R²  : {r2_extra:.4f}")


# -----------------------------
# PLOT
# -----------------------------
sns.set(style="whitegrid")
test_wts = np.linspace(0, wt_max + 3, 300)
test_df = pd.DataFrame({wt_col: test_wts})
preds_full, unc_full = predict_pipeline(test_df)

plt.figure(figsize=(9,6))
plt.scatter(df[wt_col], df[target_col], color="black", s=30, label="Data")
plt.plot(test_wts, preds_full, color="blue", lw=2.2, label="Prediction")
extrap_mask = test_wts > wt_max
plt.fill_between(test_wts[extrap_mask], preds_full[extrap_mask]-unc_full[extrap_mask], preds_full[extrap_mask]+unc_full[extrap_mask],
                 color="blue", alpha=0.2, label="Extrapolation uncertainty")
plt.axvspan(wt_min, wt_max, color="green", alpha=0.05, label="Interpolation")
plt.axvspan(wt_max, test_wts.max(), color="red", alpha=0.04, label="Extrapolation")
plt.xlabel("Weight % of Graphene")
plt.ylabel("Thermal Conductivity (W/mK)")
plt.legend()
plt.tight_layout()
plt.savefig("graphene_optuna_weighted.png", dpi=300)
print("Saved plot: graphene_optuna_weighted.png")

# -----------------------------
# Save models
# # -----------------------------

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

print("All models and metadata saved.")


summary = pd.DataFrame({
    "Stage": ["Interpolation", "Extrapolation"],
    "RMSE": [rmse, rmse_extra],
    "MAE": [mae, mae_extra],
    "R²": [r2v, r2_extra]
})
print("\nPerformance Summary:\n", summary.round(4))



# ==============================
# ALL PLOTS – FIXED VERSION
# ==============================


sns.set(style="whitegrid")

# ---------- 1. Correlation Heatmap ----------
# ---------- Clean Correlation Heatmap ----------
plt.figure(figsize=(10, 8))

# Compute correlation only for numeric columns
corr = df.corr(numeric_only=True)

# Use a mask to show only the lower triangle (for clarity)
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8}
)

plt.title("Feature Correlation Heatmap", fontsize=14, pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("1_corr_heatmap_clean.png", dpi=300)
plt.show()


# ---------- 2. Predicted vs Actual ----------
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_test, c='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Conductivity")
plt.ylabel("Predicted Conductivity")
plt.title("Predicted vs Actual – Interpolation Model")
plt.text(y_test.min(), y_test.max()*0.9, f"R²={r2v:.3f}\nRMSE={rmse:.2f}", fontsize=10)
plt.tight_layout()
plt.savefig("2_pred_vs_actual.png", dpi=300)
plt.close()

# ---------- 3. Residual Plot ----------
residuals = y_test - y_pred_test
plt.figure(figsize=(6,5))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Analysis – Interpolation Model")
plt.xlabel("Actual Conductivity")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("3_residuals.png", dpi=300)
plt.close()

# ---------- 4. Weighted Ensemble Contribution ----------
plt.figure(figsize=(6,5))
models = ['Random Forest', 'LightGBM', 'CatBoost']
plt.bar(models, weights, color=['green','orange','purple'])
plt.title("Weighted Ensemble Contribution")
plt.ylabel("Weight Value")
plt.tight_layout()
plt.savefig("4_ensemble_weights.png", dpi=300)
plt.close()

# ---------- 5. Hybrid vs Polynomial Extrapolation ----------
plt.figure(figsize=(8,5))
y_poly = linreg.predict(poly.transform(test_wts.reshape(-1,1)))
plt.plot(test_wts, y_poly, '--', label='Polynomial Only')
plt.plot(test_wts, preds_full, label='Hybrid (Poly + GPR)', color='blue')
plt.legend()
plt.title("Impact of GPR Correction on Extrapolation Smoothness")
plt.xlabel("Graphene wt%")
plt.ylabel("Thermal Conductivity")
plt.tight_layout()
plt.savefig("5_poly_vs_gpr.png", dpi=300)
plt.close()

# ---------- 6. Extrapolation Region Focus ----------
plt.figure(figsize=(7,4))
plt.plot(test_wts, preds_full, label="Prediction", color="blue")
plt.fill_between(test_wts, preds_full-unc_full, preds_full+unc_full,
                 color="blue", alpha=0.2, label="Uncertainty")
plt.axvline(wt_max, color="red", linestyle="--", label="Training Limit")
plt.xlim(wt_max-0.5, test_wts.max())
plt.title("Extrapolation Region – Model Behavior Beyond Training Data")
plt.legend()
plt.xlabel("Graphene wt%")
plt.ylabel("Thermal Conductivity (W/mK)")
plt.tight_layout()
plt.savefig("6_extrapolation_focus.png", dpi=300)
plt.close()

# ---------- 7. Full Prediction Range ----------
plt.figure(figsize=(9,6))
plt.scatter(df[wt_col], df[target_col], color="black", s=30, label="Experimental Data")
plt.plot(test_wts, preds_full, color="blue", lw=2.2, label="Model Prediction")
extrap_mask = test_wts > wt_max
plt.fill_between(test_wts[extrap_mask],
                 preds_full[extrap_mask]-unc_full[extrap_mask],
                 preds_full[extrap_mask]+unc_full[extrap_mask],
                 color="blue", alpha=0.2, label="Extrapolation Uncertainty")
plt.axvspan(wt_min, wt_max, color="green", alpha=0.05, label="Interpolation Range")
plt.axvspan(wt_max, test_wts.max(), color="red", alpha=0.04, label="Extrapolation Range")
plt.xlabel("Weight % of Graphene")
plt.ylabel("Thermal Conductivity (W/mK)")
plt.title("Final Model Prediction with Uncertainty")
plt.legend()
plt.tight_layout()
plt.savefig("7_full_prediction.png", dpi=300)
plt.close()

print("✅ All plots generated and saved successfully!")


