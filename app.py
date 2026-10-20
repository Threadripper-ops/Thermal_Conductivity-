from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

# ===================================================
# Configuration
# ===================================================
DEBUG_MODE = True  # 🔁 Toggle this to False to silence debug prints

def log(*args, **kwargs):
    """Print only if debug mode is on."""
    if DEBUG_MODE:
        print(*args, **kwargs)

# ===================================================
# Initialize Flask
# ===================================================
app = Flask(__name__)

# ===================================================
# Load models and metadata
# ===================================================
log("DEBUG: Loading saved models and metadata...")

rf = joblib.load("rf_model.pkl")
lgb = joblib.load("lgb_model.pkl")
cat = joblib.load("cat_model.pkl")
poly_model = joblib.load("poly_model.pkl")
poly_transformer = joblib.load("poly_transformer.pkl")  # ✅ new
gpr_model = joblib.load("gpr_model.pkl")
metadata = joblib.load("graphene_metadata.pkl")

wt_col = metadata["wt_col"]
wt_min = metadata["wt_min"]
wt_max = metadata["wt_max"]
feature_means = metadata["feature_means"]

log(f"DEBUG: Models loaded successfully.")
log(f"DEBUG: Weight column -> {wt_col}, wt_min -> {wt_min}, wt_max -> {wt_max}")
log(f"DEBUG: Feature means -> {list(feature_means.keys())}")

# ===================================================
# Prediction pipeline
# ===================================================
def predict_pipeline(input_df):
    """Hybrid model: interpolation inside range, extrapolation outside."""
    preds, uncertainties = [], []

    for _, row in input_df.iterrows():
        wt = row[wt_col]
        log(f"\nDEBUG: Processing wt% -> {wt}")

        try:
            # --------------------------------------------
            # INTERPOLATION
            # --------------------------------------------
            if wt_min <= wt <= wt_max:
                log("DEBUG: Inside interpolation range → using ensemble.")
                p_rf = rf.predict([row])[0]
                p_lgb = lgb.predict([row])[0]
                p_cat = cat.predict([row])[0]
                mean_pred = np.mean([p_rf, p_lgb, p_cat])
                preds.append(mean_pred)
                uncertainties.append(np.nan)
                log(f"DEBUG: RF={p_rf:.4f}, LGBM={p_lgb:.4f}, CAT={p_cat:.4f}, Mean={mean_pred:.4f}")

            # --------------------------------------------
            # EXTRAPOLATION
            # --------------------------------------------
            else:
                log("DEBUG: Outside training range → using polynomial + GPR hybrid.")
                wt_arr = np.array([[wt]])

                # ✅ Always use the same transformer as training
                X_poly = poly_transformer.transform(wt_arr)
                y_poly = poly_model.predict(X_poly)

                # Gaussian Process Regression for uncertainty
                y_gpr, sigma = gpr_model.predict(wt_arr, return_std=True)

                # Weighted combination
                y_combined = 0.7 * y_poly + 0.3 * y_gpr
                preds.append(y_combined[0])
                uncertainties.append(sigma[0])

                log(f"DEBUG: y_poly={y_poly[0]:.4f}, y_gpr={y_gpr[0]:.4f}, σ={sigma[0]:.4f}")

        except Exception as e:
            log(f"ERROR during prediction for wt={wt}: {e}")
            preds.append(np.nan)
            uncertainties.append(np.nan)

    log(f"DEBUG: Final preds={preds}, uncertainties={uncertainties}")
    return preds, uncertainties


# ===================================================
# Prediction helper for single wt% input
# ===================================================
def predict_for_input(wt_percent):
    log(f"DEBUG: predict_for_input called with wt_percent={wt_percent}")
    input_df = pd.DataFrame({wt_col: [wt_percent]})
    for c, mean_val in feature_means.items():
        if c != wt_col:
            input_df[c] = mean_val
    log(f"DEBUG: Input DataFrame:\n{input_df}")
    preds, uncertainties = predict_pipeline(input_df)
    log(f"DEBUG: Prediction result -> preds={preds}, uncertainties={uncertainties}")
    return preds[0], uncertainties[0]


# ===================================================
# Flask routes
# ===================================================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        log("DEBUG: Received POST request with form data:", request.form)
        raw_input = request.form.get("wt_percent", "").strip()
        log(f"DEBUG: Raw input -> '{raw_input}'")

        if not raw_input:
            log("DEBUG: No input provided.")
            return render_template("index.html", error="Please enter a value before submitting.")

        try:
            wt_percent = float(raw_input)
            log(f"DEBUG: Converted wt_percent -> {wt_percent}")

            prediction, uncertainty = predict_for_input(wt_percent)
            log(f"DEBUG: Final output -> Prediction={prediction}, Uncertainty={uncertainty}")

            return render_template("index.html", prediction=prediction, uncertainty=uncertainty, wt_percent=wt_percent)

        except ValueError as e:
            log("DEBUG: ValueError ->", e)
            return render_template("index.html", error=f"'{raw_input}' is not a valid numeric value.")
        except Exception as e:
            log("DEBUG: Unexpected error ->", e)
            return render_template("index.html", error="An unexpected error occurred during prediction.")

    log("DEBUG: Rendering home page (GET request).")
    return render_template("index.html")


# ===================================================
# Run Flask app
# ===================================================
if __name__ == "__main__":
    log("DEBUG: Starting Flask app...")
    app.run(debug=True)
