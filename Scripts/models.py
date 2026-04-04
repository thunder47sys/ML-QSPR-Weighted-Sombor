# @title Generate 
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from google.colab import files
import io
import warnings

warnings.filterwarnings('ignore')

# 1. UPLOAD DATA
print("--- UPLOAD STEP ---")
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_excel(io.BytesIO(uploaded[filename])) if filename.endswith('.xlsx') else pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')

# 2. CONFIGURATION
targets = ['Boiling Point (BP)', 'Molar Volume (MV)', 'Flash Point (FP)', 'Polarizability (Pol)', 'Molar Refractivity (MR)', 'Density (D)']
features = [col for col in df.columns if any(x in col for x in ['_SO', '_ESO', '_MESO'])]

# Stable Features specifically for MLR
mlr_stable_features = ['mass_SO', 'en_MESO']

# Models with your EXACT parameters
models = {
    "MLR": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LinearRegression())]),
    "Lasso": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Lasso(random_state=42, max_iter=10000))]),
    "Ridge": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge(random_state=42))]),
    "RF": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
    "XGB": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42))]),
    "ANN": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42))]),
    "SVR": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', SVR(kernel='rbf'))])
}

# 3. CALCULATION LOOP
table_rows = []
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}

print("\n--- CALCULATING PERFORMANCE  ---")

for target in targets:
    if target not in df.columns: continue
    temp_df = df.dropna(subset=[target] + features)
    y = temp_df[target]

    for name, pipeline in models.items():
        # THE FIX: Only MLR uses the 2 stable features. All others use all 28.
        X_input = temp_df[mlr_stable_features] if name == "MLR" else temp_df[features]

        scores = cross_validate(pipeline, X_input, y, cv=cv_strategy, scoring=scoring)

        table_rows.append({
            "Property": target, "Model": name,
            "R2": round(np.mean(scores['test_r2']), 4),
            "MAE": round(-np.mean(scores['test_mae']), 4),
            "RMSE": round(-np.mean(scores['test_rmse']), 4)
        })

# 4. EXPORT
final_df = pd.DataFrame(table_rows)
final_df.to_excel("Clean_Unique_200_Pharmaceuticals.xlsx", index=False)
files.download("Clean_Unique_200_Pharmaceuticals.xlsx")
print("\n--- Success! ---")
