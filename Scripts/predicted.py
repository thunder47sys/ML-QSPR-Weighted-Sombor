# @title Generate Predicted Values (Best Model per Property)
# Step 1: Install XGBoost
!pip install xgboost

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from google.colab import files
import io
import warnings

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1. UPLOAD DATA
# --------------------------------------------------------------------------------
print("--- UPLOAD STEP ---")
print("Please upload: 'Data_Alpha_Geometric_Weighted_Sombor_Indices.csv'")
uploaded = files.upload()

filename = next(iter(uploaded))
try:
    if filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        try:
            df = pd.read_csv(io.BytesIO(uploaded[filename]))
        except:
            df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')
    print(f"Success! Loaded {len(df)} compounds.")
except Exception as e:
    print(f"Error: {e}")

# --------------------------------------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------------------------------------
targets = [
    'Boiling Point (BP)', 'Molar Volume (MV)', 'Flash Point (FP)', 
    'Polarizability (Pol)', 'Molar Refractivity (MR)', 'Density (D)'
]

actual_targets = []
for t in targets:
    if t in df.columns:
        actual_targets.append(t)
    else:
        keyword = t.split('(')[0].strip()
        found = [col for col in df.columns if keyword in col]
        if found:
            actual_targets.append(found[0])

features = [col for col in df.columns if any(x in col for x in ['_SO', '_ESO', '_MESO'])]

# Force Numeric
for col in actual_targets + features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define Models
models = {
    "MLR": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LinearRegression())]),
    "Lasso": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Lasso(random_state=42, max_iter=10000))]),
    "Ridge": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge(random_state=42))]),
    "RF": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
    "XGB": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42))]),
    "ANN": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42))]),
    "SVR": Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', SVR(kernel='rbf'))])
}

# --------------------------------------------------------------------------------
# 3. GENERATE PREDICTIONS (Using 5-Fold Cross-Validation)
# --------------------------------------------------------------------------------
# We create a new DataFrame to hold the predictions
output_df = df[['ID', 'Name', 'SMILES']].copy() if 'ID' in df.columns else df.iloc[:, :3].copy()
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- GENERATING PREDICTIONS ---")

for target in actual_targets:
    print(f"Processing {target}...")
    
    # Prepare Data
    temp_df = df.dropna(subset=[target] + features)
    X = temp_df[features]
    y = temp_df[target]
    
    # 1. Find the Best Model first
    best_score = -np.inf
    best_name = ""
    
    for name, pipeline in models.items():
        # Quick check of R2
        scores = cross_validate(pipeline, X, y, cv=cv_strategy, scoring='r2')
        avg_r2 = np.mean(scores['test_score'])
        if avg_r2 > best_score:
            best_score = avg_r2
            best_name = name
            
    print(f"   Best Model found: {best_name} (R2={best_score:.3f})")
    
    # 2. Generate Predictions using the Best Model via Cross-Validation
    # This ensures the prediction for Compound X is made when Compound X was in the 'Test Set'
    best_pipeline = models[best_name]
    predictions = cross_val_predict(best_pipeline, X, y, cv=cv_strategy)
    
    # 3. Add to Output DataFrame
    # We need to align indices because of dropna()
    series_pred = pd.Series(predictions, index=temp_df.index)
    output_df[f"{target} (Exp)"] = df[target]
    output_df[f"{target} (Pred)"] = series_pred
    output_df[f"{target} (Residual)"] = output_df[f"{target} (Exp)"] - output_df[f"{target} (Pred)"]

# --------------------------------------------------------------------------------
# 4. SAVE
# --------------------------------------------------------------------------------
print("\n--- SAVING SUPPLEMENTARY DATA ---")
output_file = "Supplementary_Predicted_Values.csv"
output_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
files.download(output_file)