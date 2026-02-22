# @title Generate Table 2 (Fixed Convergence & Exact Format)
# Step 1: Install XGBoost
!pip install xgboost

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from google.colab import files
import io
import warnings

# Suppress annoying warnings for clean output
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
# Targets
targets = [
    'Boiling Point (BP)', 'Molar Volume (MV)', 'Flash Point (FP)', 
    'Polarizability (Pol)', 'Molar Refractivity (MR)', 'Density (D)'
]

# Identify columns
actual_targets = []
for t in targets:
    if t in df.columns:
        actual_targets.append(t)
    else:
        keyword = t.split('(')[0].strip()
        found = [col for col in df.columns if keyword in col]
        if found:
            actual_targets.append(found[0])

# Force Numeric
features = [col for col in df.columns if any(x in col for x in ['_SO', '_ESO', '_MESO'])]
for col in actual_targets + features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define Models (UPDATED parameters to fix warnings)
models = {
    "MLR": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    "Lasso": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # Increased max_iter to 10,000 to ensure convergence
        ('model', Lasso(random_state=42, max_iter=10000))
    ]),
    "Ridge": Pipeline([ 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(random_state=42))
    ]),
    "RF": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "XGB": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42))
    ]),
    "ANN": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # Increased max_iter to 5,000 to ensure convergence for Neural Network
        ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=5000, random_state=42))
    ]),
    "SVR": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])
}

# --------------------------------------------------------------------------------
# 3. GENERATE TABLE 2
# --------------------------------------------------------------------------------
table_rows = []
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'}

print("\n--- CALCULATING PERFORMANCE (This may take 1-2 minutes) ---")

for target in actual_targets:
    print(f"Processing {target}...")
    
    # Prepare Data
    temp_df = df.dropna(subset=[target] + features)
    if len(temp_df) < 10: continue
    
    X = temp_df[features]
    y = temp_df[target]
    
    # Store results for this property to sort them later
    prop_results = []
    
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv_strategy, scoring=scoring)
        
        r2 = np.mean(scores['test_r2'])
        mae = -np.mean(scores['test_mae'])
        rmse = -np.mean(scores['test_rmse'])
        
        prop_results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })
    
    # Sort models by R2 (Highest first)
    prop_results.sort(key=lambda x: x['R2'], reverse=True)
    
    # Add to main table list
    first_row = True
    for res in prop_results:
        table_rows.append({
            "Property": target if first_row else "", 
            "Model": res['Model'],
            "R2": round(res['R2'], 4),
            "MAE": round(res['MAE'], 4),
            "RMSE": round(res['RMSE'], 4)
        })
        first_row = False

# --------------------------------------------------------------------------------
# 4. SAVE
# --------------------------------------------------------------------------------
final_df = pd.DataFrame(table_rows)
print("\n--- TABLE 2: STATISTICAL PERFORMANCE ---")
print(final_df.head(15)) # Show first few rows

final_df.to_csv("Table_2_Statistical_Performance.csv", index=False)
files.download("Table_2_Statistical_Performance.csv")