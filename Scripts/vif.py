# @title Calculate VIF Scores for 28 Weighted Sombor Indices
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from google.colab import files
import io
import warnings

warnings.filterwarnings('ignore')

print("--- UPLOAD STEP ---")
print("Please upload your Master Dataset containing the 28 Weighted Sombor Indices.")
uploaded = files.upload()

filename = next(iter(uploaded))

try:
    # Read the data
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
    
    # Keep only the numerical columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Remove the target properties so we only test the topological indices
    targets_to_remove = ['Boiling Point', 'BP', 'Molar Volume', 'MV', 'Flash Point', 'FP', 
                         'Polarizability', 'Pol', 'Molar Refractivity', 'MR', 'Density', 'D']
    
    features = df_numeric.copy()
    for col in df_numeric.columns:
        for target in targets_to_remove:
            if target.lower() in col.lower() and col in features.columns:
                features = features.drop(columns=[col])

    print(f"\nCalculating VIF for {len(features.columns)} features...")

    # Add a constant (This is mathematically required to calculate VIF correctly)
    X = add_constant(features)

    # Calculate VIF for every single feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF Score"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # Sort the scores from Highest to Lowest
    vif_data = vif_data.sort_values(by="VIF Score", ascending=False).reset_index(drop=True)

    print("\n--- TOP HIGHEST VIF SCORES ---")
    print(vif_data.head(10))
    
    # Download the final table
    out_filename = "Calculated_VIF_Scores.csv"
    vif_data.to_csv(out_filename, index=False)
    files.download(out_filename)
    print(f"\nSUCCESS! The complete VIF table has been downloaded as '{out_filename}'.")

except Exception as e:
    print(f"Critical Error: {e}")