# @title Calculate Pearson Correlation (Fixed for Excel/CSV)
# Step 1: Install libraries if needed
!pip install seaborn matplotlib pandas openpyxl

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
import io
import os

# --------------------------------------------------------------------------------
# 1. UPLOAD DATA
# --------------------------------------------------------------------------------
print("--- UPLOAD STEP ---")
print("Please upload your data file (Excel .xlsx or CSV .csv)")
uploaded = files.upload()

# Get the filename
filename = next(iter(uploaded))
print(f"Processing file: {filename}")

# --------------------------------------------------------------------------------
# 2. LOAD DATA (Auto-Detect Format)
# --------------------------------------------------------------------------------
try:
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        print("Detected Excel file. Reading with pd.read_excel...")
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        print("Detected CSV file. Reading with pd.read_csv...")
        try:
            df = pd.read_csv(io.BytesIO(uploaded[filename]))
        except UnicodeDecodeError:
            print("UTF-8 failed. Retrying with Latin1 encoding...")
            df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')

    print(f"Success! Loaded {len(df)} compounds.")

except Exception as e:
    print(f"\nCRITICAL ERROR: Could not read the file. \nError details: {e}")
    raise

# --------------------------------------------------------------------------------
# 3. DEFINE COLUMNS
# --------------------------------------------------------------------------------
# Target Properties (Y) - We will try to find them even if case/spacing differs
potential_targets = [
    'Boiling Point (BP)', 
    'Molar Volume (MV)', 
    'Flash Point (FP)', 
    'Polarizability (Pol)', 
    'Molar Refractivity (MR)', 
    'Density (D)'
]

# Find actual column names in the dataframe that match our targets
targets = [col for col in df.columns if col in potential_targets]

# If exact names aren't found, try finding them by keywords
if not targets:
    print("Exact target names not found. Searching by keywords...")
    keywords = ['Boiling', 'Volume', 'Flash', 'Polariz', 'Refractivity', 'Density']
    for key in keywords:
        found = [col for col in df.columns if key in col]
        if found:
            targets.append(found[0])

# Calculated Indices (X) - Automatically find columns ending with _SO, _ESO, _MESO
indices = [col for col in df.columns if any(tag in col for tag in ['_SO', '_ESO', '_MESO'])]

print(f"\nIdentified {len(targets)} Target Properties: {targets}")
print(f"Identified {len(indices)} Weighted Sombor Indices.")

# --------------------------------------------------------------------------------
# 4. CALCULATE PEARSON CORRELATION
# --------------------------------------------------------------------------------
if targets and indices:
    # Ensure data is numeric
    correlation_data = df[targets + indices].apply(pd.to_numeric, errors='coerce')

    # Calculate Correlation Matrix
    full_matrix = correlation_data.corr(method='pearson')

    # Extract only Properties vs Indices
    # Rows = Indices, Columns = Properties
    results_matrix = full_matrix.loc[indices, targets]

    # --------------------------------------------------------------------------------
    # 5. SHOW TOP RESULTS & SAVE
    # --------------------------------------------------------------------------------
    print("\n--- TOP 5 CORRELATED INDICES PER PROPERTY ---")

    for target in targets:
        print(f"\nProperty: {target}")
        # Sort by absolute correlation value
        top_correlations = results_matrix[target].abs().sort_values(ascending=False).head(5)
        
        for idx_name in top_correlations.index:
            r_value = results_matrix.loc[idx_name, target]
            print(f"  {idx_name}: R = {r_value:.4f}")

    # Save the matrix
    output_filename = "Correlation_Analysis_Pearson_R.csv"
    results_matrix.to_csv(output_filename)

    print(f"\n--- SUCCESS ---")
    print(f"Correlation matrix saved to: {output_filename}")
    files.download(output_filename)

else:
    print("\nError: Could not find target properties or indices columns. Please check your file headers.")