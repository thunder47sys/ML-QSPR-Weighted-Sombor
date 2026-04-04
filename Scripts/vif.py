import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from google.colab import files
import io

# 1. UPLOAD STEP
print("--- UPLOAD STEP ---")
print("Please upload your dataset (e.g., Data_Alpha_Geometric_Weighted_Sombor_Indices.csv)")
uploaded = files.upload()
filename = next(iter(uploaded))

# Read the file
if filename.endswith('.xlsx') or filename.endswith('.xls'):
    df = pd.read_excel(io.BytesIO(uploaded[filename]))
else:
    df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')

# 2. IDENTIFY ALL 28 INDICES
all_indices = [col for col in df.columns if any(tag in col for tag in ['_SO', '_ESO', '_MESO'])]
print(f"\nProcessing {len(all_indices)} indices...")

# 3. GENERATE HIGH-RESOLUTION CORRELATION MATRIX
print("\n--- Generating High-Res Correlation Matrix ---")
corr_matrix = df[all_indices].corr().abs()

plt.figure(figsize=(24, 20))
sns.heatmap(corr_matrix, cmap='RdBu_r', vmin=0.8, vmax=1.0, 
            square=True, linewidths=0.5, linecolor='white',
            cbar_kws={"shrink": .8, "label": "Absolute Correlation"})

plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
plt.yticks(rotation=0, fontsize=14, fontweight='bold')
plt.title('Pearson Correlation Matrix of Weighted Sombor Indices', fontsize=26, pad=20, fontweight='bold')
plt.tight_layout()

# Save and download the image
hq_filename = 'HQ_Correlation_Matrix.png'
plt.savefig(hq_filename, dpi=600, bbox_inches='tight')
plt.show()
files.download(hq_filename)

# 4. SELECT THE 2 FINAL NON-NOISE INDICES & CALCULATE VIF
final_indices = ['mass_SO', 'en_MESO']
print(f"\n--- Calculating VIF for selected indices: {final_indices} ---")

# Drop missing values to prevent math errors
X_final = df[final_indices].dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_final.columns
vif_data["VIF Score"] = [variance_inflation_factor(X_final.values, i) for i in range(len(X_final.columns))]

print("\n--- FINAL VIF SCORES ---")
print(vif_data)

# Save and download the VIF table
vif_filename = 'Final_VIF_Scores.csv'
vif_data.to_csv(vif_filename, index=False)
files.download(vif_filename)

print(f"\nSuccess! Both '{hq_filename}' and '{vif_filename}' have been downloaded.")
