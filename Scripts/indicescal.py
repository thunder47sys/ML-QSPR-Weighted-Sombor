# @title Install RDKit and Calculate Indices (Fixed for Excel Files)

# Step 1: Install RDKit
try:
    import rdkit
    print("RDKit is already installed.")
except ImportError:
    print("Installing RDKit...")
    !pip install rdkit

import pandas as pd
import numpy as np
from rdkit import Chem
import math
from google.colab import files
import io
import os

# --------------------------------------------------------------------------------
# 1. UPLOAD DATA (With Auto-Detection for Excel vs CSV)
# --------------------------------------------------------------------------------
print("\n--- UPLOAD STEP ---")
print("Please upload your file (Final_Unique_200_Pharmaceuticals 15-2-26.xlsx)")
uploaded = files.upload()

# Get the filename
filename = next(iter(uploaded))
print(f"Processing file: {filename}")

# Check extension and read accordingly
try:
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        print("Detected Excel file. Reading with pd.read_excel...")
        df = pd.read_excel(io.BytesIO(uploaded[filename]))
    else:
        print("Detected CSV file. Reading with pd.read_csv...")
        # Try default UTF-8 first
        try:
            df = pd.read_csv(io.BytesIO(uploaded[filename]))
        except UnicodeDecodeError:
            print("UTF-8 failed. Retrying with Latin1 encoding...")
            df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')

    print(f"Success! Loaded {len(df)} compounds.")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: Could not read the file. \nError details: {e}")
    # Stop execution if data load fails
    raise

# --------------------------------------------------------------------------------
# 2. EXACT ATOMIC PROPERTY DICTIONARIES
# --------------------------------------------------------------------------------

# Atomic Mass (m)
mass = {'H': 1.008, 'Li': 6.94, 'Be': 9.01, 'B': 10.81, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00,
        'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.06, 'Cl': 35.45, 'K': 39.10, 
        'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00, 'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 
        'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.97, 'Br': 79.90, 
        'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95, 'Tc': 98.00, 'Ru': 101.07, 
        'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 
        'Te': 127.60, 'I': 126.90, 'Gd': 157.25, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 
        'Pb': 207.2, 'Bi': 208.98}

# Electronegativity (en)
en = {'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 
      'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'K': 0.82, 
      'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 
      'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 
      'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Tc': 1.90, 'Ru': 2.20, 
      'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 
      'I': 2.66, 'Gd': 1.20, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02}

# Ionization Potential (ip)
ip = {'H': 13.598, 'Li': 5.392, 'Be': 9.323, 'B': 8.298, 'C': 11.260, 'N': 14.534, 'O': 13.618, 'F': 17.423,
      'Na': 5.139, 'Mg': 7.646, 'Al': 5.986, 'Si': 8.152, 'P': 10.487, 'S': 10.360, 'Cl': 12.968, 'K': 4.341, 
      'Ca': 6.113, 'Sc': 6.561, 'Ti': 6.828, 'V': 6.746, 'Cr': 6.767, 'Mn': 7.434, 'Fe': 7.902, 'Co': 7.881, 
      'Ni': 7.640, 'Cu': 7.726, 'Zn': 9.394, 'Ga': 5.999, 'Ge': 7.899, 'As': 9.789, 'Se': 9.752, 'Br': 11.814, 
      'Rb': 4.177, 'Sr': 5.695, 'Y': 6.217, 'Zr': 6.634, 'Nb': 6.759, 'Mo': 7.092, 'Tc': 7.280, 'Ru': 7.361, 
      'Rh': 7.459, 'Pd': 8.337, 'Ag': 7.576, 'Cd': 8.994, 'In': 5.786, 'Sn': 7.344, 'Sb': 8.608, 'Te': 9.010, 
      'I': 10.451, 'Gd': 6.150, 'Pt': 8.959, 'Au': 9.226, 'Hg': 10.438, 'Tl': 6.108, 'Pb': 7.417, 'Bi': 7.286}

# Atomic Radius (r)
radius = {'H': 53, 'Li': 167, 'Be': 112, 'B': 87, 'C': 67, 'N': 56, 'O': 48, 'F': 42, 
          'Na': 190, 'Mg': 145, 'Al': 118, 'Si': 111, 'P': 98, 'S': 88, 'Cl': 79, 'K': 243, 
          'Ca': 194, 'Sc': 184, 'Ti': 176, 'V': 171, 'Cr': 166, 'Mn': 161, 'Fe': 156, 'Co': 152, 
          'Ni': 149, 'Cu': 145, 'Zn': 142, 'Ga': 136, 'Ge': 125, 'As': 114, 'Se': 103, 'Br': 94, 
          'Rb': 265, 'Sr': 219, 'Y': 212, 'Zr': 206, 'Nb': 198, 'Mo': 190, 'Tc': 183, 'Ru': 178, 
          'Rh': 173, 'Pd': 169, 'Ag': 165, 'Cd': 161, 'In': 156, 'Sn': 145, 'Sb': 133, 'Te': 123, 
          'I': 115, 'Gd': 233, 'Pt': 177, 'Au': 174, 'Hg': 171, 'Tl': 156, 'Pb': 154, 'Bi': 143}

# Carbon Reference Values (w_c)
carbon_refs = {
    'mass': 12.01,
    'en': 2.55,
    'ip': 11.26,
    'radius': 67.0
}

# --------------------------------------------------------------------------------
# 3. CALCULATION LOGIC (Exact Formula)
# --------------------------------------------------------------------------------

def calculate_weighted_sombor(mol, prop_dict, prop_name):
    try:
        # A. Weighted Degree Calculation (Carbon Reference Method)
        
        # Determine w_c
        w_c = carbon_refs.get(prop_name, 1.0)
        
        degrees = {}
        
        # Calculate Weighted Degree for each atom
        for atom in mol.GetAtoms():
            d_w = 0
            idx = atom.GetIdx()
            sym_i = atom.GetSymbol()
            w_i = prop_dict.get(sym_i, 0)
            
            for neighbor in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                bo = bond.GetBondTypeAsDouble()
                sym_j = neighbor.GetSymbol()
                w_j = prop_dict.get(sym_j, 0)
                
                # The Formula: dw = sum( wc^2 / (BO * wi * wj) )
                if bo != 0 and w_i != 0 and w_j != 0:
                    d_w += (w_c * w_c) / (bo * w_i * w_j)
            
            degrees[idx] = d_w

        # B. Geometric Sombor Indices
        SO = 0; SO3 = 0; SO4 = 0; SO5 = 0; SO6 = 0; ESO = 0; MESO = 0
        
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            
            du = degrees[u]
            dv = degrees[v]
            
            # Common Terms
            term = math.sqrt(du**2 + dv**2)
            sum_sq = du**2 + dv**2
            sum_deg = du + dv
            abs_diff_sq = abs(du**2 - dv**2)
            denom_so5 = (math.sqrt(2) + 2 * term)
            
            # 1. SO
            SO += term
            
            # 2. SO3
            if sum_deg != 0:
                SO3 += (math.sqrt(2) * math.pi * (sum_sq / sum_deg))
                
            # 3. SO4
            if sum_deg != 0:
                SO4 += ((math.pi / 2) * ((sum_sq / sum_deg) ** 2))
                
            # 4. SO5
            if denom_so5 != 0:
                SO5 += (2 * math.pi * (abs_diff_sq / denom_so5))
                
            # 5. SO6
            if denom_so5 != 0:
                SO6 += (math.pi * ((abs_diff_sq / denom_so5) ** 2))
                
            # 6. ESO
            ESO += (sum_deg * term)
            
        # 7. MESO
        if ESO != 0:
            MESO = 1 / ESO
            
        return {
            f'{prop_name}_SO': SO,
            f'{prop_name}_SO3': SO3,
            f'{prop_name}_SO4': SO4,
            f'{prop_name}_SO5': SO5,
            f'{prop_name}_SO6': SO6,
            f'{prop_name}_ESO': ESO,
            f'{prop_name}_MESO': MESO
        }

    except Exception as e:
        return {k: None for k in [f'{prop_name}_SO', f'{prop_name}_SO3', f'{prop_name}_SO4', f'{prop_name}_SO5', f'{prop_name}_SO6', f'{prop_name}_ESO', f'{prop_name}_MESO']}

# --------------------------------------------------------------------------------
# 4. EXECUTION
# --------------------------------------------------------------------------------

print("\n--- CALCULATION STEP ---")
results = []
# Ensure we find the smiles column even if case differs
smiles_col = 'SMILES' 
for col in df.columns:
    if col.upper().strip() == 'SMILES':
        smiles_col = col
        break

print(f"Using column '{smiles_col}' for SMILES.")

for index, row in df.iterrows():
    smiles = row[smiles_col]
    mol = Chem.MolFromSmiles(str(smiles))
    
    if mol:
        mol = Chem.AddHs(mol)
        comp_data = {}
        
        # Calculate for all 4 properties
        comp_data.update(calculate_weighted_sombor(mol, mass, 'mass'))
        comp_data.update(calculate_weighted_sombor(mol, en, 'en'))
        comp_data.update(calculate_weighted_sombor(mol, ip, 'ip'))
        comp_data.update(calculate_weighted_sombor(mol, radius, 'radius'))
        
        results.append(comp_data)
    else:
        results.append({})

# Create DataFrame
results_df = pd.DataFrame(results)
final_df = pd.concat([df, results_df], axis=1)

# Save and Download
output_filename = 'Data_Alpha_Geometric_Weighted_Sombor_Indices.csv'
final_df.to_csv(output_filename, index=False)
print(f"\n--- SUCCESS ---")
print(f"Calculated indices for {len(final_df)} compounds.")
print(f"Downloading {output_filename}...")
files.download(output_filename)
