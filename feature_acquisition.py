import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, AllChem, rdMolDescriptors
from rdkit.Chem.Fragments import fr_Al_OH, fr_C_O, fr_C_O_noCOO, fr_ester, fr_COO2

# Read CSV file
df = pd.read_csv('ZWY.csv')

# Define function to calculate molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 3D coordinate generation - for certain descriptor calculations
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        print(f"Unable to generate 3D coordinates: {smiles}")
    
    descriptors = {
        # Basic physicochemical properties
        'MolWt': Descriptors.MolWt(mol),  # Molecular weight
        'ExactMolWt': Descriptors.ExactMolWt(mol),  # Exact molecular weight
        'LogP': Crippen.MolLogP(mol),  # Partition coefficient
        'TPSA': MolSurf.TPSA(mol),  # Topological polar surface area
        'LabuteASA': MolSurf.LabuteASA(mol),  # Labute surface area
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),  # Number of hydrogen bond acceptors
        'NumHDonors': Lipinski.NumHDonors(mol),  # Number of hydrogen bond donors
        
        # Molecular topology features
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # Number of rotatable bonds
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),  # Number of aromatic rings
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),  # Number of saturated rings
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),  # Number of aliphatic rings
        'NumRings': Descriptors.RingCount(mol),  # Total number of rings
        'FractionCSP3': Descriptors.FractionCSP3(mol),  # Ratio of SP3 hybridized carbon atoms
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),  # Number of heavy atoms
        
        # Flexibility and rigidity indicators
        'RotatableBondRatio': Descriptors.NumRotatableBonds(mol) / max(1, mol.GetNumBonds()),  # Rotatable bond ratio
        'NumBonds': mol.GetNumBonds(),  # Total number of bonds
        
        # Functional group counts
        'NumHydroxyl': fr_Al_OH(mol),  # Number of hydroxyl groups
        'NumEster': fr_ester(mol),  # Number of ester groups
        'NumCarboxylate': fr_COO2(mol),  # Number of carboxylate groups
        'NumEther': fr_C_O_noCOO(mol),  # Number of ether bonds
        'NumCO': fr_C_O(mol),  # Number of C=O bonds
        
        # Connectivity and topology indices
        'BalabanJ': Descriptors.BalabanJ(mol),  # Balaban J index
        'BertzCT': Descriptors.BertzCT(mol),  # Bertz CT index
        'Chi0': Descriptors.Chi0(mol),  # 0-order connectivity index
        'Chi1': Descriptors.Chi1(mol),  # 1-order connectivity index
        'Kappa1': Descriptors.Kappa1(mol),  # 1-order kappa shape index
    }
    
    # Calculate additional special functional group features
    # Count specific functional groups using SMARTS patterns
    try:
        # Disulfide bonds
        disulfide_pattern = Chem.MolFromSmarts('S-S')
        if disulfide_pattern:
            descriptors['NumDisulfide'] = len(mol.GetSubstructMatches(disulfide_pattern))
            
        # Carbonate groups
        carbonate_pattern = Chem.MolFromSmarts('C(=O)OC(=O)')
        if carbonate_pattern:
            descriptors['NumCarbonate'] = len(mol.GetSubstructMatches(carbonate_pattern))
            
        # Aryl ethers
        aryl_ether_pattern = Chem.MolFromSmarts('c-[O;!$(O=*)]-[C,c]')
        if aryl_ether_pattern:
            descriptors['NumArylEther'] = len(mol.GetSubstructMatches(aryl_ether_pattern))
        
        # Benzene rings
        benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
        if benzene_pattern:
            descriptors['NumBenzene'] = len(mol.GetSubstructMatches(benzene_pattern))
        
        # Furan rings
        furan_pattern = Chem.MolFromSmarts('o1cccc1')
        if furan_pattern:
            descriptors['NumFuran'] = len(mol.GetSubstructMatches(furan_pattern))
            
    except Exception as e:
        print(f"Functional group calculation error: {e}")
    
    # Calculate crosslinking potential - density of functional groups that may participate in crosslinking reactions
    active_groups = descriptors.get('NumHydroxyl', 0) + descriptors.get('NumEster', 0) + descriptors.get('NumDisulfide', 0)
    descriptors['CrosslinkingPotential'] = active_groups / max(1, descriptors['HeavyAtomCount'])
    
    # Calculate aromaticity indicator - ratio of aromatic atoms
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    descriptors['AromaticAtomRatio'] = aromatic_atoms / max(1, descriptors['HeavyAtomCount'])
    
    return descriptors

# Process ratio data, adjust PTA ratio to actual dimer monomer ratio
def process_ratio(ratio_str):
    ratios = ratio_str.split(':')
    ratios = [float(r) for r in ratios]
    # Halve the ratio of PTA as it is a dimer
    ratios[2] = ratios[2] / 2
    total = sum(ratios)
    # Normalize ratios
    return [r/total for r in ratios]

# Calculate catalyst descriptors
def calculate_catalyst_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'Cat_MolWt': Descriptors.MolWt(mol),
        'Cat_LogP': Crippen.MolLogP(mol),
        'Cat_TPSA': MolSurf.TPSA(mol),
        'Cat_NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'Cat_NumHDonors': Lipinski.NumHDonors(mol),
        'Cat_NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'Cat_NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'Cat_NumRings': Descriptors.RingCount(mol),
        'Cat_Kappa1': Descriptors.Kappa1(mol),  # Shape index
        'Cat_Chi1': Descriptors.Chi1(mol),  # Connectivity index
    }
    
    # Detect metal atoms
    metal_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in 
                     ['Zn', 'Cu', 'Fe', 'Ni', 'Co', 'Pt', 'Pd', 'Ru', 'Rh', 'Mn', 'Cr', 'Mo', 'W', 'Ti', 'V'])
    descriptors['Cat_NumMetalAtoms'] = metal_atoms
    
    # Detect special structures in catalyst
    try:
        # Detect coordinating atoms - N and O often act as coordinating atoms
        n_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        s_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        descriptors['Cat_NumN'] = n_atoms
        descriptors['Cat_NumS'] = s_atoms
        descriptors['Cat_CoordinationPotential'] = n_atoms + s_atoms
        
        # Detect heterocyclic structures
        n_heterocycle_pattern = Chem.MolFromSmarts('*1[nN]*[nN]*1')
        if n_heterocycle_pattern:
            descriptors['Cat_N_Heterocycle'] = len(mol.GetSubstructMatches(n_heterocycle_pattern))
    
    except Exception as e:
        print(f"Error calculating catalyst features: {e}")
    
    return descriptors

# Initialize result DataFrame
result_df = df.copy()

# Calculate descriptors for each sample and perform weighted averaging
for index, row in df.iterrows():
    # Get monomer SMILES
    apolymer = row['Apolymer']
    bpolymer = row['Bpolymer']
    ptapolymer = row['PTApolymer']
    
    # Calculate monomer descriptors
    a_desc = calculate_descriptors(apolymer)
    b_desc = calculate_descriptors(bpolymer)
    pta_desc = calculate_descriptors(ptapolymer)
    
    # Process ratio
    ratios = process_ratio(row['A:B:PTA'])
    
    # Calculate weighted average descriptors
    if a_desc and b_desc and pta_desc:
        for key in a_desc:
            if key in b_desc and key in pta_desc:
                weighted_value = (a_desc[key] * ratios[0] + 
                                b_desc[key] * ratios[1] + 
                                pta_desc[key] * ratios[2])
                result_df.at[index, key] = weighted_value
    
    # Calculate catalyst descriptors
    cat_smiles = row['Cat']
    cat_desc = calculate_catalyst_descriptors(cat_smiles)
    if cat_desc:
        for key, value in cat_desc.items():
            result_df.at[index, key] = value
    
    # Add catalyst concentration as a feature
    result_df.at[index, 'Cat_Concentration'] = float(row['mol%'])

    # Add ratio features
    result_df.at[index, 'Ratio_A'] = ratios[0]
    result_df.at[index, 'Ratio_B'] = ratios[1]
    result_df.at[index, 'Ratio_PTA'] = ratios[2]
    
    # Calculate interaction potential between monomers
    if a_desc and b_desc:
        # Polarity difference - may affect compatibility
        polarity_diff_ab = abs(a_desc['TPSA'] - b_desc['TPSA'])
        result_df.at[index, 'Polarity_Diff_AB'] = polarity_diff_ab
        
        # LogP difference - hydrophilicity/hydrophobicity difference
        logp_diff_ab = abs(a_desc['LogP'] - b_desc['LogP'])
        result_df.at[index, 'LogP_Diff_AB'] = logp_diff_ab
        
        # Total crosslinking site density
        crosslink_potential = (a_desc['CrosslinkingPotential'] * ratios[0] + 
                              b_desc['CrosslinkingPotential'] * ratios[1] + 
                              pta_desc['CrosslinkingPotential'] * ratios[2])
        result_df.at[index, 'Total_Crosslink_Potential'] = crosslink_potential
        
        # Rigidity/flexibility balance - balance between aromatic atom ratio and rotatable bond ratio
        rigidity_flexibility_balance = (a_desc['AromaticAtomRatio'] / max(0.01, a_desc['RotatableBondRatio']) * ratios[0] + 
                                       b_desc['AromaticAtomRatio'] / max(0.01, b_desc['RotatableBondRatio']) * ratios[1] + 
                                       pta_desc['AromaticAtomRatio'] / max(0.01, pta_desc['RotatableBondRatio']) * ratios[2])
        result_df.at[index, 'Rigidity_Flexibility_Balance'] = rigidity_flexibility_balance

# Drop unnecessary original columns
columns_to_drop = ['Apolymer', 'Bpolymer', 'PTApolymer', 'A:B:PTA', 'Cat']
result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

# Save results to new CSV file
result_df.to_csv('polymer_features_enhanced.csv', index=False)

print("Enhanced feature extraction completed, results saved to polymer_features_enhanced.csv")
print(f"Extracted {len(result_df.columns) - 2} features.")  # Subtract No column and Mpa column