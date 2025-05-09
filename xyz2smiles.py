import os
import csv
from openbabel import openbabel as ob
from openbabel import pybel

def xyz_to_smiles(xyz_file_path):
    """Convert a single XYZ file to a pure SMILES string without comments"""
    # Using pybel makes it easier to get pure SMILES
    try:
        # Read XYZ file
        molecule = next(pybel.readfile("xyz", xyz_file_path))
        
        # Get the pure SMILES string
        smiles = molecule.write("smi").strip()
        
        # If SMILES contains spaces (such as possible comments or other information at the end), take only the first part
        smiles = smiles.split()[0]
        
        return smiles
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return ""

def batch_convert_xyz_to_smiles(input_dir, output_csv):
    """Batch convert all XYZ files in a directory"""
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Get all XYZ files
    xyz_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xyz')]
    
    if not xyz_files:
        print(f"Warning: No XYZ files found in directory '{input_dir}'")
        return
    
    # Convert files and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'SMILES'])
        
        converted_count = 0
        for xyz_file in xyz_files:
            try:
                xyz_path = os.path.join(input_dir, xyz_file)
                smiles = xyz_to_smiles(xyz_path)
                if smiles:
                    csv_writer.writerow([xyz_file, smiles])
                    print(f"Converted: {xyz_file} -> {smiles}")
                    converted_count += 1
                else:
                    print(f"Warning: Could not obtain valid SMILES from '{xyz_file}'")
            except Exception as e:
                print(f"Error converting file '{xyz_file}': {str(e)}")
    
    print(f"Completed! Successfully converted {converted_count}/{len(xyz_files)} files, results saved to '{output_csv}'")

# Example usage
if __name__ == "__main__":
    # Set the input directory and output CSV file
    input_directory = "./xyz_files"  # Modify to your XYZ files directory
    output_csv_file = "./xyz_to_smiles_results.csv"  # Output CSV file path
    
    # Perform batch conversion
    batch_convert_xyz_to_smiles(input_directory, output_csv_file)