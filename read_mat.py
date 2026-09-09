import os
import sys
import argparse
import numpy as np
import scipy.io
import h5py

def load_mat(filename):
    """
    Load .mat file using scipy.io.loadmat (for MATLAB v7.2 and earlier)
    or h5py (for MATLAB v7.3 format).
    Returns a dictionary of variables.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    data = {}
    # Try scipy.io.loadmat first
    try:
        mat_data = scipy.io.loadmat(filename)
        # Filter out MATLAB internal metadata keys (starting with '__')
        for k, v in mat_data.items():
            if not k.startswith('__'):
                data[k] = v
        return data, "scipy.io (v7.2 or earlier)"
    except (NotImplementedError, ValueError):
        # Fall back to h5py for v7.3 files
        try:
            with h5py.File(filename, 'r') as f:
                data = read_h5_group(f)
            return data, "h5py (MATLAB v7.3 / HDF5)"
        except Exception as e:
            raise RuntimeError(f"Failed to load MAT file '{filename}': {e}")

def read_h5_group(group):
    """Recursively read an h5py Group or File into a dictionary."""
    data = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            data[key] = item[()]
        elif isinstance(item, h5py.Group):
            data[key] = read_h5_group(item)
    return data

def inspect_content(data, indent=0):
    """
    Print the keys, types, dimensions (shapes), and summary of contents recursively.
    """
    spacing = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{spacing}Key: '{key}'")
                print(f"{spacing}  ├─ Type: numpy.ndarray ({value.dtype})")
                print(f"{spacing}  ├─ Shape / Dimensions: {value.shape}")
                if value.size <= 10:
                    print(f"{spacing}  └─ Content: {value.tolist()}")
                else:
                    print(f"{spacing}  └─ Content Preview (Min: {value.min()}, Max: {value.max()}, Mean: {value.mean():.4f})")
            elif isinstance(value, dict):
                print(f"{spacing}Group / Struct: '{key}'")
                inspect_content(value, indent + 4)
            else:
                print(f"{spacing}Key: '{key}'")
                print(f"{spacing}  ├─ Type: {type(value).__name__}")
                print(f"{spacing}  └─ Content: {value}")
    else:
        print(f"{spacing}Content: {data}")

def main():
    # parser = argparse.ArgumentParser(description="Read a .mat file, extract its content, and display key dimensions.")
    # parser.add_argument("mat_path", type=str, help="Path to the .mat file")
    # args = parser.parse_args()

    file_path = "cine_sax_kus_Uniform8.mat"
    print(f"Reading MAT file: {file_path}")
    print("=" * 60)

    try:
        contents, load_method = load_mat(file_path)
        print(f"Loaded successfully using {load_method}\n")
        print("Extracted Content & Dimensions:")
        print("-" * 60)
        inspect_content(contents)
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
