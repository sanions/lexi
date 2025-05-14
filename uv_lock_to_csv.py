# To call this, use the following form in gitbash 
# python uv_lock_to_csv.py uv.lock packages.csv

import csv
import sys
import os

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # For Python <3.11, install with `pip install tomli`

def extract_packages_from_uv_lock(lock_path, output_csv):
    if not os.path.exists(lock_path):
        print(f"Error: File {lock_path} not found.")
        return

    with open(lock_path, "rb") as f:
        data = tomllib.load(f)

    packages = data.get("package", [])
    if not packages:
        print("No packages found in uv.lock file.")
        return

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["package", "version"])
        for pkg in packages:
            name = pkg.get("name")
            version = pkg.get("version")
            if name and version:
                writer.writerow([name, version])

    print(f"✅ CSV written to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python uv_lock_to_csv.py <path_to_uv.lock> <output_csv_path>")
    else:
        extract_packages_from_uv_lock(sys.argv[1], sys.argv[2])
