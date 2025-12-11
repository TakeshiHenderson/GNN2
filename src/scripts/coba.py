import os
import glob

lg_dir = "/home/takeshi/Documents/AOL DL/crohme_dataset/valid/lg_new_1"
lg_files = glob.glob(os.path.join(lg_dir, "*.lg"))

zero_relation_files = []

for file_path in lg_files:
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Count lines that start with 'EO,'
        relation_lines = [line for line in lines if line.strip().startswith("EO,")]
        if len(relation_lines) == 0:
            zero_relation_files.append(file_path)

print("Files with 0 relations:")
for fname in zero_relation_files:
    print(fname)

print(f"Total files with 0 relations: {len(zero_relation_files)}")