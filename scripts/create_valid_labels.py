import os
import pandas as pd

# Read training CSV
train_csv = r"T:\bone_can_pre\dataset\train\_classes.csv"
df = pd.read_csv(train_csv)
df.columns = [c.strip().lower() for c in df.columns]

# Get all files in valid directory
valid_dir = r"T:\bone_can_pre\dataset\valid"
valid_files = set(os.listdir(valid_dir))
valid_files = {f for f in valid_files if f.endswith(('.jpg', '.png', '.jpeg'))}

# Filter rows that exist in valid directory
valid_rows = df[df['filename'].str.strip().isin(valid_files)]

# Save to valid directory
output_path = os.path.join(valid_dir, "_classes.csv")
valid_rows.to_csv(output_path, index=False)

print(f"Created {output_path} with {len(valid_rows)} entries")
print(f"Cancer: {valid_rows['cancer'].sum()}, Normal: {valid_rows['normal'].sum()}")
