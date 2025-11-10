import os
import pandas as pd

# Get all files in valid directory
valid_dir = r"T:\bone_can_pre\dataset\valid"
valid_files = [f for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Create labels based on filename patterns
# Cancer indicators: bone-cancer, osteosarcoma, ewing, metastasis, etc.
# Normal indicators: normal, IMG0000, Picture, etc.
rows = []
for fname in valid_files:
    fname_lower = fname.lower()
    
    # Determine if cancer or normal based on filename patterns
    is_cancer = any(keyword in fname_lower for keyword in [
        'bone-cancer', 'osteosarcoma', 'ewing', 'metastasis', 
        'chondrosarcoma', 'fibrous', 'sarcoma', 'tumor', 
        'malignant', 'cancer'
    ])
    
    is_normal = any(keyword in fname_lower for keyword in [
        'normal', 'img0', 'picture', 'istockphoto'
    ]) or (fname_lower.startswith(('-', 'pelvis', 'foot', 'hand', 'ankle', 
                                     'elbow', 'forearm', 'chest', 'tf_', 
                                     'cervical-spine')) and 'other' not in fname_lower)
    
    # Default to cancer if ambiguous (most files are cancer in medical datasets)
    if is_cancer and not is_normal:
        cancer, normal = 1, 0
    elif is_normal and not is_cancer:
        cancer, normal = 0, 1
    elif is_cancer and is_normal:
        # Check for explicit "normal" in filename
        cancer, normal = (0, 1) if 'normal' in fname_lower else (1, 0)
    else:
        # Check anatomical location patterns
        anatomical_normal = any(fname_lower.startswith(prefix) for prefix in [
            'pelvis', 'foot', 'hand', 'ankle', 'elbow', 'forearm', 'chest', 'tf_'
        ])
        cancer, normal = (0, 1) if anatomical_normal else (1, 0)
    
    rows.append({'filename': fname, 'cancer': cancer, 'normal': normal})

# Create DataFrame
df = pd.DataFrame(rows)

# Save
output_path = os.path.join(valid_dir, "_classes.csv")
df.to_csv(output_path, index=False)

print(f"Created {output_path} with {len(df)} entries")
print(f"Cancer: {df['cancer'].sum()}, Normal: {df['normal'].sum()}")
print(f"\nSample cancer files:")
print(df[df['cancer'] == 1]['filename'].head(5).tolist())
print(f"\nSample normal files:")
print(df[df['normal'] == 1]['filename'].head(5).tolist())
