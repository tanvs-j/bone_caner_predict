import os
from PIL import Image
from tqdm import tqdm

def check_and_clean_dataset(root_dir):
    """Check for corrupted images and remove them"""
    corrupted_files = []
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in ['cancer', 'normal']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\nChecking {split}/{class_name}: {len(files)} files")
            
            for filename in tqdm(files, desc=f"{split}/{class_name}"):
                filepath = os.path.join(class_dir, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # Verify it's an image
                    # Try to actually load it
                    with Image.open(filepath) as img:
                        img.load()  # Force loading
                except Exception as e:
                    print(f"\n❌ Corrupted: {filepath}")
                    print(f"   Error: {e}")
                    corrupted_files.append(filepath)
                    try:
                        os.remove(filepath)
                        print(f"   ✓ Removed")
                    except Exception as remove_error:
                        print(f"   Failed to remove: {remove_error}")
    
    print(f"\n{'='*60}")
    print(f"Cleaning Summary")
    print(f"{'='*60}")
    print(f"Total corrupted files found: {len(corrupted_files)}")
    print(f"{'='*60}\n")
    
    if corrupted_files:
        print("Corrupted files:")
        for f in corrupted_files:
            print(f"  - {f}")

if __name__ == "__main__":
    dataset_root = r"T:\bone_can_pre\dataset\dataset"
    print(f"Scanning dataset: {dataset_root}\n")
    check_and_clean_dataset(dataset_root)
    print("\n✓ Dataset cleaning completed!")
