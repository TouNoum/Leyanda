# Imports
import os
from PIL import Image
from tqdm.notebook import tqdm

# Data format fixes
def data_formats_fixes(raw_data_path):
    """
    Walks through a directory to detect and remove problematic image files.
    Removes:
    - Files that are not actually JPG format
    - Corrupted or unreadable images
    Converts:
    - Invalid shape files to RGB format
    Parameters:
    - raw_data_path: Path to the raw data folder.
    """
    print(f"--Starting data format fixes--")

    stats = {
        "processed": 0,
        "wrong_format_removed": 0,
        "invalid_shape_converted": 0, # Includes grayscale
        "corrupted_removed": 0,
        "valid_images": 0
    }

    total_files = 0
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':
                total_files += 1

    with tqdm(total=total_files, desc="Checking images") as pbar:
        for root, dirs, files in os.walk(raw_data_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, extension = os.path.splitext(file)

                if extension.lower() != '.jpg':
                    continue

                stats["processed"] += 1
                pbar.update(1)

                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(4)

                    if header[:2] != b'\xff\xd8':  # Not a valid JPEG header
                        os.remove(file_path)
                        stats["wrong_format_removed"] += 1
                        continue

                    try:
                        with Image.open(file_path) as img:
                            if img.mode != 'RGB':
                                img_rgb = img.convert('RGB')
                                img_rgb.save(file_path, 'JPEG', quality=95)
                                stats["invalid_shape_converted"] += 1
                                stats["valid_images"] += 1
                                continue

                            stats["valid_images"] += 1

                    except Exception as e:
                        os.remove(file_path)
                        stats["corrupted_removed"] += 1

                except Exception as e:
                    os.remove(file_path)
                    stats["corrupted_removed"] += 1

    print(f"\nSummary:")
    print(f"Files processed: {stats['processed']}")
    print(f"Wrong format files removed: {stats['wrong_format_removed']}")
    print(f"Invalid shape files converted: {stats['invalid_shape_converted']}")
    print(f"Corrupted files removed: {stats['corrupted_removed']}")
    print(f"Valid images remaining: {stats['valid_images']}")
    print(f"Data format fixes completed.")
