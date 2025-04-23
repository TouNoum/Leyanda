import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.preprocessing import image


def convert_to_binary_dataset_structure(source_path, target_binary_class, output_path=None, max_workers=8):
    """
    Optimized conversion to binary dataset structure using multithreading.
    Parameters:
    - source_path (str): Path to the source dataset directory
    - target_binary_class (str): Name of the target binary class
    - output_path (str, optional): Path to save the binary dataset
    - max_workers (int, optional): Number of threads to use for copying files
    Returns:
    - output_path (str): Path to the created binary dataset
    """
    print(f"\n--Converting dataset to binary format--")

    if not os.path.isdir(source_path):
        raise ValueError(f"Source path '{source_path}' does not exist or is not a directory.")

    class_dirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]

    if target_binary_class not in class_dirs:
        raise ValueError(f"Class '{target_binary_class}' not found in source directory. Found classes: {class_dirs}")

    if output_path is None:
        parent_dir = os.path.dirname(source_path)
        output_path = os.path.join(parent_dir, f"Dataset_binary_{target_binary_class}")
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping conversion.")
        return output_path

    pos_dir = os.path.join(output_path, target_binary_class)
    neg_dir = os.path.join(output_path, f"not_{target_binary_class}")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    copy_tasks = []

    for class_name in class_dirs:
        src_dir = os.path.join(source_path, class_name)
        dst_dir = pos_dir if class_name == target_binary_class else neg_dir

        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            if os.path.isfile(src_file):
                dst_file = os.path.join(dst_dir, filename)
                copy_tasks.append((src_file, dst_file))

    print(f"Total files to copy: {len(copy_tasks)}")

    def copy_file(task):
        src, dst = task
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            return src, str(e)
        return None

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_file, task) for task in copy_tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Copying files"):
            error = f.result()
            if error:
                errors.append(error)

    if errors:
        print(f"\n{len(errors)} errors occurred during copy:")
        for src, err in errors[:5]:  # Print only first 5 errors
            print(f"Error copying {src}: {err}")

    print(f"Binary dataset created in: {output_path}")
    print(f"Classes: {target_binary_class}, not_{target_binary_class}")
    return output_path