import os
import csv

# Define folder paths
class_0_folder = "/home/pie_crusher/CNN_AI_REAL/FAKE"  # Fake
class_1_folder = "/home/pie_crusher/CNN_AI_REAL/REAL"  # Real
output_csv = "labels.csv"  # Output CSV file

def rename_images(folder, suffix):
    """Renames images by appending '_R' or '_F'."""
    renamed_files = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg"):  # Ensure it's a JPG file
            old_path = os.path.join(folder, filename)
            new_name = f"{os.path.splitext(filename)[0]}_{suffix}.jpg"
            new_path = os.path.join(folder, new_name)

            # Rename file
            os.rename(old_path, new_path)
            renamed_files.append(new_name)
    
    return renamed_files

# Rename files and get updated filenames
class_0_images = rename_images(class_0_folder, "F")  # Fake images -> _F
class_1_images = rename_images(class_1_folder, "R")  # Real images -> _R

# Create list of tuples with filename and label
labeled_data = [(img, 0) for img in class_0_images] + [(img, 1) for img in class_1_images]

# Save to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(labeled_data)

print(f"\nRenaming complete! CSV file '{output_csv}' created successfully!")
