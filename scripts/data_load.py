import os
from datasets import load_dataset
from tqdm import tqdm

# Load the data
dataset = load_dataset("wikipedia", "20220301.en")

# Define subset sizes
sizes = {
    "tiny": 50,
    "very_small": 500,
    "small": 1000,
    "medium": 5000,
    "medium_large": 10000,
    "large": 50000,
    "very_large": 100000,
    "huge": 1000000,
    "enormous": len(dataset['train'])  # get the total number of entries
}

subsets = {}
start_index = 0
dataset_length = len(dataset['train'])

# Create subsets based on the sizes
print("Creating subsets...")
for name, size in tqdm(sizes.items()):
    # Ensure we're not trying to select more data than available
    if start_index >= dataset_length:
        break

    end_index = start_index + size
    # Ensure the end index doesn't exceed the dataset length
    end_index = min(end_index, dataset_length)

    subsets[name] = dataset['train'].select(range(start_index, end_index))
    start_index = end_index

# Save each entry of each subset in its own file
print("Saving subsets...")
base_directory = r"C:\Users\Pascal\Desktop\WannaDB\lab23_wannadb_scale-up\datasets\wikipedia"
for name, subset in tqdm(subsets.items()):
    # Define the path for each subset
    folder_path = os.path.join(base_directory, name)
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Write the content of each subset entry to its own text file
    for idx, entry in enumerate(subset):
        file_name = os.path.join(folder_path, f"{name}_{idx}.txt")
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(entry['text'] + "\n\n")  # Two line breaks between entries

print("Subsets saved!")




