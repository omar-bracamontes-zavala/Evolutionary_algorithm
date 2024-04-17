import json

def load_json_data(filepath):
    """Load JSON data from a specified filepath."""
    with open(filepath, 'r') as file:
        return json.load(file)

def increment_index(data, index, shift):
    """Increment the specified index for each design in data by a shift."""
    for design in data:
        design['combination_indexes'][index] += shift

def remove_unwanted_data(data):
    """Remove data entries where 'performance_metrics' is a string."""
    indices_to_delete = [i for i, design in enumerate(data) if isinstance(design['performance_metrics'], str)]
    # Sort indices in descending order to avoid shifting issues
    indices_to_delete.sort(reverse=True)
    for index in indices_to_delete:
        del data[index]

def save_data(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file)

# Load mu data and increment indexes
mu_data = load_json_data('results/results_mu.json')
increment_index(mu_data, 0, 1)  # selection functions are in column 0 and shifted 1 place
increment_index(mu_data, -1, 2)  # replacement functions are in column -1 and shifted 2 places

# Load results data and clean it
data = load_json_data('results/results.json')
remove_unwanted_data(data)

# Join mu_data with cleaned data
combined_data = data + mu_data

# Save the combined data
save_data(combined_data, 'results/cleaned_results.json')

# Print lengths of data for verification
print(f"Length of mu_data: {len(mu_data)}")
print(f"Length of cleaned data: {len(data)}")
print(f"Total length of combined data: {len(combined_data)}")
