from datasets import load_dataset

# Load the dataset
ds = load_dataset("mteb/amazon_massive_intent", "en")


# Define a function to filter the columns
def filter_columns(example):
    return {'label': example['label'], 'text': example['text']}


# Apply the function to the dataset
filtered_dataset = ds.map(filter_columns)

# Save the filtered dataset
filtered_dataset.save_to_disk('./datasets')
