import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Load the Excel sheet
df_airports = pd.read_excel('client/Client_Airport.xlsx')

# Load the .npz file containing all data instances
data_archive = np.load('DASHlink_full_fourclass_raw_comp.npz')
data_instances = data_archive['data']

# Define the base directory for storing the datasets
base_dir = 'Dataset_IID'

# Initialize a dictionary to store the count of successful CSV saves
successful_counts = {airport: [0, 0, 0, 0] for airport in df_airports['Airport']}

# Process each airport
for index, row in tqdm(df_airports.iterrows(), total=df_airports.shape[0], desc="Processing airports"):
    airport_name = row['Airport']
    client_id = row['Client']
    
    # Read the corresponding CSV file for the airport
    try:
        airport_csv = pd.read_csv(f'data_instance_output/{airport_name}.csv')
    except FileNotFoundError:
        print(f"Warning: No CSV file found for {airport_name}. Continuing to next airport.")
        continue

    # Client directory path
    client_dir = f'{base_dir}/local_{client_id}'

    # Ensure client directories and label subdirectories exist
    for label in range(4):
        label_dir = f'{client_dir}/{label}'
        os.makedirs(label_dir, exist_ok=True)
    
    # Extract and save data instances for each label
    for label in range(4):
        label_data = airport_csv[f'Label{label}'].dropna().astype(int)
        label_dir = f'{client_dir}/{label}'

        for i, instance_id in enumerate(label_data):
            if instance_id == -1:
                continue  # Skip this instance as it's marked with -1

            try:
                data_instance = data_instances[instance_id].reshape(160, 20)
                # Update filename format to include airport name
                filename = f'{label_dir}/{airport_name}_{instance_id}.csv'
                np.savetxt(filename, data_instance, delimiter=',', fmt='%g')
                successful_counts[airport_name][label] += 1  # Increment only on successful save
            except Exception as e:
                print(f"Error processing instance {instance_id} at {filename}: {e}")

