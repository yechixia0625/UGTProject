import os
import pandas as pd

excel_file = 'Airport_Label.xlsx'
dataset_folder = 'datasetsByArrivalAirport' 
output_folder = 'data_instance_noniid_simplex'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

excel_df = pd.read_excel(excel_file)

for index, row in excel_df.iterrows():
    airport = row['Airport']
    label_counts = {f'Label{i}': int(row[f'Label{i}']) for i in range(4)}
    
    airport_file = os.path.join(dataset_folder, f"{airport}.csv")
    if not os.path.exists(airport_file):
        print(f"File {airport_file} not existence, skipping...")
        continue
    
    airport_df = pd.read_csv(airport_file)
    
    label_data = {}
    for label, count in label_counts.items():
        label_index = int(label[-1])
        label_df = airport_df[airport_df['label'] == label_index]
        
        if len(label_df) < count:
            print(f"Caution: {airport} 's {label} available data is less than the requested quantity. Only output {len(label_df)} data instance")
            count = len(label_df)
        
        label_data[label] = label_df['data_instance'].head(count).reset_index(drop=True)
    
    output_df = pd.DataFrame(label_data)

    for column in output_df.columns:
        if output_df[column].isnull().any():
            print(f"Caution: Column {column} contains NaN values and will be filled with -1.")
            output_df[column] = output_df[column].fillna(-1)
        output_df[column] = output_df[column].astype(int)

    output_file = os.path.join(output_folder, f"{airport}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"{airport}'s data has been saved to {output_file}")
