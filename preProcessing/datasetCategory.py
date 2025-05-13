import pandas as pd
import os

df = pd.read_csv('DASHlink_full_fourclass_raw_meta.csv')

save_directory = 'datasetsByArrivalAirport'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

arrival_airports = df['arrival_airport'].unique()

airport_data_counts = []

for airport in arrival_airports:
    airport_data = df[df['arrival_airport'] == airport]
    
    count = len(airport_data)
    
    label_counts = airport_data['label'].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    
    airport_data_counts.append({
        'arrival_airport': airport,
        'data_count': count,
        'label_0_count': label_counts[0],
        'label_1_count': label_counts[1],
        'label_2_count': label_counts[2],
        'label_3_count': label_counts[3]
    })
    
    filename = os.path.join(save_directory, f"{airport}.csv")
    
    airport_data.to_csv(filename, index=False)

counts_df = pd.DataFrame(airport_data_counts)

counts_filename = os.path.join(save_directory, "airport_data_counts.csv")
counts_df.to_csv(counts_filename, index=False)

excel_filename = os.path.join(save_directory, "airport_data_counts.xlsx")
counts_df.to_excel(excel_filename, index=False)

print("That's OK！")
