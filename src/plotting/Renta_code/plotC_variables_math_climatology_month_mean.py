import pandas as pd
import glob
import os

file_patterns = [
    r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Amazon/new_data_csv/means/plotA/*.csv",
    r"C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Amazon/new_data_csv/pc-95.0/plotA/*.csv"
]

for pattern in file_patterns:
    csv_files = glob.glob(pattern)

    for file_path in csv_files:
        df = pd.read_csv(file_path, header=None, skipfooter=3, engine='python')
        
        # Ensure all values are numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        monthly_means = []
        for i in range(12): #loop over months
            month_columns = df.iloc[:, i::12]
            month_mean_per_row = month_columns.mean(axis=1)
            monthly_means.append(month_mean_per_row)
        
        monthly_mean_df = pd.concat(monthly_means, axis=1)
        
    #copy first 2 columns to the end and last 4 to the beginning    
        first_2 = monthly_mean_df.iloc[:, :2]
        last_4 = monthly_mean_df.iloc[:, -4:]
        final_df = pd.concat([last_4, monthly_mean_df, first_2], axis=1)      
        
    #output file path
        output_path = file_path.replace("plotA", "plotC")
        base, ext = os.path.splitext(output_path)
        new_filename = f"{base}_monthly_climatology{ext}"
        
    #save
        final_df.to_csv(new_filename, index=False, header=False)