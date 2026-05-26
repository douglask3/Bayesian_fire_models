import pandas as pd
import numpy as np

moisture_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/3_points-standard-Moisture_copy.csv', header=None, skipfooter=3, engine='python')
moisture_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/3_points-standard-Moisture_copy.csv', header=None, skipfooter=3, engine='python')
fuel_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/2_points-standard-Fuel_copy.csv', header=None, skipfooter=3, engine='python')
fuel_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/2_points-standard-Fuel_copy.csv', header=None, skipfooter=3, engine='python')

wind_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/7_points-standard-Wind_copy.csv', header=None, skipfooter=3, engine='python')
wind_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/7_points-standard-Wind_copy.csv', header=None, skipfooter=3, engine='python')
weather_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/6_points-standard-Weather_copy.csv', header=None, skipfooter=3, engine='python')
weather_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/6_points-standard-Weather_copy.csv', header=None, skipfooter=3, engine='python')

ignition_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/4_points-standard-Ignition_copy.csv', header=None, skipfooter=3, engine='python')
ignition_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/4_points-standard-Ignition_copy.csv', header=None, skipfooter=3, engine='python')
suppression_means = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotA/5_points-standard-Suppression_copy.csv', header=None, skipfooter=3, engine='python')
suppression_pc95 = pd.read_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotA/5_points-standard-Suppression_copy.csv', header=None, skipfooter=3, engine='python')

sqrt_fuel_moisture_means = np.sqrt(moisture_means*fuel_means)
sqrt_fuel_moisture_pc95 = np.sqrt(moisture_pc95*fuel_pc95)

sqrt_wind_weather_means = np.sqrt(wind_means*weather_means)
sqrt_wind_weather_pc95 = np.sqrt(wind_pc95*weather_pc95)

sqrt_human_means = np.sqrt(ignition_means*suppression_means)
sqrt_human_pc95 = np.sqrt(ignition_pc95*suppression_pc95)

sqrt_fuel_moisture_means.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotB/points-standard-Fuel_Moisture_sqrt.csv', header=False, index=False)
sqrt_fuel_moisture_pc95.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotB/points-standard-Fuel_Moisture_sqrt.csv', header=False, index=False)

sqrt_wind_weather_means.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotB/points-standard-Wind_Weather_sqrt.csv', header=False, index=False)
sqrt_wind_weather_pc95.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotB/points-standard-Wind_Weather_sqrt.csv', header=False, index=False)

sqrt_human_means.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/means/plotB/points-standard-Humans.csv', header=False, index=False)
sqrt_human_pc95.to_csv('C:/Users/rmvre/OneDrive/Documents/SoW_2024_2025/Congo/data_csv/pc-95.0/plotB/points-standard-Humans.csv', header=False, index=False)