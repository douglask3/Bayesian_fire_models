import numpy as np

def climtatology_difference(data, nmths = 12):

    # `data` is your (n x m) array
    n, m = data.shape

    # Step 1: Calculate month index for each column (0=Jan, 11=Dec)
    month_indices = np.arange(m) % nmths  # shape (m,)
    
    # Step 2: Create an empty (n x 12) array to hold monthly means
    climatology = np.zeros((n, nmths))
    
    # Step 3: Fill in the monthly means
    for month in range(nmths):
        climatology[:, month] = data[:, month_indices == month].mean(axis=1)

    # Step 4: Create anomaly and ratio arrays (n x m)
    anomaly = np.zeros_like(data)
    ratio = np.zeros_like(data)
    
    for i in range(m):
        month = month_indices[i]
        anomaly[:, i] = data[:, i] - climatology[:, month]
        ratio[:, i] = data[:, i] / climatology[:, month]
    return climatology, anomaly, ratio

