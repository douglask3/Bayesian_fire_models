
anoms_pos = anom_p10

for i in range(len(anoms_pos)):
    anoms_pos[i] = bilinear_interpolate_cube(anoms_pos[i])
    anoms_pos[i].data[anoms_pos[i].data < 0.0] = 0.0

#cube.data[cube.data < 0.0] = 0.0
# anoms_pos is your list of 6 cubes, one per control
data_stack = np.stack([cube.data for cube in anoms_pos], axis=-1)  # shape: (lat, lon, 6)
mask = np.any(data_stack > 0, axis=-1)  # Only include pixels with at least one > 0

# Flatten spatial dims
coords = np.argwhere(mask)
samples = data_stack[mask]  # shape: (n_valid_pixels, 6)

from sklearn.cluster import KMeans

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = -1 * np.ones(mask.shape, dtype=int)
labels_flat = kmeans.fit_predict(samples)
labels[mask] = labels_flat

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(max_depth=4, random_state=42)

clf.fit(samples, labels_flat)

# Visualize
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=[f'Standard_{i}' for i in range(6)], filled=True)
plt.title("Decision Tree Explaining Cluster Membership")
plt.show()

import iris

cluster_cube = anoms_pos[0].copy()
cluster_cube.data = labels  # shape: (lat, lon), labels -1 means not clustered
set_trace()
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

def plot_clusters(cluster_cube):
    from scipy.ndimage import zoom

    def upscale_categorical(data, factor=4):
        # Repeat nearest-neighbor values to upscale without blending
        return zoom(data, zoom=factor, order=0)  # order=0 → nearest neighbor

    # Upsample and plot
    upscaled_data = upscale_categorical(cluster_cube.data, factor=4)
    cmap = ListedColormap(["#ffffff", "#ee007f", "#0096a1", "#7a44ff", "#e98400"])
    
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    img = qplt.pcolormesh(cluster_cube, cmap=cmap, axes=ax)
    
    plt.colorbar(img, ax=ax, label='Cluster')
    ax.set_title("Clustered Positive Anomaly Patterns")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.RIVERS)
    plt.show()

def plot_clusters_smooth(cluster_cube):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import cartopy.crs as ccrs

    # Extract data
    data = cluster_cube.data
    lats = cluster_cube.coord('latitude').points
    lons = cluster_cube.coord('longitude').points

    # Create color map
    cmap = ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'])
    
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    # Use imshow with interpolation
    img = ax.imshow(
        data,
        origin='lower',
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap=cmap,
        interpolation='bilinear',  # try 'bilinear' for fuzzier look
        transform=ccrs.PlateCarree()
    )

    # Features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.RIVERS)
    plt.colorbar(img, ax=ax, label='Cluster')
    ax.set_title("Clustered Anomaly Patterns (Smoothed Display)")
    plt.show()

plot_clusters(cluster_cube)

