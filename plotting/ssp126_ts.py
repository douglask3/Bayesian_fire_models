
import os
import glob
import numpy as np
import iris
import matplotlib.pyplot as plt

# --------------------------------------------------
# Settings
# --------------------------------------------------
base_dir = (
    "data/data/driving_data2526/Northwest_Iberia/"
    "isimp3b/ssp126"
)

models = [
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "UKESM1-0-LL",
]

period = "period_2015_2099"
out_dir = "multi_model_spread"
os.makedirs(out_dir, exist_ok=True)

# Find variables available for all models
files_by_model = {
    model: {
        os.path.basename(f)
        for f in glob.glob(
            os.path.join(base_dir, model, period, "*.nc")
        )
    }
    for model in models
}

variables = sorted(set.intersection(
    *(files_by_model[m] for m in models)
))

# --------------------------------------------------
# Process each variable
# --------------------------------------------------
for variable in variables:
    print(variable)
    model_series = []
    dates = None

    for model in models:
        filename = os.path.join(
            base_dir, model, period, variable
        )

        cube = iris.load_cube(filename)

        # Convert time coordinate to dates
        time_coord = cube.coord("time")
        model_dates = time_coord.units.num2date(
            time_coord.points
        )

        if dates is None:
            dates = time_coord.units.num2date(  
                time_coord.points,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True
            )
        elif len(model_dates) != len(dates):
            raise ValueError(
                f"Different time lengths for {variable}"
            )

        # Spatial cosine-latitude weighting
        lat = cube.coord("latitude").points
        weights = np.cos(np.deg2rad(lat))[:, None]

        data = np.ma.asarray(cube.data)

        # Regional mean at each timestep
        regional_mean = np.ma.average(
            data,
            axis=(1, 2),
            weights=np.broadcast_to(
                weights, data.shape[1:]
            )
        )

        model_series.append(regional_mean)

    # Shape: (model, time)
    data = np.ma.stack(model_series)

    # Multi-model statistics at each timestep
    mean = np.ma.mean(data, axis=0)
    std = np.ma.std(data, axis=0)
    minimum = np.ma.min(data, axis=0)
    maximum = np.ma.max(data, axis=0)
    from pdb import set_trace
    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    #set_trace()
    ax1.plot(
        dates, mean,
        color="black", label="Multi-model mean"
    )
    ax1.fill_between(
        dates, minimum, maximum,
        alpha=0.25, label="Min–max range"
    )

    ax1.set_ylabel(
        f"{variable.replace('.nc', '')}"
    )
    ax1.set_title(
        f"{variable.replace('.nc', '')}: "
        "ISIMIP3b SSP126"
    )
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(dates, std)
    ax2.set_ylabel("Model standard deviation")
    ax2.set_xlabel("Year")
    ax2.grid(alpha=0.25)

    fig.tight_layout()

    output = os.path.join(
        out_dir,
        variable.replace(".nc", ".png")
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"Processed {len(variables)} variables.")
print(f"Figures saved in {out_dir}/")

