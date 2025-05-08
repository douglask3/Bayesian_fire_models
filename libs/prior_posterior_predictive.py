import matplotlib.pyplot as plt
import pymc as pm
import pytensor
import pytensor.tensor as tt
import numpy as np
import copy


def sample_priors(priors):
    """Recursively sample from a priors dict that may contain PyMC distributions or nested lists."""
    sampled = {}
    
    for key, val in priors.items():
        #set_trace()
        if isinstance(val, pytensor.tensor.variable.TensorVariable):
            sampled[key] = pm.draw(val)#val.eval() if hasattr(val, 'eval') else pm.draw(val)
        elif isinstance(val, list):
            sampled[key] = [
                pm.draw(v) if isinstance(v, pytensor.tensor.variable.TensorVariable) else v
                for v in val
            ]
        else:
            sampled[key] = val
    return sampled

def prior_predictive_check(model_class, X, Y, priors_template, n_samples=100):
    predictions = []
    
    for _ in range(n_samples):
        sampled_priors = sample_priors(priors_template)
        #set_trace()
        model = model_class(sampled_priors, inference=False)
        y_pred = model.burnt_area(X)  # Assumes it returns a 1D array of shape (len(Y),)
        #set_trace()
        predictions.append(y_pred)

    predictions = np.array(predictions)  # Shape: (n_samples, len(Y))
    return predictions


def plot_prior_predictive(predictions, Y, title="Prior Predictive Vases", jitter=0.01):
    Y = np.asarray(Y)
    preds = np.asarray(predictions)  # shape: (n_samples, len(Y))

    lower90 = np.percentile(preds, 5, axis=0)
    upper90 = np.percentile(preds, 95, axis=0)
    lower100 = np.min(preds, axis=0)
    upper100 = np.max(preds, axis=0)
    #set_trace()
    # Jitter Y values horizontally to avoid overplotting
    #rng = np.random.default_rng(42)
    #Y_jittered = Y + rng.normal(0, jitter, size=Y.shape)

    fig, ax = plt.subplots(figsize=(10, 6))



    for x, lo100, hi100, lo90, hi90 in zip(Y, lower100, upper100, lower90, upper90):
        ax.plot([x, x], [lo100, hi100], color='lightgray', alpha=0.7, lw=1)
        ax.plot([x, x], [lo90, hi90], color='steelblue', alpha=0.9, lw=2)

    ax.set_xlabel("Observed Burned Area Fraction")
    ax.set_ylabel("Prior Predicted Burned Area Fraction")
    ax.set_title(title)

    #  Apply pseudo-log scales to both axes
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.set_yscale('symlog', linthresh=1e-4)

    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    #plt.show()

def posterior_predictive_plot(pcc, variable = "fx_pred", dim = ["chain", "draw"]):
    # Get mean prediction for each point
    pred_mean = ppc.posterior_predictive[variable].mean(dim=dim).values
    # If fx_pred is 2D (samples x points), reshape accordingly

    # Assuming your observed Y is still in memory:
    plt.scatter(Y, pred_mean, alpha=0.5)
    plt.plot([0, 0.15], [0, 0.15], 'r--')  # Line of equality
    plt.xlabel("Observed Burned Fraction")
    plt.ylabel("Predicted Burned Fraction (Mean)")
    plt.title("Posterior Predictive vs Observed")
    plt.grid(True)
    plt.savefig(dir_outputs + "/figs/posterio_predictive.png")


