import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as pt
from scipy.stats import expon, logistic

# Simulate some data
np.random.seed(42)
n_locations = 10
n_observations = 100

# Drivers of fire probability
temp = np.random.normal(20, 5, n_observations)
humidity = np.random.normal(60, 10, n_observations)
location_idx = np.random.randint(0, n_locations, n_observations)

# True parameters
true_beta_temp = 0.05
true_beta_humidity = -0.03
true_location_effect = np.random.normal(0, 0.2, n_locations)
true_base_fire_prob = pm.math.sigmoid(true_beta_temp * temp + true_beta_humidity * humidity + true_location_effect[location_idx])

# Generate fire counts with exponential tail above base
true_fire_counts = np.zeros(n_observations, dtype=int)
for i in range(n_observations):
    if np.random.rand() < true_base_fire_prob[i]:
        true_fire_counts[i] = 0  # Likely no fire
    else:
        true_fire_counts[i] = int(np.random.exponential(1) + 1) # Exponential tail

# Custom distribution for fire counts
class ExponentialTail(pm.Continuous):
    def __init__(self, base_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_prob = base_prob

    def logp(self, value):
        logp = pt.switch(
            value < self.base_prob,
            -np.inf,
            pt.log(self.base_prob) - pt.log(1 - self.base_prob) - pt.log(expon.sf(value - self.base_prob)),
        )
        return logp

# Custom logistic max entropy distribution
class LogisticMaxEnt(pm.Continuous):
    def __init__(self, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def logp(self, value):
        logp = pt.log(logistic.pdf(value, loc=self.mu, scale=1))
        return logp

# PyMC5 model
with pm.Model() as model:
    # Priors
    beta_temp = pm.Normal("beta_temp", mu=0, sigma=1)
    beta_humidity = pm.Normal("beta_humidity", mu=0, sigma=1)
    location_effect = pm.Normal("location_effect", mu=0, sigma=0.5, dims="location")

    # Linear model and sigmoid transformation
    base_fire_prob = pm.Deterministic(
        "base_fire_prob", pm.math.sigmoid(beta_temp * temp + beta_humidity * humidity + location_effect[location_idx])
    )

    # Custom distribution for fire counts
    fire_counts = ExponentialTail(
        "fire_counts", base_prob=base_fire_prob, observed=true_fire_counts
    )

    #Custom logistic distribution for testing.
    logistic_test = LogisticMaxEnt("logistic_test", mu = base_fire_prob, observed = true_fire_counts)

    # Inference
    idata = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.95)

# Summarize results
az.summary(idata, var_names=["beta_temp", "beta_humidity", "location_effect"])
