## Zero-inflated logistic function 
This distribution is described in Kelley et al. (2021): https://doi.org/10.5194/bg-18-787-2021.

It describes a normal distribution under logistic transformation, with an initial "fire/no fire" test to see how likely it is that there is no burning.

To use it, make this line is in your namelist:
`link_func_class:: zero_inflated_logit`

This link distribution then required the following priors, noting that you can use different `dist` from (pymc)[https://www.pymc.io/projects/docs/en/stable/api/distributions.html] if you choose, though the distributions are recommended 

* `priors:: {'pname': "link-sigma",'np': 1, 'dist': 'HalfNormal', 'sigma': 0.5}`
  describes the standard deviation of the normal distribution component of the function

* `priors:: {'pname': "link-p0",'np': 1, 'dist': 'Uniform', 'lower': 0.0, 'upper': 1.0}`
  `priors:: {'pname': "link-p1",'np': 1, 'dist': 'LogNormal', 'mu': -1.0, 'sigma': 1.0}`
  Two parameters that describe the likelihood of a zero burnt area
