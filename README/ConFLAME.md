# ConFLAME
Model description can be found here: [ConFLAME_info.pdf](https://github.com/douglask3/Bayesian_fire_models/blob/main/README/ConFLAME_info.pdf). Basically, though, the model assigns drivers to "controls" and these controls are multiplied to give burnt area.

To run the model, you should set the following in the namelist:
`model_class:: ConFire`
`link_func_class:: MaxEnt`
There are also some compulsory `priors` that you need to add to the namelists. `pname` needs to match these, but other parameters are examples:

* `priors:: {'pname': "controlID", 'value': [[1, 2], [0, 2, 3, 4, 5, 7, 8], [9, 10, 11, 12], [10, 11, 12]]}`

  This assigns the different driver inputs to different controls. `value` takes a list of lists. The number of out lists matches the number of controls ConFire will use. The numbers within each embedded list match the position of the variables in the `x_filen_list` namelist item ([see main readme](https://github.com/douglask3/Bayesian_fire_models/blob/main/README) that you want to include in each control.

* `priors:: {'pname': "control_Direction", 'value': [1, -1, 1, -1]}`

  This says if each of the controls has a positive or negative effect on fire. i.e in the control increases and fire increases, set to 1. else set to -1.

* `priors:: {'pname': "driver_Direction", 'value': [[1, 1], [-1, 1, 1, -1, -1, -1, -1], [1, 1, 1, 1], [1, 1, 1]]}`

  This is similar but for whether the control increases or decreases with the driver.

* `priors:: {'pname': "x0",'np': 4, 'dist': 'Normal', 'mu': 0.0, 'sigma': 1.0}`

  This parameter describes the prior assumption on the "shifting" of the control compared (see pdf). `'np'` must match the number of controls. You can use any '`dist`' from [pymc](https://www.pymc.io/projects/docs/en/stable/api/distributions.html).

* `priors:: {'pname': "betas",'np': 2, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 1.0}`
    Describes the contribution of each driver to each control. You need an entry on a new line for each control, and '`np`' should be set to the number of drivers in that control. You can use any '`dist`' from [pymc](https://www.pymc.io/projects/docs/en/stable/api/distributions.html).

An optional parameter:
* `priors:: {'pname': "powers",'np': 2, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 1.0}`
    Describes the paramater for X^param. You need an entry on a new line for each control, and '`np`' should be set to the number of drivers in that control. You can use any '`dist`' from [pymc](https://www.pymc.io/projects/docs/en/stable/api/distributions.html).
