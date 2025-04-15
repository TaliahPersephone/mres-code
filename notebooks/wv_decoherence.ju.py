"""
# %%

%load_ext autoreload
%autoreload 2

# %% [md]
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from mres.pipelines import DecoherencePipeline
from mres.plot import deco_plot

# %%


delta = 1e-2
cos_d = np.cos(delta)
sin_d = np.sin(delta)
sq2 = np.sqrt(2)

# %% [md]

## Drude-Lorentz

# %%


params = {
    "v": np.array([0, -cos_d, sin_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 1, int(1e3)),
    "spectral": "ohmic",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1j * 1/sq2,
        "ohm": 10,
        "omega_0": 1e-2,
        "gamma": 1,
    },
}

deco_pipeline_ohmic = DecoherencePipeline(params)
results_ohmic = deco_pipeline_ohmic.run()
deco_plot(results_ohmic)

# %% [md]

## Drude-Lorentz shifted

# %%


params = {
    "v": np.array([0, -cos_d, sin_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 1, int(1e3)),
    "spectral": "ohmic_shifted",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1j * 1/sq2,
        "ohm": 1e2,
        "omega_0": 1,
        "gamma": 1e-1,
    },
}

deco_pipeline_ohmic_shifted = DecoherencePipeline(params)
results_ohmic_shifted = deco_pipeline_ohmic_shifted.run()
deco_plot(results_ohmic_shifted)

# %% [md]

## Jaynes-Cummings

# %%

lam = 0.1

# %% [md]

### $$\bar{u} = (0,0,1) \quad \bar{v} = (0, \sin\delta, -\cos\delta)$$

# %%


params_jc = {
    "v": np.array([0, sin_d, -cos_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 30, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1,
        "c0": 0,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %%


params_jc = {
    "v": np.array([0, sin_d, -cos_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(5, 200, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1,
        "c0": 0,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %% [md]

### $$\bar{u} = (1,0,0) \quad \bar{v} = (-\cos\delta, 0, \sin\delta)$$

# %%


params_jc = {
    "v": np.array([-cos_d, 0, sin_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 50, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %%


params_jc = {
    "v": np.array([-cos_d, 0, sin_d]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(5, 100, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %% [md]

### $$\bar{u} = (1,0,0) \quad \bar{v} = (-\cos\delta, \sin\delta, 0)$$

# %%


params_jc = {
    "v": np.array([-cos_d, sin_d, 0]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 50, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %%


params_jc = {
    "v": np.array([-cos_d, -sin_d, 0]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(1, 1e2, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %% [md]

### $$\delta = 1^{-4}$$

# %%

params_jc = {
    "v": np.array([-np.cos(1e-4), 0, -np.sin(1e-4)]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 1e-2, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %%


params_jc = {
    "v": np.array([-np.cos(1e-4), 0, np.sin(1e-4)]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0.5, 1e2, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)


# %% [md]

### $$\bar{u} = (0,1,0) \quad \bar{v} = (0, -\cos\delta, \sin\delta)$$

# %%


params_jc = {
    "v": np.array([0, -np.cos(1e-4), np.sin(1e-4)]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(1e-1, 1e2, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1j * 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)

# %% [md]

### $$\bar{u} = (0,1,0) \quad \bar{v} = (-\cos\delta, 0, \sin\delta)$$

# %%


params_jc = {
    "v": np.array([np.cos(1e-4), 0, np.sin(1e-4)]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 1e2, int(1e4)),
    "spectral": "jc",
    "spectral_params": {
        "c1_0": 1/sq2,
        "c0": 1j * 1/sq2,
        "lam": lam,
        "gamma_0": 1,
    },
}

deco_pipeline_jc = DecoherencePipeline(params_jc)

results_jc = deco_pipeline_jc.run()


deco_plot(results_jc)
