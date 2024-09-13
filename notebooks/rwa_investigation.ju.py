'''
# %%

%load_ext autoreload
%autoreload 2

# %% [md]
'''
# %%


from mres import wv_pipeline, plot_fixed_results
import numpy as np
import matplotlib.pyplot as plt

# %%

generator = np.random.default_rng()

# %%

params = {
    'omega_as': None,
    'omega_a_0': None,
    'omega_f': 1,
    'g': 1,
    'i': np.array([0,0,1], dtype=np.complex128),
    'f': np.array([1,0,0], dtype=np.complex128),
    'm': np.array([1,0,0], dtype=np.complex128),
    't': None,
    'tau': 0,
    'gamma': 0,
    'm_to_int': False,
    'a': 0,
    'b': 1,
    'rwa': True,
}


# %%

# Across all omega_as
omega_a_samples = np.int64(5e3)

params['omega_as'] = np.linspace(0, 4 * np.pi, omega_a_samples)
params['omega_a_0'] = 2 * np.pi
params['t'] = 1
params['omega_f'] = 1
params['tau'] = 0
params['gamma'] = 0
params['g'] = 1

results = wv_pipeline(params)
plot_fixed_results(results)

# %%

# Gaussian disorder, tau=0

samples = 10000
omega_a_mean = 1e4
omega_a_std = 1e3

params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-3
params['tau'] = 0
params['gamma'] = 0
params['omega_f'] = 1e4
params['g'] = 1

results = wv_pipeline(params)
plot_fixed_results(results)

# %%

# Gaussian disorder, tau ~ sigma >> t

samples = 10000
omega_a_mean = 1e8
omega_a_std = 1e5
params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6

results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results)

# %%

# Gaussian disorder, tau << 1/std
samples = 10000
omega_a_mean = 1e8
omega_a_std = 1e2

params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6

results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results)

# %%

# Gaussian disorder, tau > 1/std
samples = 10000
omega_a_mean = 1e8
omega_a_std = 2e5

params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6

results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results, 0)


# %%

# Gaussian disorder, 1/std > tau >> t

samples = 10000
omega_a_mean = 9.993e7
omega_a_std = 3e4
params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6

results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results, 0)

# %%

# Constant disorder, tau ~ 1/std >> t
samples = 10000
omega_a_0 = 1e8
omega_a_range = 1e5

params['omega_as'] = generator.uniform(omega_a_0-omega_a_range, omega_a_0+omega_a_range, samples)
params['omega_a_0'] = omega_a_0
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6


results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results)

# %%

# Constant disorder, tau ~ 1/std >> t
samples = 10000
omega_a_0 = 9.993e7
omega_a_range = 1e5

params['omega_as'] = generator.uniform(omega_a_0-omega_a_range, omega_a_0+omega_a_range, samples)
params['omega_a_0'] = omega_a_0
params['t'] = 1e-9
params['tau'] = 1e-5
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6


results = wv_pipeline(params, field_quads=False)
plot_fixed_results(results)

# %%

delta = 1e-4
params = {
    'omega_as': None,
    'omega_a_0': None,
    'omega_f': 1,
    'g': 1,
    'i': np.array([0,1,0], dtype=np.complex128),
    'f': np.array([0,-np.cos(delta),np.sin(delta)], dtype=np.complex128),
    'm': np.array([1,0,0], dtype=np.complex128),
    't': None,
    'tau': 0,
    'gamma': 0,
    'm_to_int': False,
    'a': 0,
    'b': 1,
    'rwa': True,
}


# %%

# Across all omega_as
omega_a_samples = np.int64(5e3)

params['omega_as'] = np.linspace(0, 4 * np.pi, omega_a_samples)
params['omega_a_0'] = 2 * np.pi
params['t'] = 1
params['omega_f'] = 1
params['tau'] = 0
params['gamma'] = 0
params['g'] = 1

results = wv_pipeline(params)
plot_fixed_results(results)

# %%

# Zoom in
omega_a_samples = np.int64(5e3)

params['omega_as'] = np.linspace(0, 1e-2 * np.pi, omega_a_samples)
params['omega_a_0'] = 2 * np.pi
params['t'] = 1
params['omega_f'] = 1
params['tau'] = 0
params['gamma'] = 0
params['g'] = 1

results = wv_pipeline(params)
plot_fixed_results(results)
# %%

# Gaussian disorder, tau=0

samples = 10000
omega_a_mean = 1e2
omega_a_std = 1

params['omega_as'] = generator.normal(omega_a_mean, omega_a_std, samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-5
params['tau'] = 0
params['gamma'] = 0
params['omega_f'] = 1e4
params['g'] = 1

results = wv_pipeline(params)
plot_fixed_results(results)

# %%

def gaus_prob(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-np.pow((x - mu),2)/ (2 * sigma ** 2))

def f(x):
    return 1/ (1 + x)

mu = 1
sigma = 1
samples = np.int64(1e4)
span = 5

sum(gaus_prob(x, mu, sigma**2) * f(x) for x in np.linspace(mu-span, mu+span, samples)) * 2 * span / samples

# %%

xs = generator.normal(mu, sigma, samples)
np.mean(f(xs))

