# %%


wvs_all = results_all['wvs']
params_all = results_all['params']
omega_as_all = params_all['omega_as']
tau_all = params_all['tau']
t_all = params_all['t']
g_all = params_all['g']
omega_f_all = params_all['omega_f']

wvs_qp = results_qp['wvs']
params_qp = results_qp['params']
omega_as_qp = params_qp['omega_as']
tau_qp = params_qp['tau']
t_qp = params_qp['t']
g_qp = params_qp['g']
omega_f_qp = params_qp['omega_f']

# %%

def test_angles(wvs, omega_f, g, ts, taus, omega_as=None):
    omega_as, ts, taus = prepare_params(omega_as, ts, taus)

    ts_shaped = ts.reshape((*ts.shape,1))
    taus_shaped = taus.reshape((*taus.shape,1))
    f_times = omega_f * (ts_shaped * 0.5 + taus_shaped)
    # omega_times = np.angle(wvs) - times
    a_times = 0.5 * ts_shaped * omega_as

    return np.angle(wvs)[0,0], f_times[0,0], a_times[0,0]

# %%

phis_all, f_times_all, a_times_all = test_angles(wvs_all, omega_f_all, g_all, t_all, tau_all, omega_as_all)
phis_qp, f_times_qp, a_times_qp = test_angles(wvs_qp, omega_f_qp, g_qp, t_qp, tau_qp, omega_as_qp)

# %%

fig, ax = plt.subplots(1,2, figsize=(12,8))

ax[0].set_title("all")
ax[0].scatter(x=omega_as_all, y=phis_all, label="phis", marker=".", linewidths=0, edgecolors=None)
ax[0].scatter(x=omega_as_all, y=a_times_all, label="a_times", marker=".", linewidths=0, edgecolors=None)
#ax[0].axhline(y=f_times_all, color='tab:red', linestyle='dashed', label='f_times')
ax[0].legend()

ax[1].set_title("qp")
ax[1].scatter(x=omega_as_qp, y=phis_qp, label="phis", marker=".", linewidths=0, edgecolors=None)
ax[1].scatter(x=omega_as_qp, y=a_times_qp, label="a_times", marker=".", linewidths=0, edgecolors=None)
#ax[1].axhline(y=f_times_qp, color='tab:red', linestyle='dashed', label='f_times')
ax[1].legend()

# %%

def bvs_test(v, omega_as, ts, taus):
    """ Return the interaction picture representation of a bloch vector, according to (42) of Ferraz et al. (2024) """
    times = np.einsum('...i,j', ts + taus, omega_as)
    sin_times = np.sin(times)
    cos_times = np.cos(times)

    v_int_x = v[0] * cos_times + v[1] * sin_times
    v_int_y = v[1] * cos_times - v[0] * sin_times
    v_int_z = np.broadcast_to(v[2], v_int_x.shape)

    return np.stack([v_int_x, v_int_y, v_int_z], axis=-1)


omega_as = np.linspace(0,2*np.pi,20)
i = np.array([0,0,1])
f = np.array([-1,0,0])
m = np.array([1,-1j,0])/2
t = np.array([0,1])
tau = np.array([0,0.5,1])
ts = np.array([t]) if np.isscalar(t) else t
taus = np.array([tau]) if np.isscalar(tau) else tau
ts_mesh, taus_mesh = np.meshgrid(ts, taus, sparse=True, indexing='ij')
f_ints = bvs_test(f, omega_as, ts_mesh, taus_mesh)
f_ints.shape, f_ints


# %%

def calculate_fgammas_test(f_ints, taus, gamma=0):
    """ Return fgamma_int, according to (43) of Ferraz et al. (2024) """
    e_gtaus = np.exp(-0.5 * gamma * taus)
    gamma_terms = np.stack([e_gtaus, e_gtaus, e_gtaus**2], axis=-1).reshape((1,-1,1,3))

    return f_ints * gamma_terms

gamma = 0.1
fgamma_ints = calculate_fgammas_test(f_ints, taus_mesh, gamma)
fgamma_ints.shape, fgamma_ints

# %%

np.all([[[fgamma_ints[i,j,k] == calculate_fgammas_test(f_ints[i,j,k], taus[j], gamma) for i in range(2)] for j in range(3)] for k in range(20)])


# %%

m_ints = bvs_test(m, omega_as, 0.5 * ts, 0).reshape((*ts.shape, 1, -1, 3))
m_ints.shape, m_ints

# %%

np.all([m_ints[i,0] == bvs_test(m, omega_as, 0.5 * ts[i].reshape(1,1), 0) for i in range(2)])

# %%

c = np.cross(m_ints, i)
c.shape, c

# %%

np.all([c[j,0] == np.cross(m_ints[j,0], i) for j in range(2)])

# %%

wvs = wvs_ferraz(i, f_ints, fgamma_ints, m_ints)
wvs.shape, wvs
# %%

ts_mesh.shape

# %%

tts = ts_mesh.reshape((*ts_mesh.shape,1))
tauss = taus_mesh.reshape(*taus_mesh.shape,1)
times = 1 * (tts * 0.5 + tauss)
omega_times = np.angle(wvs) - times

omega_times -= tts * omega_as / 2

np.all(np.isclose(omega_times, [[[np.angle(wvs[i,j,k]) - ts_mesh[i,0] * 0.5 - taus_mesh[0,j] - 0.5 * omega_as[k] * ts_mesh[i,0] for k in range(20)] for j in range(3)] for i in range(2)]))

# %%

prefactor = 2 * 1 * tts * np.sqrt(1 / 2)

Q = prefactor * np.sqrt(1/1) * np.abs(wvs) * np.sin(omega_times)
P = -1 * prefactor * np.sqrt(1) * np.abs(wvs) * np.cos(omega_times)

np.all(np.isclose(Q, [[[prefactor[i,0,0] * np.abs(wvs[i,j,k]) * np.sin(omega_times[i,j,k]) for k in range(20)] for j in range(3)] for i in range(2)]))

# %%



# Gaussian disorder, tau ~ sigma >> t
omega_a_samples = np.int64(1e7)

omega_a_mean = 1e8
omega_a_std = 1e5

params['omega_as'] = omega_a_mean # generator.normal(omega_a_mean, omega_a_std, omega_a_samples)
params['omega_a_0'] = omega_a_mean
params['t'] = 1e-9
params['tau'] = 1
params['gamma'] = 1e-1
params['omega_f'] = 1e9
params['g'] = 1e6

results = wv_pipeline(params, field_quads=False)

# %%

print("\tWith disorder\t|\tWithout disorder")
print("-------------------------------------")
print(f"     {np.round(results['wv_0_tau'][0,0,0],4)}  |    {np.round(results['wv_0_0'][0,0,0],4)}")




## Ohmic solve dump 

# %%

np.sqrt(
    np.power(ohm - 1j * omega_0, 2) + 1j * 4 * gamma * np.power(ohm, 2) / omega_0
)

# %%

c1_ts, c_vecs = solvable_jaynes_cumming(times, c1_0, c0, 1 + 0j, 1 + 0j)
c1_ts

# %%

np.power(np.abs(1/np.sqrt(2)), 2), np.power(np.abs((1 + 1j)/np.sqrt(4)), 2)

# %%


np.power(np.abs(c1_0), 2), np.power(np.abs(c0), 2)

# %%

omegas = np.linspace(0, 100, 1000)
j_w = gamma * omegas * np.power(ohm, 2) / (np.pi * omega_0 * (np.power(ohm, 2) + np.power(omegas, 2)))
plt.scatter(omegas, j_w)

# %%

import scipy.stats as ss 

# %%


class Ohmic(ss.rv_continuous):
    def _pdf(self, omega):
        return omega / (np.power(ohm, 2) + np.power(omega, 2))

# %%

ohmic_sd = Ohmic(a=-1e3, b=1e3)

# %%

times = np.arange(0, 1e2)
ohm = 1e-1
omega_0 = 1
gamma = 1

# %%


def exp_term(omega, t):
    return np.exp(-1j * omega * t)


def ohmic(omega):
    numerator = gamma * np.power(ohm, 2) * omega
    denominator = omega_0 * np.pi * (np.power(ohm, 2) + np.power(omega, 2))
    return numerator / denominator


def int_estimate(f, samples):
    values = np.array([f(x) for x in samples[:-1]]) * (samples[1] - samples[0]) #/ (samples[-1] - samples[0])
    return values.sum()


def ohmic_analytic(t):
    return - np.sign(t) * 1j * gamma * np.power(ohm, 2) * np.exp(- ohm * np.abs(t)) / omega_0

# %%

omegas = np.linspace(-1e2, 1e2, int(1e3))

# %%

omega_0 = 1
ohm = 1e-1

ts_2 = np.linspace(-25, 24, int(2e2))
estimates_2 = [int_estimate(lambda omega: ohmic(omega) * exp_term(omega, t), omegas) for t in ts_2]
anals_2 = ohmic_analytic(ts_2)


plt.scatter(ts_2, np.imag(estimates_2))
plt.scatter(ts_2, np.imag(anals_2))

# %%


ts_1 = np.linspace(-1e-1, 1e-1, int(2e2))
estimates_1 = [int_estimate(lambda omega: ohmic(omega) * exp_term(omega, t), omegas) for t in ts_1]
anals_1 = ohmic_analytic(ts_1)


plt.scatter(ts_1, np.imag(estimates_1))
plt.scatter(ts_1, np.imag(anals_1))


# %%

omega_0 = 1
ohm = 1

ts_3 = np.linspace(-25, 24, int(2e2))
estimates_3 = [int_estimate(lambda omega: ohmic(omega) * exp_term(omega, t), omegas) for t in ts_3]
anals_3 = ohmic_analytic(ts_3)


ts_3 = np.linspace(-25, 24, int(2e2))
plt.scatter(ts_3, np.imag(estimates_3))
plt.scatter(ts_3, np.imag(anals_3))


# %%

ts_4 = np.linspace(-1e-1, 1e-1, int(2e2))
estimates_4 = [int_estimate(lambda omega: ohmic(omega) * exp_term(omega, t), omegas) for t in ts_4]
anals_4 = ohmic_analytic(ts_4)

plt.scatter(ts_4, np.imag(estimates_4))
plt.scatter(ts_4, np.imag(anals_4))


# %%

ohmic_dist = Ohmic(a=0, b=int(1e4))
samples = ohmic_dist.rvs(size=int(1e4))


# %%

def prefactor():
    return (gamma * ohm^2) / (np.pi * omega_0)


ts_sampling = np.linspace(-5, 5, int(1e4))
values = samples * prefactor() * np.exp(-1j * samples * ts_sampling)
sample_estimate = values.sum()/1e4
sample_anals = ohmic_analytic(ts_sampling)

# %%


plt.scatter(ts_sampling, np.imag(sample_estimate))
plt.scatter(ts_sampling, np.imag(sample_anals))

# %%

def ohmic_d(ohm, omega_0):
    return np.array(
        [
            np.sqrt(np.power(ohm - 1j * omega_0, 2) + 4j),
            np.sqrt(np.power(ohm - 1j * omega_0, 2) + 4j * np.power(ohm, 2)),
        ]
    )


omega_0 = 1e2
ohms = np.linspace(20 * omega_0 - 50, 20 * omega_0 + 50, int(1e3))
ohm_ds = np.array([ohmic_d(ohm, omega_0) for ohm in ohms])
plt.figure(figsize=(12, 4))
# plt.scatter(ohms, np.real(ohm_ds[:, 0]), color="#2ea3a3", marker=".", linewidths=0, edgecolors=None, label=r'No \Omega^2')
plt.scatter(
    ohms,
    np.real(ohm_ds[:, 1]),
    color="#b347ab",
    marker=".",
    linewidths=0,
    edgecolors=None,
    label=r"\Omega^2",
)
plt.scatter(ohms, ohms, color="#2d59bc", marker=".", linewidths=0, edgecolors=None)
# plt.axvline(x=omega_0*2*np.pi, color="tab:blue", linestyle="dashed")
plt.legend()


# %%


def ohmic_ft(times, ohm):
    exp_terms = np.exp(-ohm * np.abs(times))
    return -1j * exp_terms * np.power(ohm, 2) * np.sign(times) / (2 * np.pi)


times = np.linspace(0, 10, int(1e3))
for ohm in np.linspace(0, 2 * np.pi, 5):
    plt.scatter(
        times,
        np.imag(ohmic_ft(times, ohm)),
        marker=".",
        linewidths=0,
        edgecolors=None,
        label=r"\Omega={}".format(ohm),
    )

plt.legend()
