# %% [markdown]
# # 08 · Long-term three-hour heave maxima
#
# **Question:** how does the three-hour maximum upward heave change when the
# sea state varies? Use the same illustrative semi-rig and JONSWAP model as
# lesson 07, with $M_T=\max x(t)$ rather than $\max|x(t)|$.
# A discretized Weibull–lognormal environment gives a mixture of conditional
# maximum laws. Compare this sum with Monte Carlo samples and four pooled fits.
# The grid is synthetic, not a measured joint Hs–Tp distribution.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats

np.set_printoptions(precision=5, suppress=True)
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 10})
seed, sample_count = 2026, 500
rng = np.random.default_rng(seed)
T, dt, maximum_frequency, gamma = 10800.0, 0.5, 0.5, 3.3
rho, gravity, effective_area, draft = 1025.0, 9.81, 400.0, 10.0
mass, damping_ratio = 4e7, 0.5
stiffness = rho * gravity * effective_area
natural_omega = np.sqrt(stiffness / mass)
damping = 2 * damping_ratio * np.sqrt(stiffness * mass)
nt = int(round(T / dt))
assert nt % 2 == 0 and np.isclose(nt * dt, T)
assert 0 < maximum_frequency < 1 / (2 * dt)
f = np.fft.rfftfreq(nt, dt)
omega, df = 2 * np.pi * f, 1 / T
print(f'Seed: {seed}; {sample_count:,} sampled three-hour maxima')
print(f'Heave natural period: {2 * np.pi / natural_omega:.2f} s')

# %% [markdown]
# ## 1 · The same wave → force → heave model
#
# Repeat lesson 07's functions so this lesson runs independently. Spectra are
# one-sided per Hz, normalized on the retained FFT grid. The excitation transfer
# is $H_{Fη}=ρgA\exp(-ω^2d/g)$ and $H_{xF}=(K-Mω^2+iCω)^{-1}$.
# These are illustrative linear hydrodynamics, with no transient from rest.

# %%
def jonswap_spectrum(frequency, significant_height, peak_period,
                     peak_enhancement=3.3, cutoff=0.5):
    """JONSWAP on an FFT grid, normalized to the prescribed Hm0 [m²/Hz]."""
    spectrum = np.zeros_like(frequency)
    active = (frequency > 0) & (frequency <= cutoff)
    positive = frequency[active]
    peak_frequency = 1 / peak_period
    width = np.where(positive <= peak_frequency, 0.07, 0.09)
    peak_shape = np.exp(-0.5 * ((positive - peak_frequency)
                               / (width * peak_frequency))**2)
    log_shape = (-5 * np.log(positive)
                 - 1.25 * (peak_frequency / positive)**4
                 + np.log(peak_enhancement) * peak_shape)
    spectrum[active] = np.exp(log_shape - log_shape.max())
    spectrum[-1] = 0.0
    spacing = frequency[1] - frequency[0]
    return spectrum * (significant_height / 4)**2 / (spectrum.sum() * spacing)


def wave_force_transfer(frequency, density, g, area, depth):
    wave_number = (2 * np.pi * frequency)**2 / g
    return density * g * area * np.exp(-wave_number * depth)


def heave_transfer(frequency, effective_mass, damping_coefficient, restoring):
    angular_frequency = 2 * np.pi * frequency
    return 1 / (restoring - effective_mass * angular_frequency**2
                + 1j * damping_coefficient * angular_frequency)


def response_moments(significant_height, peak_period):
    """Return m0, m2, m4 on the same spectral grid as lesson 07."""
    wave_psd = jonswap_spectrum(f, significant_height, peak_period,
                                gamma, maximum_frequency)
    assert np.isclose(wave_psd.sum() * df, (significant_height / 4)**2)
    response_psd = abs(Hheave * Hforce)**2 * wave_psd
    return np.array([(omega**order * response_psd).sum() * df
                     for order in (0, 2, 4)])


Hforce = wave_force_transfer(f, rho, gravity, effective_area, draft)
Hheave = heave_transfer(f, mass, damping, stiffness)

# %% [markdown]
# ## 2 · Weibull Hs and lognormal Tp on a 10 × 10 grid
#
# Assume independent marginals: Hs ~ Weibull(shape=2, scale=5 m), and
# ln(Tp / 1 s) ~ Normal(ln(14), 0.2²). Thus 14 s is the median Tp, not its mean.
# These editable parameters are illustrative, not fitted site statistics.
# The joint density is f(h,t)=f_Hs(h) f_Tp(t), in 1/(m s).
#
# Bound each marginal by its 0.1% and 99.9% quantiles and divide that interval
# into ten equal-width bins. Exact CDF differences give bin probabilities;
# normalize their products by the captured joint mass. Each bin midpoint is
# one representative sea state. This approximates the continuous environment
# conditional on the retained box; it is not an equal-probability grid.
# Time and block weights agree because every block lasts three hours.

# %%
hs_shape, hs_scale = 2.0, 5.0              # Weibull shape [-], scale [m]
tp_median, tp_log_std = 14.0, 0.2          # median [s], std of ln(Tp / 1 s) [-]
grid_size, tail_probability = 33, 0.001
hs_distribution = stats.weibull_min(c=hs_shape, scale=hs_scale)
tp_distribution = stats.lognorm(s=tp_log_std, scale=tp_median)
hs_bounds = hs_distribution.ppf([tail_probability, 1 - tail_probability])
tp_bounds = tp_distribution.ppf([tail_probability, 1 - tail_probability])
hs_edges = np.linspace(*hs_bounds, grid_size + 1)
tp_edges = np.linspace(*tp_bounds, grid_size + 1)
hs_values = (hs_edges[:-1] + hs_edges[1:]) / 2
tp_values = (tp_edges[:-1] + tp_edges[1:]) / 2
hs_bin_mass = np.diff(hs_distribution.cdf(hs_edges))
tp_bin_mass = np.diff(tp_distribution.cdf(tp_edges))
joint_bin_mass = np.outer(hs_bin_mass, tp_bin_mass)
captured_probability = joint_bin_mass.sum()
hs_grid, tp_grid = np.meshgrid(hs_values, tp_values, indexing='ij')
states = np.column_stack((hs_grid.ravel(), tp_grid.ravel()))
state_count = len(states)
weights = (joint_bin_mass / captured_probability).ravel()


def joint_environment_pdf(heights, periods):
    """Continuous joint PDF within the retained box, normalized to one."""
    heights, periods = np.broadcast_arrays(heights, periods)
    inside = ((heights >= hs_bounds[0]) & (heights <= hs_bounds[1])
              & (periods >= tp_bounds[0]) & (periods <= tp_bounds[1]))
    density = hs_distribution.pdf(heights) * tp_distribution.pdf(periods)
    return np.where(inside, density / captured_probability, 0.0)


print(f'Hs Weibull: shape={hs_shape:g}, scale={hs_scale:g} m')
print(f'Tp lognormal: median={tp_median:g} s, log-space std={tp_log_std:g}')
print('Hs / Tp bounds:', hs_bounds, tp_bounds)
print(f'Captured joint probability: {captured_probability:.6f}; '
      f'outside box: {1 - captured_probability:.6f}')
print(f'{state_count} weighted states; probability range: '
      f'{weights.min():.6g} to {weights.max():.6g}')

# Smooth continuous density, distinct from each discrete bin's probability.
plot_hs = np.linspace(*hs_bounds, 180)
plot_tp = np.linspace(*tp_bounds, 180)
density_hs, density_tp = np.meshgrid(plot_hs, plot_tp, indexing='ij')
joint_density = joint_environment_pdf(density_hs, density_tp)
fig_density = plt.figure(figsize=(10, 7), layout='constrained')
ax_density = fig_density.add_subplot(111, projection='3d')
surface = ax_density.plot_surface(density_tp, density_hs, joint_density,
                                  cmap='viridis', alpha=0.8, linewidth=0)
contour_levels = np.linspace(joint_density.max() / 10, joint_density.max() * 0.95, 9)
ax_density.contour3D(density_tp, density_hs, joint_density,
                     levels=contour_levels, colors='0.2', linewidths=0.7)
ax_density.contour(density_tp, density_hs, joint_density,
                   levels=contour_levels, zdir='z', offset=0, cmap='viridis')
ax_density.set(xlabel='Tp [s]', ylabel='Hs [m]',
               zlabel='Joint PDF [1/(m s)]', zlim=(0, 1.08 * joint_density.max()),
               title='Independent Weibull Hs × lognormal Tp\nDensity conditioned on retained box')
ax_density.view_init(elev=28, azim=-55)
fig_density.colorbar(surface, ax=ax_density, shrink=0.65, pad=0.1,
                     label='Joint PDF [1/(m s)]')
plt.show()

# For each representative state, σ=sqrt(m0), N=T sqrt(m2/m0)/(2π).
moments = np.array([response_moments(hs, tp) for hs, tp in states])
sigmas = np.sqrt(moments[:, 0])
nu0 = np.sqrt(moments[:, 1] / moments[:, 0]) / (2 * np.pi)
cycle_counts = T * nu0
assert states.shape == (grid_size**2, 2) and np.isclose(weights.sum(), 1)
assert np.all(weights > 0) and np.all(cycle_counts >= 1)
assert np.isclose(captured_probability, (1 - 2 * tail_probability)**2)
assert np.allclose(weights.reshape(grid_size, grid_size).sum(axis=1),
                   hs_bin_mass / hs_bin_mass.sum())
assert np.allclose(weights.reshape(grid_size, grid_size).sum(axis=0),
                   tp_bin_mass / tp_bin_mass.sum())
assert np.isclose(integrate.trapezoid(
    integrate.trapezoid(joint_density, plot_tp, axis=1), plot_hs), 1, atol=2e-4)
print('Response RMS range [m]:', sigmas.min(), sigmas.max())
print('Expected cycles in three hours:', cycle_counts.min(), cycle_counts.max())
fig_states, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
for ax, values, label in zip(axes, (sigmas, cycle_counts),
                            ('Heave RMS [m]', 'Expected cycles in 3 h')):
    points = ax.scatter(states[:, 1], states[:, 0], c=values, s=55)
    fig_states.colorbar(points, ax=ax, label=label)
    ax.set(xlabel='Tp [s]', ylabel='Hs [m]', title=label)
fig_states.suptitle('Weibull–lognormal environmental grid · bin midpoints')
plt.show()

# %% [markdown]
# ## 3 · Instantaneous heave is a Gaussian mixture
#
# Within a stationary state, instantaneous heave is centered Gaussian.
# $f_X(x)=\sum_jw_j φ(x/σ_j)/σ_j$; its variance is
# $\sum_jw_jσ_j^2$. This distribution is distinct from a three-hour maximum.

# %%
grid_x = np.linspace(-6 * sigmas.max(), 6 * sigmas.max(), 1000)
pdf_instantaneous = stats.norm.pdf(grid_x[:, None], scale=sigmas) @ weights
variance_long = weights @ sigmas**2
fig_instantaneous, ax = plt.subplots(layout='constrained')
ax.plot(grid_x, pdf_instantaneous, label='Instantaneous Gaussian mixture')
ax.plot(grid_x, stats.norm.pdf(grid_x, scale=np.sqrt(variance_long)), '--',
        label='Gaussian with the same variance')
ax.set(xlabel='Instantaneous heave [m]', ylabel='PDF [1/m]')
ax.legend()
plt.show()

# %% [markdown]
# ## 4 · Sum the conditional three-hour maximum laws
#
# Approximate independent cycle peaks by Rayleigh(σ_j), as in lesson 07:
# $F_j(m)=[1-\exp(-m^2/(2σ_j^2))]^{N_j}$ for $m>0$, zero otherwise.
# The long-term block CDF is $F_{LT}(m)=\sum_jw_jF_j(m)$.
# Sum PDFs and survival probabilities in the same way. The fractional effective
# counts and independence of successive peaks are modeling approximations.
# Use log CDFs, log1p and expm1 to retain small tail probabilities.

# %%
def conditional_maximum_law(values):
    """Return CDF, PDF, SF; the last axis indexes environmental states."""
    m = np.asarray(values, dtype=float)[..., None]
    positive = m > 0
    # Evaluate only safe positive arguments; set the actual support below.
    z = np.where(positive, m, sigmas) / sigmas
    a = z**2 / 2
    log_peak_cdf = np.empty_like(a)
    small = a <= np.log(2)
    log_peak_cdf[small] = np.log(-np.expm1(-a[small]))
    log_peak_cdf[~small] = np.log1p(-np.exp(-a[~small]))
    log_max_cdf = cycle_counts * log_peak_cdf
    cdf = np.where(positive, np.exp(log_max_cdf), 0.0)
    sf = np.where(positive, -np.expm1(log_max_cdf), 1.0)
    log_pdf = (np.log(cycle_counts) + np.log(z) - np.log(sigmas) - a
               + (cycle_counts - 1) * log_peak_cdf)
    pdf = np.where(positive, np.exp(log_pdf), 0.0)
    return cdf, pdf, sf


def block_cdf(values):
    return conditional_maximum_law(values)[0] @ weights


def block_pdf(values):
    return conditional_maximum_law(values)[1] @ weights


def block_sf(values):
    return conditional_maximum_law(values)[2] @ weights


def conditional_maximum_ppf(probabilities, state_indices):
    """Sample a conditional maximum via F_Rayleigh(m) = U**(1/N)."""
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError('Inverse sampling requires probabilities in (0, 1).')
    log_peak_cdf = np.log(probabilities) / cycle_counts[state_indices]
    return sigmas[state_indices] * np.sqrt(-2 * np.log(-np.expm1(log_peak_cdf)))


def maximum_from_sf(target_sf):
    """Invert the mixture in survival space, including small annual tails."""
    if not 0 < target_sf < 1:
        raise ValueError('Target survival probability must lie in (0, 1).')
    upper = 8 * sigmas.max()
    while block_sf(upper) > target_sf:
        upper *= 2
    return optimize.brentq(lambda value: block_sf(value) - target_sf, 0, upper)

# %% [markdown]
# ## 5 · Monte Carlo sampling and four pooled fits
#
# Select a state using its bin probability, then invert its maximum CDF. This is
# fast Monte Carlo of the same mixture, not simulation of heave time histories.
# ECDF agreement checks sampling and summation, not the Rayleigh approximation.
# The Hs–Tp scatter aggregates repeated draws at each midpoint: bubble area and
# colour indicate sample count. The histogram and ECDF then show the empirical
# three-hour maximum distribution before any parametric fits are introduced.
# Fit all four candidate distributions to the same pooled three-hour maxima.
# Ordinary Weibull means scipy.stats.weibull_min; the two-parameter fit fixes
# location to zero, while the three-parameter fit estimates it. Neither is
# scipy.stats.weibull_max. The lognormal fit also fixes location to zero:
# its shape is std(ln M) and its scale is the median maximum in metres.

# %%
state_draws = rng.choice(state_count, size=sample_count, p=weights)
uniform = rng.uniform(np.nextafter(0.0, 1.0), 1.0, size=sample_count)
maxima = conditional_maximum_ppf(uniform, state_draws)
sampled_states = states[state_draws]
state_sample_counts = np.bincount(state_draws, minlength=state_count)

# Repeated draws share exactly the same bin midpoint. Aggregate overlaps into
# bubbles whose areas and colours show how many samples selected each state;
# no jitter or extra environmental randomness is introduced for plotting.
visited = state_sample_counts > 0
fig_sampled_states, ax = plt.subplots(figsize=(9, 5), layout='constrained')
ax.scatter(states[:, 1], states[:, 0], s=16, marker='x', color='0.7',
           label='Environmental grid midpoints', zorder=1)
sample_points = ax.scatter(
    states[visited, 1], states[visited, 0],
    s=700 * state_sample_counts[visited] / state_sample_counts.max(),
    c=state_sample_counts[visited], cmap='viridis', alpha=0.85,
    edgecolors='black', linewidths=0.4, zorder=2,
    label='Sampled cases (bubble area ∝ count)')
fig_sampled_states.colorbar(sample_points, ax=ax, label='Number of Monte Carlo samples')
ax.set(xlabel='Sampled Tp [s]', ylabel='Sampled Hs [m]',
       title=f'Where the {sample_count:,} Monte Carlo cases were sampled')
ax.legend(fontsize=8)
plt.show()

# Show the sampled output distribution on its own before fitting any models.
ordered = np.sort(maxima)
ecdf = np.arange(1, sample_count + 1) / sample_count
fig_empirical, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
axes[0].hist(maxima, bins=85, density=True, color='tab:blue', alpha=0.65)
axes[0].set(ylabel='Empirical density [1/m]', title='Histogram of sampled maxima')
axes[1].step(np.r_[0, ordered], np.r_[0, ecdf], where='post', color='tab:blue')
axes[1].set(ylabel='Empirical cumulative probability', title='Empirical CDF',
            ylim=(0, 1.02))
for ax in axes:
    ax.set(xlabel='Three-hour maximum upward heave [m]', xlim=(0, ordered[-1] * 1.05))
fig_empirical.suptitle(f'Empirical three-hour maximum distribution · {sample_count:,} samples')
plt.show()

# Fit the same pooled maxima shown above.
fit_specs = (
    ('Gumbel', stats.gumbel_r, {}),
    ('Weibull 2p', stats.weibull_min, {'floc': 0}),
    ('Weibull 3p', stats.weibull_min, {}),
    ('Lognormal', stats.lognorm, {'floc': 0}),
)
fitted = {}
for name, distribution, options in fit_specs:
    parameters = distribution.fit(maxima, **options)
    rv = distribution(*parameters)
    assert np.all(np.isfinite(parameters)) and parameters[-1] > 0
    assert np.all(np.isfinite(rv.logpdf(maxima)))
    if name.startswith('Weibull') or name == 'Lognormal':
        assert parameters[0] > 0
    fitted[name] = rv
    parameter_names = ('loc', 'scale') if name == 'Gumbel' else ('shape', 'loc', 'scale')
    print(name + ': ' + ', '.join(f'{key}={value:.6g}'
                                 for key, value in zip(parameter_names, parameters)))
    if name == 'Lognormal':
        print(f'  std(ln M)={parameters[0]:.6g}; median maximum={parameters[-1]:.6g} m')

# Midpoint plotting positions keep every displayed empirical SF strictly positive.
empirical_sf = (sample_count - np.arange(sample_count) - 0.5) / sample_count
probabilities = np.array([0.5, 0.9, 0.99])
mixture_quantiles = np.array([maximum_from_sf(1 - p) for p in probabilities])
print('Three-hour maximum quantiles [m]')
print(f'{"Model":<20} {"q50":>10} {"q90":>10} {"q99":>10}')
quantiles = {'Empirical': np.quantile(maxima, probabilities),
             'Rayleigh mixture': mixture_quantiles}
quantiles.update({name: rv.ppf(probabilities) for name, rv in fitted.items()})
for name, values in quantiles.items():
    assert np.all(np.isfinite(values))
    print(f'{name:<20}' + ''.join(f'{value:11.4f}' for value in values))

# %% [markdown]
# ## 6 · Compare density, cumulative probability, and upper tails
#
# A single fitted family need not reproduce a mixture. The survival plot makes
# differences in the upper tail visible even when CDF curves look similar.
# Empirical tail information is limited by sample size; fitted curves beyond
# the observations are extrapolations, not additional Monte Carlo evidence.

# %%
grid_m = np.linspace(0, max(ordered[-1] * 1.1, maximum_from_sf(1e-6)), 1600)
fig_comparison, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
axes[0].hist(maxima, bins=85, density=True, alpha=0.3, color='0.4', label='Monte Carlo')
axes[1].step(np.r_[0, ordered], np.r_[0, ecdf], where='post',
             color='0.5', label='Monte Carlo ECDF')
axes[2].semilogy(ordered, empirical_sf, color='0.5', label='Monte Carlo (midpoints)')
for ax, values in zip(axes, (block_pdf(grid_m), block_cdf(grid_m), block_sf(grid_m))):
    ax.plot(grid_m, values, color='black', lw=2.2, label='Rayleigh mixture')
for index, (name, rv) in enumerate(fitted.items()):
    for ax, method in zip(axes, (rv.pdf, rv.cdf, rv.sf)):
        values = method(grid_m)
        if ax is axes[2]:
            values = np.where(values > 0, values, np.nan)
        ax.plot(grid_m, values, color=f'C{index}', ls=('--', '-.', ':', '-')[index], label=name)
for ax, ylabel, title in zip(axes, ('PDF [1/m]', 'CDF', 'Survival probability'),
                             ('Density', 'Cumulative distribution', 'Upper tail')):
    ax.set(xlabel='Three-hour maximum upward heave [m]', ylabel=ylabel, title=title,
           xlim=(0, grid_m[-1]))
    ax.legend(fontsize=8)
axes[0].set_ylim(bottom=0)
axes[1].set_ylim(0, 1.02)
axes[2].set_ylim(1e-6, 1)
fig_comparison.suptitle(f'{state_count}-state Weibull–lognormal mixture · {sample_count:,} Monte Carlo maxima')
plt.show()

# %% [markdown]
# ## 7 · Extension: annual extremes from the analytical mixture
#
# Assume 365 days and B=2920 independent three-hour blocks, with independently
# selected states. Then $F_A(m)=F_{LT}(m)^B$. Solve return levels through survival
# probabilities: a Y-year level satisfies $P(A>z_Y)=1/Y$.
# This extension uses the analytical Rayleigh mixture, not the four pooled fits.
# Weather persistence, nonstationarity, and uncertainty in the peak model are
# outside this calculation. Environmental truncation and grid resolution also
# affect the tails. These extrapolated levels are not validated designs.

# %%
seconds_per_year = 365 * 24 * 3600
B = seconds_per_year / T


def annual_cdf(values):
    with np.errstate(divide='ignore'):
        return np.exp(B * np.log1p(-block_sf(values)))


def return_level(years):
    if years <= 1:
        raise ValueError('Return period must exceed one year.')
    target_sf = -np.expm1(np.log1p(-1 / years) / B)
    return maximum_from_sf(target_sf)


periods = np.array([2., 5., 10., 20., 50., 100.])
levels = np.array([return_level(years) for years in periods])
print(f'Independent three-hour blocks/year: {int(B)}')
for years, level in zip(periods, levels):
    print(f'{years:g}-year return level: {level:.4f} m')
# Inverse sampling checks annual aggregation, not the physical tail model.
annual_uniform = rng.uniform(np.nextafter(0.0, 1.0), 1.0, size=600)
annual_samples = np.array([maximum_from_sf(-np.expm1(np.log(u) / B))
                           for u in annual_uniform])
annual_ordered = np.sort(annual_samples)
grid_a = np.linspace(min(annual_ordered[0], return_level(2)),
                     max(annual_ordered[-1], levels[-1]) * 1.05, 600)
fig_annual, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
axes[0].step(annual_ordered, np.arange(1, len(annual_ordered) + 1) / len(annual_ordered),
             where='post', label='Annual inverse samples')
axes[0].plot(grid_a, annual_cdf(grid_a), '--', label='Annual formula')
axes[0].set(xlabel='Annual maximum upward heave [m]', ylabel='CDF')
axes[0].legend()
axes[1].semilogx(periods, levels, 'o-')
axes[1].set(xlabel='Return period [years]', ylabel='Return level [m]')
plt.show()
print('Annual median sample / formula [m]:', np.median(annual_samples), return_level(2))

# %% [markdown]
# ## 8 · Numerical checks and interpretation
#
# Check normalization, inverse transforms, and Monte Carlo agreement. The ECDF
# tolerance is deliberately conservative. A large sample can verify the mixture
# formula without verifying the underlying short-term peak assumptions.

# %%
# The expanded environment spans very different response scales. A logarithmic
# grid resolves narrow low-sea-state densities as well as the largest maxima.
check_grid = np.r_[0, np.geomspace(0.001 * sigmas.min(), 12 * sigmas.max(), 20001)]
conditional_cdf, conditional_pdf, conditional_sf = conditional_maximum_law(check_grid)
assert np.all(np.diff(conditional_cdf, axis=0) >= -1e-12)
assert np.allclose(conditional_cdf + conditional_sf, 1)
assert np.all(conditional_cdf[0] == 0) and np.allclose(conditional_cdf[-1], 1)
assert np.allclose(integrate.trapezoid(conditional_pdf, check_grid, axis=0), 1, atol=2e-6)
assert np.isclose(integrate.trapezoid(block_pdf(check_grid), check_grid), 1, atol=2e-6)
assert np.allclose(integrate.cumulative_trapezoid(block_pdf(check_grid), check_grid, initial=0),
                   block_cdf(check_grid), atol=2e-5)
for index in range(state_count):
    recovered = conditional_maximum_law(conditional_maximum_ppf(probabilities, index))[0][:, index]
    assert np.allclose(recovered, probabilities, atol=1e-12)
assert np.allclose(block_cdf(mixture_quantiles), probabilities)
# Work in chunks to avoid a sample_count × state_count × several-arrays allocation.
model_at_samples = np.concatenate([block_cdf(chunk) for chunk in np.array_split(ordered, 20)])
ecdf_error = max(np.max(ecdf - model_at_samples),
                 np.max(model_at_samples - (ecdf - 1 / sample_count)))
assert ecdf_error < 3 / np.sqrt(sample_count)
assert np.all(np.diff(levels) > 0)
assert np.isclose(annual_cdf(return_level(50)), 0.98)
assert np.allclose(annual_cdf(annual_samples), annual_uniform, atol=1e-10)
print(f'Maximum ECDF error: {ecdf_error:.5f}; tolerance: {3 / np.sqrt(sample_count):.5f}')
print('Spectral, probability normalization, inverse, and annual checks passed.')
print('Exercise: change marginal parameters, grid_size, or tail_probability, then rerun.')
print('Conditional sampling verifies the mixture, not Rayleigh peak independence.')
