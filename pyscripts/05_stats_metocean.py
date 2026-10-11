# %% [markdown]
# # 05 · Site 14 wind, wave-height and wave-period statistics
#
# **Goal:** plot analytical Hs–Tp scatter diagrams, a 50-year IFORM contour
# surface, and its intersections with fixed wind-speed planes.
# Site 14 (Norway 5) uses Li, Gao and Moan's OMAE2013-10156 model,
# equations (4), (11)–(20), Appendix A Table 5. These are verified **2013**
# coefficients; they have not been checked against the restricted 2015 article.
# Source and all five parameter sets: ../documents/five_site_models/models_2013.md
# https://doi.org/10.1115/OMAE2013-10156
#
# U is the **one-hour mean at 10 m height**, in m/s; Hs is in m and Tp in s.
# The model uses hourly hindcast data from 2001–2010. It describes environmental
# states, not individual wave heights, response maxima or weather persistence.
# No measured observations are plotted: these are analytical density heatmaps.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import integrate, stats

np.set_printoptions(precision=5, suppress=True)
print('Site 14: Norway 5. U is a one-hour mean wind speed at 10 m height.')
print('Source: OMAE2013-10156, Table 5; 2015 journal coefficients unverified.')

# %% [markdown]
# ## Site 14 parameters and joint density
#
# f(U,Hs,Tp) = f(U) f(Hs|U) f(Tp|U,Hs).
# Weibull parameters below are **shape then scale**, with zero location.
# The displayed conditional density is f(Hs,Tp|U=u); do not multiply it by
# f(U=u) unless a trivariate density is wanted. A density is not a bin probability.

# %%
alpha_u, beta_u = 2.029, 9.409
a1, a2, a3 = 2.136, 0.013, 1.709
b1, b2, b3 = 1.816, 0.024, 1.787
theta, gamma = -0.255, 1.0
e1, e2, e3 = 8.0, 1.938, 0.486
f1, f2, f3 = 2.5, 3.001, 0.745
k1, k2, k3 = -0.001, 0.316, -0.145


def hs_distribution(wind_speed):
    """Significant wave height conditional on U, with Hs in metres."""
    wind_speed = np.asarray(wind_speed, dtype=float)
    if np.any(~np.isfinite(wind_speed)) or np.any(wind_speed <= 0):
        raise ValueError('Wind speeds must be finite and positive in m/s.')
    shape = a1 + a2 * wind_speed**a3
    scale = b1 + b2 * wind_speed**b3
    return stats.weibull_min(c=shape, scale=scale)


def tp_parameters(wind_speed, hs):
    """Return log-space mean, standard deviation and validity mask for Tp."""
    hs = np.asarray(hs, dtype=float)
    if np.any(~np.isfinite(hs)) or np.any(hs < 0):
        raise ValueError('Hs must contain finite nonnegative heights in metres.')
    wind_speed = np.asarray(wind_speed, dtype=float)
    if np.any(~np.isfinite(wind_speed)) or np.any(wind_speed <= 0):
        raise ValueError('Wind speeds must be finite and positive in m/s.')
    reference_period = e1 + e2 * hs**e3
    reference_wind = f1 + f2 * hs**f3
    mean_period = reference_period * (
        1 + theta * ((wind_speed - reference_wind) / reference_wind)**gamma
    )
    cov_period = k1 + k2 * np.exp(k3 * hs)
    valid = (mean_period > 0) & (cov_period > 0)
    log_variance = np.log1p(cov_period**2)
    # Invalid empirical fits remain NaN; no clipping or renormalization.
    log_mean = np.log(np.where(valid, mean_period, np.nan)) - log_variance / 2
    log_std = np.where(valid, np.sqrt(log_variance), np.nan)
    return log_mean, log_std, valid


def conditional_density(wind_speed, hs, tp):
    """f(Hs,Tp|U=u), in 1/(m s); broadcast Hs and Tp arrays as needed."""
    tp = np.asarray(tp, dtype=float)
    if np.any(~np.isfinite(tp)) or np.any(tp <= 0):
        raise ValueError('Tp must contain finite positive periods in seconds.')
    log_mean, log_std, valid = tp_parameters(wind_speed, hs)
    period_pdf = stats.lognorm.pdf(tp, s=log_std, scale=np.exp(log_mean))
    density = hs_distribution(wind_speed).pdf(hs) * period_pdf
    return np.where(valid, density, np.nan)


print('U Weibull: shape =', alpha_u, '; scale =', beta_u, 'm/s')
example_u, example_hs = 12.0, 3.0
example_mu, example_sigma, _ = tp_parameters(example_u, example_hs)
print(f'At U={example_u:g} m/s, Hs={example_hs:g} m: '
      f'E[Tp]={np.exp(example_mu + example_sigma**2 / 2):.3f} s, '
      f'std(ln Tp)={example_sigma:.4f}')

# %% [markdown]
# ## Plot wind-speed probability density
#
# The requested panels cover U=4–25 m/s at 10 m height. They are conditional
# on exact U values, not averages over wind-speed bins. Transform hub-height
# winds to the reference height before using this model.

# %%
wind_distribution = stats.weibull_min(c=alpha_u, scale=beta_u)
wind_axis = np.linspace(0, 35, 500)
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
ax.plot(wind_axis, wind_distribution.pdf(wind_axis), color='tab:blue')
ax.axvspan(4, 25, color='tab:blue', alpha=0.12, label='heatmap wind-speed range')
ax.set(xlabel='One-hour mean wind speed U at 10 m [m/s]',
       ylabel='Wind-speed density [1/(m/s)]', title='Site 14 · marginal wind speed')
ax.legend()
plt.show()
print('Probability of 4 ≤ U ≤ 25 m/s:',
      wind_distribution.cdf(25) - wind_distribution.cdf(4))

# %% [markdown]
# ## Rosenblatt transformation and the 50-year IFORM radius
#
# Use the conditional order U -> Hs|U -> Tp|U,Hs:
#
# z1 = Φ⁻¹(F_U(u)), z2 = Φ⁻¹(F_Hs|U(h|u)),
# z3 = [ln(t) - μ_lnTp(u,h)] / σ_lnTp(h).
#
# For an everywhere-valid joint model these coordinates are independent
# standard normals. Inverse Rosenblatt transformation maps the sphere
# z1² + z2² + z3² = β² back into physical (U,Hs,Tp) space.
# Following Eq. (21) of the source, p = 1/N and β = Φ⁻¹(1-p), where N is
# the number of one-hour states in 50 years. This is the **IFORM one-sided
# normal-tail convention**, not a sphere chosen to enclose probability 1-p.
# Under independent standard normals, the probability outside the sphere is
# P(χ²_3 > β²), which differs from p. See Mackay and Haselsteiner (2021):
# https://doi.org/10.1016/j.marstruc.2020.102863
#
# At a fixed physical wind u, z1 is fixed, so the intersection is a circle
# in (z2,z3) with radius sqrt(β²-z1²). It is not a separately constructed
# conditional 50-year contour using radius β. No intersection exists if
# |z1| > β. If a sphere reaches the empirical fit's invalid domain, missing
# Tp values remain gaps. The default 50-year surface is checked below.

# %%
return_period_years = 50.0
sea_state_duration_hours = 1.0
hours_per_year = 365.25 * 24
number_of_states = return_period_years * hours_per_year / sea_state_duration_hours
exceedance_probability = 1 / number_of_states
iform_beta = stats.norm.isf(exceedance_probability)
print(f'{return_period_years:g}-year IFORM: N={number_of_states:g}, '
      f'p={exceedance_probability:.6e}, beta={iform_beta:.6f}')
print('Standard-normal sphere exterior probability (not p):',
      stats.chi2.sf(iform_beta**2, df=3))


def inverse_rosenblatt(z1, z2, z3):
    """Normal coordinates -> (U, Hs, Tp); invalid fitted Tp remains NaN."""
    z1, z2, z3 = np.broadcast_arrays(z1, z2, z3)
    # A Weibull inverse at Φ(z) is scale * [-ln(1-Φ(z))]**(1/shape).
    # logsf avoids cancellation of small upper-tail probabilities.
    wind = beta_u * (-stats.norm.logsf(z1))**(1 / alpha_u)
    shape = a1 + a2 * wind**a3
    scale = b1 + b2 * wind**b3
    height = scale * (-stats.norm.logsf(z2))**(1 / shape)
    mu, sigma, valid = tp_parameters(wind, height)
    period = np.where(valid, np.exp(mu + sigma * z3), np.nan)
    return wind, height, period


def rosenblatt(wind, height, period):
    """Physical states -> normal coordinates; use only valid model states."""
    wind, height, period = np.broadcast_arrays(wind, height, period)
    mu, sigma, valid = tp_parameters(wind, height)
    if np.any(~valid) or np.any(~np.isfinite(period)) or np.any(period <= 0):
        raise ValueError('The Rosenblatt transform requires valid positive Tp.')
    # Use log survival functions to preserve upper-tail accuracy.
    wind_log_sf = -(wind / beta_u)**alpha_u
    height_shape = a1 + a2 * wind**a3
    height_scale = b1 + b2 * wind**b3
    height_log_sf = -(height / height_scale)**height_shape
    normal_wind = stats.norm.isf(np.exp(wind_log_sf))
    normal_height = stats.norm.isf(np.exp(height_log_sf))
    normal_period = (np.log(period) - mu) / sigma
    return normal_wind, normal_height, normal_period


def contour_at_wind(wind_speed, points=721):
    """Intersect the 3D IFORM surface with U=wind_speed, preserving gaps."""
    normal_wind = stats.norm.isf(wind_distribution.sf(wind_speed))
    if abs(normal_wind) > iform_beta:
        return np.array([]), np.array([])
    circle_radius = np.sqrt(max(0.0, iform_beta**2 - normal_wind**2))
    angle = np.linspace(0, 2 * np.pi, points)
    _, height, period = inverse_rosenblatt(
        normal_wind, circle_radius * np.cos(angle), circle_radius * np.sin(angle)
    )
    return height, period


normal_polar = np.linspace(0, np.pi, 121)[:, None]
normal_azimuth = np.linspace(0, 2 * np.pi, 241)[None, :]
surface_z1 = np.broadcast_to(iform_beta * np.cos(normal_polar), (121, 241))
surface_z2 = iform_beta * np.sin(normal_polar) * np.cos(normal_azimuth)
surface_z3 = iform_beta * np.sin(normal_polar) * np.sin(normal_azimuth)
surface_u, surface_hs, surface_tp = inverse_rosenblatt(
    surface_z1, surface_z2, surface_z3
)
print('Wind range on the IFORM surface [m/s]:', surface_u.min(), surface_u.max())
print('Invalid surface vertices retained as gaps:', np.isnan(surface_tp).sum())

# %% [markdown]
# ## Hs–Tp density heatmaps at each wind speed
#
# The empirical full model can give nonpositive mean Tp for very small Hs
# at high U, or nonpositive Tp coefficient of variation at very large Hs.
# Such combinations are outside its valid domain and are masked in grey.
# Their Hs probability is reported below; masking does not repair the fit.
#
# Every panel uses the same axes and logarithmic colour scale. Values below
# the colour-scale floor are white; grey cells indicate an invalid model.
# A cell's approximate probability is density × ΔHs × ΔTp. The plotted box
# is finite and its probability is reported without rescaling it to one.

# %%
# Edit these settings to choose wind speeds, ranges or plotting resolution.
wind_speeds = np.arange(4.0, 26.0, 1.0)
wind_contours = {u: contour_at_wind(u) for u in wind_speeds}
# Expand the axes if needed so the requested contour lines are not clipped.
contour_height_max = max(np.nanmax(h) for h, t in wind_contours.values() if h.size)
contour_period_max = max(np.nanmax(t) for h, t in wind_contours.values() if t.size)
hs_plot_max = max(20.0, np.ceil(contour_height_max))
tp_plot_max = max(30.0, np.ceil(contour_period_max))
bin_width = 0.125
hs_edges = np.linspace(0.0, hs_plot_max, int(np.ceil(hs_plot_max / bin_width)) + 1)
tp_edges = np.linspace(0.0, tp_plot_max, int(np.ceil(tp_plot_max / bin_width)) + 1)
panels_per_figure = 6
hs_centres = (hs_edges[:-1] + hs_edges[1:]) / 2
tp_centres = (tp_edges[:-1] + tp_edges[1:]) / 2
hs_grid = hs_centres[:, None]
tp_grid = tp_centres[None, :]
cell_area = np.diff(hs_edges)[:, None] * np.diff(tp_edges)[None, :]

density_grids = np.array([
    conditional_density(u, hs_grid, tp_grid) for u in wind_speeds
])
# These approximate probabilities are useful for constructing scatter tables.
# Their rows correspond to Hs bins and columns to Tp bins; invalid bins are NaN.
bin_probability_grids = density_grids * cell_area
colour_floor = 1e-5
colour_norm = LogNorm(vmin=colour_floor, vmax=np.nanmax(density_grids))
colour_map = plt.get_cmap('viridis').copy()
colour_map.set_bad('lightgrey')
colour_map.set_under('white')

for start in range(0, len(wind_speeds), panels_per_figure):
    stop = min(start + panels_per_figure, len(wind_speeds))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True,
                             layout='constrained')
    active_axes = []
    for index, ax in zip(range(start, stop), axes.flat):
        density = density_grids[index]
        # LogNorm masks zero as undefined. Map exact zeros below the colour
        # floor for display only, so they are white rather than invalid grey.
        display_density = np.where(density == 0, colour_floor / 10, density)
        mesh = ax.pcolormesh(tp_edges, hs_edges, np.ma.masked_invalid(display_density),
                             cmap=colour_map, norm=colour_norm, shading='flat',
                             rasterized=True)
        contour_hs, contour_tp = wind_contours[wind_speeds[index]]
        ax.plot(contour_tp, contour_hs, color='crimson', linewidth=1.5,
                label=f'{return_period_years:g}-year surface intersection')
        ax.set(title=f'U = {wind_speeds[index]:g} m/s at 10 m',
               xlim=(tp_edges[0], tp_edges[-1]),
               ylim=(hs_edges[0], hs_edges[-1]))
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=8)
        active_axes.append(ax)
    for ax in list(axes.flat)[stop - start:]:
        ax.set_visible(False)
    fig.supxlabel('Spectral peak period Tp [s]')
    fig.supylabel('Significant wave height Hs [m]')
    fig.suptitle('Site 14 · conditional density and '
                 f'{return_period_years:g}-year IFORM intersections\n'
                 'OMAE2013 full model · grey: invalid fit · white: below colour floor')
    fig.colorbar(mesh, ax=active_axes, extend='min',
                 label='f(Hs, Tp | U) [1/(m s)] · logarithmic scale', shrink=0.85)
    plt.show()
    print('Displayed wind speeds [m/s]:', wind_speeds[start:stop])

# %% [markdown]
# ## Check captured probability and excluded model domain
#
# Integrate f(Hs|U) P(Tp within the plot | U,Hs) independently using adaptive
# quadrature, then compare with the midpoint heatmap sum. The fit's validity
# boundaries are derived from mean Tp > 0 and coefficient of variation > 0.
# Probability beyond the plot and probability in the invalid fitted domain
# are distinct. A large discrepancy calls for finer bins or a wider plot.

# %%
valid_hs_max = np.log(-k1 / k2) / k3  # cov(Tp) = 0 at this height
print('Probability checks (conditional on U; no renormalization):')
print(' U [m/s]   midpoint box   quadrature box   invalid Hs mass')
coverage_results = []
for u, bin_probabilities in zip(wind_speeds, bin_probability_grids):
    # For gamma=1, mean Tp > 0 requires reference_wind > -theta*u/(1-theta).
    required_reference_wind = -theta * u / (1 - theta)
    valid_hs_min = max(0.0, (required_reference_wind - f1) / f2)**(1 / f3)
    height_distribution = hs_distribution(u)
    invalid_mass = (height_distribution.cdf(valid_hs_min)
                    + height_distribution.sf(valid_hs_max))

    def height_integrand(h):
        mu, sigma, valid = tp_parameters(u, h)
        if not valid:
            return 0.0
        period_distribution = stats.lognorm(s=sigma, scale=np.exp(mu))
        period_probability = (period_distribution.cdf(tp_edges[-1])
                              - period_distribution.cdf(tp_edges[0]))
        return height_distribution.pdf(h) * period_probability

    lower = max(hs_edges[0], valid_hs_min)
    upper = min(hs_edges[-1], valid_hs_max)
    box_probability = integrate.quad(height_integrand, lower, upper,
                                     epsabs=1e-9, epsrel=1e-8)[0]
    midpoint_probability = np.nansum(bin_probabilities)
    assert 0 <= box_probability <= 1
    assert abs(midpoint_probability - box_probability) < 0.005
    coverage_results.append((u, midpoint_probability, box_probability, invalid_mass))
    print(f'{u:8.1f}   {midpoint_probability:12.6f}   '
          f'{box_probability:14.6f}   {invalid_mass:15.3e}')

coverage_results = np.array(coverage_results)
print('Largest absolute midpoint/quadrature difference:',
      np.max(np.abs(coverage_results[:, 1] - coverage_results[:, 2])))
print('Exercise: use coarser Hs/Tp bins, then compare density and bin probability.')

# %% [markdown]
# ## Three-dimensional environmental contour surface
#
# Surface axes are U at 10 m, Hs, and Tp. Coloured lines are intersections
# with selected constant-U planes. All 22 wind-speed intersections are also
# shown on the density panels above. NaN vertices preserve invalid-fit gaps;
# no artificial closure or extrapolated replacement period is introduced.

# %%
fig = plt.figure(figsize=(11, 8), layout='constrained')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(surface_u, surface_hs, surface_tp, color='steelblue',
                alpha=0.3, rstride=2, cstride=3, linewidth=0,
                antialiased=True)
selected_winds = [4.0, 8.0, 12.0, 16.0, 20.0, 25.0]
for u, colour in zip(selected_winds, plt.get_cmap('plasma')(np.linspace(0.1, 0.85, 6))):
    height, period = contour_at_wind(u)
    ax.plot(np.full_like(height, u), height, period, color=colour,
            linewidth=1.6, label=f'U = {u:g} m/s')
ax.set(xlabel='Wind speed U at 10 m [m/s]',
       ylabel='Significant wave height Hs [m]',
       zlabel='Spectral peak period Tp [s]',
       title=f'Site 14 · {return_period_years:g}-year IFORM contour surface\n'
             f'OMAE2013 full model · {sea_state_duration_hours:g}-hour states')
ax.set_box_aspect((1.2, 1.0, 1.0))
ax.view_init(elev=24, azim=-55)
ax.legend(loc='upper left', fontsize=9)
plt.show()
print('Plotted full 3D contour surface with six highlighted wind-speed slices.')

# %% [markdown]
# ## Verify the transform and sphere intersections
#
# Check round-trip coordinates on valid surface points and fixed-wind slices.
# This verifies the contour geometry and conditional ordering, not the empirical
# model's extreme-tail accuracy or a structure's 50-year response prediction.

# %%
valid_surface = np.isfinite(surface_tp)
recovered_z = rosenblatt(surface_u[valid_surface], surface_hs[valid_surface],
                         surface_tp[valid_surface])
radius_squared = sum(z**2 for z in recovered_z)
radius_error = np.max(np.abs(radius_squared - iform_beta**2))
assert radius_error < 1e-6
coordinate_error = max(np.max(np.abs(recovered - original[valid_surface]))
                       for recovered, original in zip(
                           recovered_z, (surface_z1, surface_z2, surface_z3)))
assert coordinate_error < 1e-6
print('Maximum surface round-trip coordinate error:', coordinate_error)
print('Maximum surface squared-radius error:', radius_error)
print('Fixed-U slice checks: U [m/s], valid vertices, max Hs [m], invalid vertices')
for u, (height, period) in wind_contours.items():
    valid = np.isfinite(period)
    if not valid.any():
        print(f'{u:5.1f}: no valid intersection')
        continue
    z = rosenblatt(np.full(valid.sum(), u), height[valid], period[valid])
    assert np.max(np.abs(sum(component**2 for component in z) - iform_beta**2)) < 1e-6
    print(f'{u:5.1f}   {valid.sum():4d}   {height[valid].max():8.3f}   '
          f'{(~valid).sum():4d}')
print('Exercise: change the return period and compare the fixed-wind contours.')
