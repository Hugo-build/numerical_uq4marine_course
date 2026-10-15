# %% [markdown]
# # 07 · Three-hour heave extremes of a semi-rig
#
# **Question:** given a stationary JONSWAP sea state, a wave-force transfer function,
# and a linear heave model, what is the maximum upward displacement in three hours?
# We study $M_T=\max_{0\leq t<T}x(t)$, not $\max|x(t)|$.
#
# The rectangular semi-rig and its load model are illustrative, not calibrated
# hydrodynamics. Gaussian Fourier coefficients give stationary Gaussian waves,
# forces, and heave. Each independent record is periodic over its three-hour FFT
# interval; this is steady-state synthesis, not integration from rest.

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from scipy import integrate, stats

np.set_printoptions(precision=5, suppress=True)
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 10})

seed = 2026                                     # A random seed value
rng = np.random.default_rng(seed)               # random number generator for reproducibility
hs, tp, gamma = 5.0, 14.0, 3.3                  # define Hs, Tp, and γ for the JONSWAP sea state (default γ=3.3)
T, dt, maximum_frequency = 10800.0, 0.5, 0.5
R, examples_to_keep = 300, 5
rho, gravity, effective_area, draft = 1025.0, 9.81, 400.0, 10.0
mass, damping_ratio = 4e7, 0.08                          # effective mass includes added mass [kg] & damping ratio [-]
stiffness = rho * gravity * effective_area               # hydrostatic stiffness [N/m] for the semi-rig

natural_omega = np.sqrt(stiffness / mass)                # natural frequency [rad/s] of the heave system
damping = 2 * damping_ratio * np.sqrt(stiffness * mass)  # damping coefficient [N·s/m] for the heave system

nt = int(round(T / dt))                        # number of time samples
assert nt % 2 == 0 and np.isclose(nt * dt, T)  # insert a DC and Nyquist frequency for the FFT
assert 0 < maximum_frequency < 1 / (2 * dt) 

f = np.fft.rfftfreq(nt, dt)                    # FFT frequencies for the one-sided spectrum
omega, df = 2 * np.pi * f, 1 / T               # angular frequencies and frequency spacing

print(f'Seed: {seed}; {R} independent {T / 3600:g} h records; dt={dt:g} s')                         
print(f'Heave natural period: {2 * np.pi / natural_omega:.2f} s; damping: {damping_ratio:g}')


# %% [markdown]
# ## 1 · Start with the wave spectrum
#
# All spectra are one-sided and per Hz. Normalize the truncated JONSWAP shape
# so that $\sum S_{\eta \eta} (f)\Delta f=(H_s/4)^2$. DC and Nyquist are zero.
# This repeats the spectrum definition from lesson 06 so this lesson runs alone.

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


Seta = jonswap_spectrum(f, hs, tp, gamma, maximum_frequency)                   # Initialize the JONSWAP spectrum with the given parameters
assert np.isclose(Seta.sum() * df, (hs / 4)**2)                                # check that the spectrum is normalized correctly   

fig_spectrum, ax = plt.subplots(figsize=(9, 4), layout='constrained')          
ax.plot(f, Seta, color='tab:blue', lw=2)
ax.axvline(1 / tp, color='0.4', ls='--', label=f'Peak period {tp:g} s')
ax.set(xlim=(0, 0.3), xlabel='Frequency [Hz]', ylabel='Wave PSD [m²/Hz]',
       title=f'JONSWAP sea state · Hs={hs:g} m · Tp={tp:g} s · γ={gamma:g}')
ax.legend()
plt.show()

# %% [markdown]
# ## 2 · Waves → force → linear heave
#
# Use deep-water $k=\omega^2/g$ and the illustrative excitation transfer
# $H_{F\eta}=\rho g A_{\mathrm{eff}}\exp(-kd)$ [N/m]. Its depth attenuation is
# a simple teaching assumption; diffraction, radiation memory, and nonlinear
# drag are outside this model.
#
# $M\ddot x+C\dot x+Kx=F$, with $K=\rho g A_{\mathrm{eff}}$ and
# $C=2\zeta\sqrt{KM}$. Thus $H_{xF}=(K-M\omega^2+iC\omega)^{-1}$ [m/N],
# $S_{FF}=|H_{F\eta}|^2S_{\eta \eta}$, and $S_{XX}=|H_{xF}|^2S_{FF}$.

# %%
def wave_force_transfer(frequency, density, g, area, depth):
    wave_number = (2 * np.pi * frequency)**2 / g
    return density * g * area * np.exp(-wave_number * depth)


def heave_transfer(frequency, effective_mass, damping_coefficient, restoring):
    angular_frequency = 2 * np.pi * frequency
    return 1 / (restoring - effective_mass * angular_frequency**2
                + 1j * damping_coefficient * angular_frequency)


def gaussian_coefficients(spectrum, sample_count, frequency_spacing, generator):
    a, b = generator.normal(size=(2, spectrum.size))
    coefficients = (sample_count / 2) * np.sqrt(spectrum * frequency_spacing) * (a - 1j * b)
    coefficients[0] = coefficients[-1] = 0.0
    return coefficients


Hforce = wave_force_transfer(f, rho, gravity, effective_area, draft)           # Setup the wave-to-force transfer function based on the given parameters
Hheave = heave_transfer(f, mass, damping, stiffness)                           # Setup the force-to-heave transfer function based on the given parameters
Sf = abs(Hforce)**2 * Seta                                                     # Setup the amplitude of the transfer function for the force spectrum based on the wave spectrum
Sx = abs(Hheave)**2 * Sf                                                       # Setup the amplitude of the transfer function for the heave spectrum based on the force spectrum

assert np.allclose(Sx, abs(Hheave * Hforce)**2 * Seta)
m0, m2, m4 = [(omega**order * Sx).sum() * df for order in (0, 2, 4)]           # Calculate the spectral moments m0, m2, and m4 based on the heave spectrum Sx
sigma = np.sqrt(m0)                                                            # Calculate the standard deviation of the heave response based on the zeroth spectral moment
nu0 = np.sqrt(m2 / m0) / (2 * np.pi)                                           # zero-upcrossing rate based on the spectral moments m0 and m2 (narrow-band approximation)
nu_peak = np.sqrt(m4 / m2) / (2 * np.pi)                                       # peak-upcrossing rate based on the spectral moments m2 and m4 (wide-band approximation)
N_rayleigh, N_rice = T * nu0, T * nu_peak                                      # Calculate the expected number of peaks.
regularity = m2 / np.sqrt(m0 * m4)                                             # Calculate the spectral regularity based on the spectral moments m0, m2, and m4
bandwidth = np.sqrt(np.clip(1 - regularity**2, 0, 1))                          # Calculate the spectral bandwidth based on the spectral regularity
print(f'Heave RMS: {sigma:.4f} m; spectral bandwidth ε={bandwidth:.4f}')
print(f'Rayleigh cycle count N_R={N_rayleigh:.2f}; Rice local-peak count N_Rice={N_rice:.2f}')

fig_model = plt.figure(figsize=(14, 8), layout='constrained')
layout = fig_model.add_gridspec(2, 3, width_ratios=(1.4, 1, 1))             
sketch = fig_model.add_subplot(layout[:, 0])
sketch.axhspan(-16, 0, color='#e4f3fa')
sketch.axhline(0, color='tab:blue', lw=1.5)
for left in (-15, 7):
    sketch.add_patch(Rectangle((left, -12), 8, 4, fc='#586e85', ec='#233548'))
for left in (-13, 9):
    sketch.add_patch(Rectangle((left, -8), 4, 16, fc='#9aafc2', ec='#233548'))
sketch.add_patch(Rectangle((-18, 8), 36, 4, fc='#d9a65b', ec='#805a27'))
sketch.text(0, 10, 'Deck', ha='center', va='center')
sketch.text(0, -14, 'Submerged pontoons', ha='center')
sketch.text(-20, 1, 'Waterline', color='tab:blue')
sketch.annotate('', xy=(0, 6), xytext=(0, -6),
                arrowprops={'arrowstyle': '->', 'color': 'tab:red', 'lw': 2})
sketch.text(1, -3, 'Wave force F(t)', color='tab:red', rotation=90, va='center')
sketch.annotate('', xy=(21, 11), xytext=(21, 3),
                arrowprops={'arrowstyle': '<->', 'color': 'tab:green', 'lw': 2})
sketch.text(23, 7, 'Heave x(t)', color='tab:green', rotation=90, va='center')
sketch.text(0, 17, r'$M\ddot{x}+C\dot{x}+Kx=F(t)$', ha='center', fontsize=13)
sketch.text(0, -19, 'Illustrative geometry and linear load model\nEffective mass includes added mass',
            ha='center', va='center', fontsize=9)
sketch.set(xlim=(-24, 28), ylim=(-22, 22), title='Semi-rig · vertical heave')
sketch.set_aspect('equal')
sketch.axis('off')
for row, col, values, ylabel, title in (
        (0, 1, abs(Hforce) / 1e6, '|H_Fη| [MN/m]', 'Wave → force'),
        (0, 2, abs(Hheave) * 1e6, '|H_xF| [m/MN]', 'Force → heave'),
        (1, 1, Sf / 1e12, 'Force PSD [MN²/Hz]', 'Excitation spectrum'),
        (1, 2, Sx, 'Heave PSD [m²/Hz]', 'Response spectrum')):
    ax = fig_model.add_subplot(layout[row, col])
    ax.plot(f, values)
    ax.set(xlim=(0, 0.3), xlabel='Frequency [Hz]', ylabel=ylabel, title=title)
plt.show()

# %% [markdown]
# ### 2b · Play the waves and semi-rig response
#
# Show a two-minute excerpt of the first three-hour realization. The deep-water
# wave profile travels to the right: each Fourier coefficient is multiplied by
# $\exp(-ikx)$ while the inverse FFT uses $\exp(i\omega t)$. The excitation is
# evaluated at the rig centre, $x=0$, using the same transfer functions as above.
# The rig translates vertically by the computed heave, with no pitch or surge.
# Geometry is schematic; wave elevations and heave are in metres without amplification.
#
# Jupyter displays a JavaScript player with play/pause, frame stepping, and speed
# controls, without FFmpeg or a widget backend. A GUI Python session uses the
# Matplotlib animation window. Replay jumps because the clip is shorter than T.
# Keep `rig_animation` alive. To export a browser player manually, use
# `Path('semi_rig.html').write_text(rig_animation.to_jshtml(), encoding='utf-8')`
# after importing `Path` from `pathlib`.

# %%
animation_duration = 120.0     # physical seconds shown, not the full FFT period
playback_fps = 10             # dt=0.5 s gives five physical seconds per playback second
animation_indices = np.arange(min(int(animation_duration / dt) + 1, nt))
animation_times = animation_indices * dt
profile_x = np.linspace(-90, 90, 181)  # contains the rig centre x=0
# A separate generator repeats realization 1 without advancing the statistics RNG.
animation_coefficients = gaussian_coefficients(Seta, nt, df, np.random.default_rng(seed))
animation_heave = np.fft.irfft(Hheave * Hforce * animation_coefficients, n=nt)[animation_indices]
animation_wave_at_rig = np.fft.irfft(animation_coefficients, n=nt)[animation_indices]
profile_coefficients = (animation_coefficients[:, None]
                        * np.exp(-1j * (omega**2 / gravity)[:, None] * profile_x))
wave_profiles = np.fft.irfft(profile_coefficients, n=nt, axis=0)[animation_indices].copy()
del profile_coefficients
assert np.allclose(wave_profiles[:, len(profile_x) // 2], animation_wave_at_rig)

fig_animation, ax_animation = plt.subplots(figsize=(10, 4.8), dpi=85,
                                           layout='constrained')
animation_lower = min(-22, -14 + animation_heave.min(), wave_profiles.min() - 2)
animation_upper = max(20, 14 + animation_heave.max(), wave_profiles.max() + 2)
ax_animation.set(xlim=(profile_x[0], profile_x[-1]),
                 ylim=(animation_lower, animation_upper), aspect='equal',
                 xlabel='Horizontal position [m]', ylabel='Elevation [m]')
ax_animation.axhline(0, color='0.5', ls='--', lw=0.8, label='Mean water level')
water = Polygon(np.vstack(([profile_x[0], animation_lower],
                           np.column_stack((profile_x, wave_profiles[0])),
                           [profile_x[-1], animation_lower])),
                closed=True, fc='#e4f3fa', ec='none', zorder=1)
ax_animation.add_patch(water)
wave_line, = ax_animation.plot(profile_x, wave_profiles[0], color='tab:blue',
                               lw=1.8, zorder=4, label='Wave surface')
rig_parts = []
for left, bottom, width, height, color in (
        (-15, -12, 8, 4, '#586e85'), (7, -12, 8, 4, '#586e85'),
        (-13, -8, 4, 16, '#9aafc2'), (9, -8, 4, 16, '#9aafc2'),
        (-18, 8, 36, 4, '#d9a65b')):
    part = Rectangle((left, bottom), width, height, fc=color, ec='#233548', zorder=3)
    ax_animation.add_patch(part)
    rig_parts.append((part, bottom))
animation_title = ax_animation.set_title('')
animation_readout = ax_animation.text(0.02, 0.04, '', transform=ax_animation.transAxes,
                                     fontsize=9, zorder=5,
                                     bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
ax_animation.legend(loc='upper right', fontsize=8)


def update_rig_animation(index):
    """Update artists from the same physical wave/heave realization."""
    wave_line.set_ydata(wave_profiles[index])
    water.set_xy(np.vstack(([profile_x[0], animation_lower],
                            np.column_stack((profile_x, wave_profiles[index])),
                            [profile_x[-1], animation_lower])))
    for part, bottom in rig_parts:
        part.set_y(bottom + animation_heave[index])
    animation_title.set_text(f'Wave-driven semi-rig heave · t={animation_times[index]:.1f} s')
    animation_readout.set_text(f'Wave at rig: {animation_wave_at_rig[index]:+.2f} m   '
                              f'Heave: {animation_heave[index]:+.2f} m   '
                              f'Playback: {dt * playback_fps:g}× real time')
    return wave_line, water, animation_title, animation_readout, *(part for part, _ in rig_parts)


update_rig_animation(0)
rig_animation = FuncAnimation(fig_animation, update_rig_animation,
                              frames=len(animation_times), interval=1000 / playback_fps,
                              blit=False, repeat=False)
try:
    from IPython import get_ipython
    in_jupyter = getattr(get_ipython(), 'kernel', None) is not None
except ImportError:
    in_jupyter = False
if in_jupyter:
    from IPython.display import HTML, display
    # Close the static canvas so the notebook shows just the playable version.
    plt.close(fig_animation)
    with plt.rc_context({'animation.embed_limit': 50}):  # MB; all frames embedded
        display(HTML(rig_animation.to_jshtml(fps=playback_fps, default_mode='once')))
else:
    plt.show()

# %% [markdown]
# ## 3 · Independent three-hour realizations
#
# Normal sine/cosine coefficients create an exactly Gaussian ensemble. Apply both
# complex transfer functions to the **same** wave coefficients within a record:
# $\hat F=H_{F\eta}\hat\eta$, $\hat x=H_{xF}\hat F$. Independent records get
# independent coefficients. A single realization need not have the ensemble variance.

# %%
maxima           = np.empty(R)        # preallocate arrays for maxima
record_variances = np.empty(R)        # preallocate arrays for record variances
upcrossings = np.empty(R)             # preallocate arrays for zero-upcrossing counts
peak_counts = np.empty(R)             # preallocate arrays for local peak counts
examples = []
for index in range(R):
    wave_coefficients = gaussian_coefficients(Seta, nt, df, rng)
    force_coefficients = Hforce * wave_coefficients
    coefficients = Hheave * force_coefficients
    history = np.fft.irfft(coefficients, n=nt)
    maxima[index] = history.max()
    record_variances[index] = np.var(history)
    # Include the periodic seam in event counts to match the FFT ensemble.
    upcrossings[index] = np.count_nonzero((history < 0) & (np.roll(history, -1) >= 0))
    peak_counts[index] = np.count_nonzero((history > np.roll(history, 1))
                                         & (history > np.roll(history, -1)))
    if index < examples_to_keep:
        wave = np.fft.irfft(wave_coefficients, n=nt)
        force = np.fft.irfft(force_coefficients, n=nt)
        assert np.allclose(np.fft.rfft(wave), wave_coefficients, atol=1e-9)
        assert np.allclose(np.fft.rfft(force), force_coefficients, atol=1e-3)
        assert np.allclose(np.fft.rfft(history), coefficients, atol=1e-9)
        examples.append(history.copy())
print('Mean record variance / theoretical [m²]:', record_variances.mean(), m0)
print('Mean zero-upcrossing rate / theoretical [Hz]:', upcrossings.mean() / T, nu0)
print('Mean local-maximum rate / theoretical [Hz]:', peak_counts.mean() / T, nu_peak)

time_hours = np.arange(nt) * dt / 3600
fig_histories, axes = plt.subplots(examples_to_keep, 1, figsize=(13, 12),
                                    sharex=True, sharey=True, squeeze=False,
                                    layout='constrained')
for index, (ax, heave) in enumerate(zip(axes[:, 0], examples)):
    color = f'C{index % 10}'
    ax.plot(time_hours, heave, color=color, lw=0.65)
    peak_index = np.argmax(heave)
    ax.plot(time_hours[peak_index], heave[peak_index], 'o', color=color,
            ms=6, mec='black', mew=0.6,
            label=f'Maximum: {heave[peak_index]:.2f} m')
    ax.set(ylabel='Heave [m]', xlim=(0, T / 3600),
           title=f'Response realization {index + 1}')
    ax.legend(loc='upper right', fontsize=8)
axes[-1, 0].set_xlabel('Time [h]')
fig_histories.suptitle('Independent three-hour heave responses · one sea state')
plt.show()

# %% [markdown]
# ## 4 · Instantaneous values, local peaks, and block maxima
#
# Instantaneous heave has a Gaussian marginal. Narrow-band cycle peaks are
# approximately Rayleigh with scale $\sigma=\sqrt{m_0}$. The maximum of $N_R$
# independent such peaks has CDF $F_R^{N_R}$, where $N_R=T\nu_0$.
#
# Rice local peaks account for bandwidth: $a=m_2/\sqrt{m_0m_4}$,
# $\epsilon=\sqrt{1-a^2}$, and $z=u/\sigma$ give
# $F_p(u)=\Phi(z/\epsilon)-a e^{-z^2/2}\Phi(az/\epsilon)$.
# Local maxima can be negative. Use $N_{Rice}=T\sqrt{m_4/m_2}/(2\pi)$,
# not the zero-upcrossing count. These effective counts need not be integers.
# In the narrow-band limit, $a\to1$, Rice peaks become Rayleigh and the counts agree.
# Reference: [Gaussian-process local-peak heights, Eq. 18](https://pmc.ncbi.nlm.nih.gov/articles/PMC6733272/).
#
# Both powered peak models assume independent peaks; actual peaks may be dependent.
# Rice here means the Gaussian local-peak law, not scipy.stats.rice (Rician amplitudes).
# Gumbel parameters are fitted to the independent three-hour maxima, not time samples.

# %%
def rayleigh_peak_law(values, response_sigma):
    """Return log CDF and PDF, retaining accuracy in the positive tail."""
    z = np.asarray(values) / response_sigma
    logcdf = np.full_like(z, -np.inf)
    positive = z > 0
    logcdf[positive] = np.log(-np.expm1(-0.5 * z[positive]**2))
    return logcdf, stats.rayleigh.pdf(values, scale=response_sigma)


def rice_peak_law(values, response_sigma, a):
    """Gaussian local-maximum log CDF and PDF; a is the spectral regularity."""
    if not 0 <= a <= 1:
        raise ValueError('Spectral regularity must lie in [0, 1].')
    if 1 - a**2 < 1e-12:
        return rayleigh_peak_law(values, response_sigma)
    z = np.asarray(values) / response_sigma
    eps = np.sqrt(1 - a**2)
    if a == 0:
        return stats.norm.logcdf(z), stats.norm.pdf(z) / response_sigma
    # Use survival probabilities for positive tails and log subtraction for z<0.
    log_first = stats.norm.logcdf(z / eps)
    log_second = np.log(a) - z**2 / 2 + stats.norm.logcdf(a * z / eps)
    difference = np.minimum(log_second - log_first, 0)
    with np.errstate(divide='ignore'):
        logcdf = log_first + np.log(-np.expm1(difference))
    positive = z >= 0
    survival = (stats.norm.sf(z[positive] / eps)
                + a * np.exp(-z[positive]**2 / 2) * stats.norm.cdf(a * z[positive] / eps))
    logcdf[positive] = np.log1p(-survival)
    pdf = (eps * stats.norm.pdf(z / eps)
           + a * z * np.exp(-z**2 / 2) * stats.norm.cdf(a * z / eps)) / response_sigma
    return logcdf, np.maximum(pdf, 0)


def maximum_law(logcdf, peak_pdf, count):
    """CDF and PDF of an independent-peak maximum (effective count >= 1)."""
    if count < 1:
        raise ValueError('The maximum approximation requires at least one peak.')
    cdf = np.exp(count * logcdf)
    pdf = peak_pdf if count == 1 else count * np.exp((count - 1) * logcdf) * peak_pdf
    return cdf, pdf


location, scale = stats.gumbel_r.fit(maxima)                                                 # define the gumbel distribution parameters based on the maxima (this is based on maxima of each time series)
gumbel = stats.gumbel_r(loc=location, scale=scale)                                           # define the gumbel distribution based on the location and scale parameters

grid = np.linspace(-4 * sigma, max(7 * sigma, 1.15 * maxima.max()), 1600)                    # define the grid for plotting the distributions
rayleigh_logcdf, rayleigh_pdf = rayleigh_peak_law(grid, sigma)                               # define the rayleigh distribution (pdf) based on the grid and sigma
rayleigh_cdf = np.exp(rayleigh_logcdf)                                                       # define the rayleigh distribution (cdf) based on the logcdf
rayleigh_max_cdf, rayleigh_max_pdf = maximum_law(rayleigh_logcdf, rayleigh_pdf, N_rayleigh)  # define the maximum distribution (cdf and pdf) based on the rayleigh distribution and the number of peaks

rice_logcdf, rice_pdf = rice_peak_law(grid, sigma, regularity)                               # define the rice distribution (pdf) based on the grid, sigma, and regularity
rice_max_cdf, rice_max_pdf = maximum_law(rice_logcdf, rice_pdf, N_rice)                      # define the maximum distribution (cdf and pdf) based on the rice distribution and the number of peaks
ordered = np.sort(maxima)
ecdf = np.arange(1, R + 1) / R

print('Gumbel location / scale [m]:', location, scale)
print('Empirical q50 / q90 / q99 [m]:', np.quantile(maxima, [.5, .9, .99]))
print('Gumbel q50 / q90 / q99 [m]:', gumbel.ppf([.5, .9, .99]))


fig_probability, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
for pdf, cdf, label, style in (
        (stats.norm.pdf(grid, scale=sigma), stats.norm.cdf(grid, scale=sigma),
         'Gaussian instantaneous heave', '-'),
        (rayleigh_pdf, rayleigh_cdf, 'Rayleigh cycle peaks', '--'),
        (rayleigh_max_pdf, rayleigh_max_cdf, 'Rayleigh^N_R · 3 h maximum', '-.')):
    axes[0, 0].plot(grid, pdf, style, label=label)
    axes[0, 1].plot(grid, cdf, style, label=label)
axes[1, 0].hist(maxima, bins=22, density=True, alpha=0.35, color='0.4',
                label=f'Block maxima · {R} records')
axes[1, 1].step(np.r_[ordered[0], ordered], np.r_[0, ecdf], where='post',
                color='0.4', label='Block-maxima ECDF')
for pdf, cdf, label, style in (
        (gumbel.pdf(grid), gumbel.cdf(grid), 'Gumbel fit', '-'),
        (rayleigh_max_pdf, rayleigh_max_cdf, 'Rayleigh^N_R', '--'),
        (rice_max_pdf, rice_max_cdf, 'Rice^N_Rice · local peaks', ':')):
    axes[1, 0].plot(grid, pdf, style, label=label)
    axes[1, 1].plot(grid, cdf, style, label=label)
for row in range(2):
    for col in range(2):
        ax = axes[row, col]
        ax.set(xlabel='Heave / peak / maximum [m]' if row == 0 else 'Three-hour maximum heave [m]',
               ylabel='PDF [1/m]' if col == 0 else 'CDF')
        ax.legend(fontsize=8)
        if col == 1:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)
        if row == 0:
            ax.set_xlim(grid[0], grid[-1])
        else:
            ax.set_xlim(min(gumbel.ppf(.001), ordered[0] - .3 * sigma),
                        max(gumbel.ppf(.999), ordered[-1] + .3 * sigma))
axes[0, 0].set_title('Instantaneous response → cycle peaks → block maximum')
axes[0, 1].set_title('CDF comparison')
axes[1, 0].set_title('Simulated block maxima and three models')
axes[1, 1].set_title('Block-maximum CDF comparison')
fig_probability.suptitle(f'3 h heave extremes · N_R={N_rayleigh:.1f} · '
                         f'N_Rice={N_rice:.1f} · ε={bandwidth:.3f}')
plt.show()

# %% [markdown]
# ## Numerical checks and interpretation
#
# Check normalized densities and CDF derivatives on a broad grid, including the
# negative local peaks of a broad-band process. A finer evaluation of the same
# Fourier record illustrates time-grid error; it is not a full tail convergence study.

# %%
check_grid = np.linspace(-12 * sigma, 12 * sigma, 40001)
for a in (0.0, 0.5, regularity, 0.999, 1.0):
    logcdf, pdf = rice_peak_law(check_grid, sigma, a)
    cdf = np.exp(logcdf)
    assert np.all(np.diff(cdf) >= -1e-12)
    assert np.isclose(integrate.trapezoid(pdf, check_grid), 1, atol=2e-6)
    # At a=1 the Rayleigh support boundary has a kink in the PDF. Compare
    # CDF increments with integrated density, avoiding a derivative stencil
    # that straddles that boundary.
    integrated_pdf = integrate.cumulative_trapezoid(pdf, check_grid, initial=0)
    assert np.allclose(cdf, integrated_pdf, atol=2e-6)
    max_cdf, max_pdf = maximum_law(logcdf, pdf, N_rice)
    assert np.all(np.diff(max_cdf) >= -1e-12)
    assert np.isclose(integrate.trapezoid(max_pdf, check_grid), 1, atol=2e-6)
    assert np.allclose(np.gradient(max_cdf, check_grid), max_pdf, atol=2e-5 / sigma)
limit_cdf, limit_pdf = rice_peak_law(check_grid, sigma, 1 - 1e-8)
reference_cdf, reference_pdf = rayleigh_peak_law(check_grid, sigma)
assert np.allclose(np.exp(limit_cdf), np.exp(reference_cdf), atol=2e-6)
assert np.allclose(limit_pdf, reference_pdf, atol=1e-4 / sigma)
rayleigh_check_cdf, rayleigh_check_pdf = maximum_law(reference_cdf, reference_pdf, N_rayleigh)
assert np.isclose(integrate.trapezoid(rayleigh_check_pdf, check_grid), 1, atol=2e-6)
assert np.allclose(np.gradient(rayleigh_check_cdf, check_grid), rayleigh_check_pdf,
                   atol=2e-5 / sigma)
assert np.isclose(integrate.trapezoid(gumbel.pdf(check_grid), check_grid), 1, atol=2e-6)
fine_coefficients = np.zeros(nt + 1, dtype=complex)
fine_coefficients[:coefficients.size] = 2 * coefficients
fine_history = np.fft.irfft(fine_coefficients, n=2 * nt)
assert np.allclose(fine_history[::2], history)
print('Same-record maximum at dt / dt/2 [m]:', history.max(), fine_history.max())
print('Probability normalization, CDF/PDF, narrow-band, and Fourier checks passed.')
print('Exercise: vary damping or Tp. How do bandwidth, peak counts, and maximum models change?')
print('Peak independence and Gumbel tail extrapolation remain modeling approximations.')