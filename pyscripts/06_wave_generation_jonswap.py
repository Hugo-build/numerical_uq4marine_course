# %% [markdown]
# # 06 · Irregular waves from a JONSWAP spectrum
#
# **Goal:** generate reproducible elevation histories at a fixed station,
# animate a 2D wave profile eta(x,t), and animate a directional 3D sea surface
# eta(x,y,t). Run cells in order, or execute this file with Python.
# Outputs go to ../previews/06_wave_generation_jonswap/ relative to this file.
# Open index.html in a browser for offline play/pause and frame sliders.
# No FFmpeg, additional packages, or internet connection are needed to play.
#
# This is linear, stationary, deep-water wave synthesis, without currents,
# shoaling, breaking, or wave-wave interactions. Directions indicate where
# waves travel, measured counterclockwise from +x (not nautical coming-from).
# The 2D example is long-crested; the 3D example has directional spreading.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from scipy.signal import welch

plt.rcParams.update({'figure.facecolor': 'white', 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 10,
                     'animation.embed_limit': 100})  # MB per HTML animation

# Edit these settings, then rerun all downstream cells.
hs = 5.0                       # spectral significant wave height Hm0 [m]
tp = 14.0                      # spectral peak period [s]
gamma = 3.3                    # JONSWAP peak enhancement (1 gives PM shape)
gravity = 9.81                 # [m/s^2]
seed = 2026
record_duration = 600.0         # [s]; also sets df = 1 / duration
time_step = 0.25                # [s] for station records
maximum_frequency = 0.30       # [Hz]; truncated spectrum is renormalized
station_x, station_y = 0.0, 0.0 # [m]
mean_direction_deg = 0.0
direction_offsets_deg = np.array([-60., -30., -15., 0., 15., 30., 60.])
# Discrete energy fractions, not a continuous density per radian.
# Equal weights reproduce the directional choice in the MATLAB reference.
direction_weights = np.ones(len(direction_offsets_deg))
x = np.linspace(-250.0, 250.0, 200)
y = np.linspace(-130.0, 130.0, 100)
animation_duration = 30.0       # simulated seconds
animation_time_step = 0.25     # seconds per frame
playback_fps = 4                # real-time playback: fps * frame dt = 1
export_html = True             # HTML works even with a noninteractive backend
show_figures = True             # False for batch/headless use

# __file__ is available in script execution; interactive cells use the cwd.
if '__file__' in globals():
    course_root = Path(__file__).resolve().parents[1]
else:
    course_root = Path.cwd().resolve()
    if course_root.name == 'pyscripts':
        course_root = course_root.parent
output_dir = course_root / 'previews' / '06_wave_generation_jonswap'
output_dir.mkdir(parents=True, exist_ok=True)
print('Wave-generation outputs:', output_dir)

# %% [markdown]
# ## Spectrum, energy and component amplitudes
#
# Use a one-sided spectrum in Hz: S_f has units m²/Hz. With fp = 1/Tp,
# S_f(f) is proportional to
# f^(-5) exp[-5/4 (fp/f)^4] gamma^r,
# r = exp[-(f-fp)^2/(2 sigma² fp²)], sigma = 0.07 below fp, 0.09 above.
# Scale the discrete spectrum so m0 = sum(S_f df) = (Hs/4)^2.
# This avoids the Hs mismatch in the MATLAB reference when its conversion
# factor is set to one. Normalization is on the retained frequency band;
# changing the cutoff changes the model and should be checked for convergence.
#
# Angular-frequency conversion: omega = 2 pi f and S_omega = S_f/(2 pi).
# Never combine S_f with domega in the amplitude formula.
# For energy fraction w_j in direction j:
# a_ij = sqrt(2 S_f(f_i) df w_j), sum_j w_j = 1.
#
# JONSWAP shape and Hm0 normalization reference:
# https://wavespectra.readthedocs.io/en/latest/construction.html

# %%
def jonswap_spectrum(frequency, significant_height, peak_period,
                     peak_enhancement=3.3):
    """One-sided S_f [m²/Hz] normalized on a uniform positive frequency grid."""
    frequency = np.asarray(frequency, dtype=float)
    if (frequency.ndim != 1 or frequency.size < 2
            or not np.all(np.isfinite(frequency)) or np.any(frequency <= 0)):
        raise ValueError('Use at least two finite positive frequencies in Hz.')
    spacing = np.diff(frequency)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError('The frequency grid must be increasing and uniform.')
    if (not np.all(np.isfinite([significant_height, peak_period, peak_enhancement]))
            or significant_height <= 0 or peak_period <= 0 or peak_enhancement < 1):
        raise ValueError('Hs and Tp must be positive, and gamma must be >= 1.')
    peak_frequency = 1 / peak_period
    if not frequency[0] <= peak_frequency <= frequency[-1]:
        raise ValueError('The frequency grid must include the spectral peak.')
    sigma = np.where(frequency <= peak_frequency, 0.07, 0.09)
    peak_shape = np.exp(-0.5 * ((frequency - peak_frequency)
                               / (sigma * peak_frequency))**2)
    # Logarithms keep the low-frequency exponential numerically stable.
    log_shape = (-5 * np.log(frequency) - 1.25 * (peak_frequency / frequency)**4
                 + np.log(peak_enhancement) * peak_shape)
    shape = np.exp(log_shape - log_shape.max())
    return shape * (significant_height / 4)**2 / (shape.sum() * spacing[0])


if (not np.all(np.isfinite([record_duration, time_step, maximum_frequency,
                           animation_duration, animation_time_step, playback_fps]))
        or min(record_duration, time_step, maximum_frequency, animation_duration,
               animation_time_step, playback_fps) <= 0):
    raise ValueError('Durations, sampling steps, cutoff, and FPS must be positive.')
df = 1 / record_duration
frequency = np.arange(1, int(np.floor(maximum_frequency / df + 1e-10)) + 1) * df
spectrum = jonswap_spectrum(frequency, hs, tp, gamma)
omega = 2 * np.pi * frequency
wave_number = omega**2 / gravity   # deep water: omega² = g k
amplitude = np.sqrt(2 * spectrum * df)
m0 = np.sum(spectrum * df)
time = np.arange(0.0, record_duration, time_step)
frame_times = np.arange(0.0, animation_duration, animation_time_step)
if frequency[-1] >= min(0.5 / time_step, 0.5 / animation_time_step):
    raise ValueError('Reduce time steps to satisfy temporal Nyquist sampling.')
if max(np.diff(x).max(), np.diff(y).max()) * wave_number.max() >= np.pi:
    raise ValueError('Refine the spatial grids to resolve the shortest wave.')
print(f'Hs={hs:g} m, Tp={tp:g} s, gamma={gamma:g}; '
      f'{frequency.size} frequencies, df={df:.6f} Hz')
print(f'Spectral Hm0 = {4 * np.sqrt(m0):.6f} m; '
      f'peak-grid period = {1 / frequency[np.argmax(spectrum)]:.3f} s')

# %% [markdown]
# ## Random phases and elevation synthesis
#
# eta(x,y,t) = sum_ij a_ij cos(k_i cos(theta_j) x +
#                              k_i sin(theta_j) y - omega_i t + phi_ij).
# Independent uniform phases phi_ij in [0,2 pi) give ensemble variance
# sum(a_ij²)/2 = m0. A finite record's measured variance can differ, especially
# when several directions share a frequency. Fixed amplitudes and random
# phases approximate Gaussian seas with many components; they do not define
# an exactly Gaussian ensemble. Integer frequencies i/record_duration make
# this realization periodic in time with record_duration.

# %%
rng = np.random.default_rng(seed)
profile_phase = rng.uniform(0, 2 * np.pi, (3, frequency.size))
direction_weights = np.asarray(direction_weights, dtype=float)
if (direction_weights.shape != direction_offsets_deg.shape
        or not np.all(np.isfinite(direction_weights))
        or np.any(direction_weights < 0) or direction_weights.sum() <= 0):
    raise ValueError('Direction weights must be finite, nonnegative, with positive sum.')
direction_weights = direction_weights / direction_weights.sum()
directions = np.radians(mean_direction_deg + direction_offsets_deg)
directional_amplitude = amplitude[:, None] * np.sqrt(direction_weights[None, :])
directional_phase = rng.uniform(0, 2 * np.pi, directional_amplitude.shape)
kx = wave_number[:, None] * np.cos(directions)[None, :]
ky = wave_number[:, None] * np.sin(directions)[None, :]


def station_history(times, phase, amplitudes, wave_kx, wave_ky):
    """Sum flattened components at the selected station; times is a 1D array."""
    component_omega = np.broadcast_to(omega[:, None], np.shape(phase)).ravel()
    spatial_phase = (wave_kx * station_x + wave_ky * station_y + phase).ravel()
    # Chunk time to keep memory bounded for longer records.
    times = np.asarray(times)
    result = np.empty(times.size)
    for start in range(0, times.size, 512):
        stop = min(start + 512, times.size)
        angles = spatial_phase[:, None] - component_omega[:, None] * times[start:stop]
        result[start:stop] = np.asarray(amplitudes).ravel() @ np.cos(angles)
    return result


long_kx = wave_number[:, None] * np.cos(np.radians(mean_direction_deg))
long_ky = wave_number[:, None] * np.sin(np.radians(mean_direction_deg))
histories = np.array([
    station_history(time, phase[:, None], amplitude[:, None], long_kx, long_ky)
    for phase in profile_phase
])
directional_history = station_history(time, directional_phase, directional_amplitude, kx, ky)
np.savetxt(output_dir / 'station_time_series.csv',
           np.column_stack((time, histories.T, directional_history)), delimiter=',',
           header='time_s,long_crested_1_m,long_crested_2_m,long_crested_3_m,directional_m',
           comments='')
print('Fixed station [m]:', (station_x, station_y))
for index, history in enumerate([*histories, directional_history]):
    print(f'Record {index + 1}: mean={history.mean():.4f} m, '
          f'4 std(eta)={4 * history.std():.3f} m')

# %% [markdown]
# ## Inspect the spectrum and station records
#
# Welch's estimate is a noisy, smoothed density, not the exact input spectrum.
# Three long-crested records use independent phases but identical Hs, Tp and
# gamma. The first record drives the 2D animation; the directional record
# drives the 3D animation. The full record and its first 120 seconds are shown.

# %%
fig_spectrum, ax = plt.subplots(figsize=(9, 4), layout='constrained')
estimated_frequency, estimated_spectrum = welch(
    histories[0], fs=1 / time_step, nperseg=min(1024, time.size))
ax.plot(frequency, spectrum, label='Input JONSWAP', linewidth=2)
ax.plot(estimated_frequency, estimated_spectrum, label='Welch: record 1', alpha=0.75)
ax.axvline(1 / tp, color='grey', linestyle=':', label='1 / Tp')
ax.set(xlim=(0, frequency[-1]), xlabel='Frequency f [Hz]',
       ylabel='One-sided S(f) [m²/Hz]', title='JONSWAP spectrum and synthesized record')
ax.legend()
fig_spectrum.savefig(output_dir / 'spectrum.png', dpi=140)

fig_histories, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
for index, history in enumerate(histories):
    axes[0].plot(time, history, linewidth=0.8, label=f'Long-crested record {index + 1}')
axes[1].plot(time, histories[0], label='Long-crested record 1 (2D)')
axes[1].plot(time, directional_history, label='Directional sea (3D)', alpha=0.8)
axes[1].set_xlim(0, min(120, record_duration))
for ax in axes:
    ax.set(xlabel='Time [s]', ylabel='Elevation eta [m]')
    ax.legend(loc='upper right', fontsize=8)
fig_histories.suptitle(f'Fixed station ({station_x:g}, {station_y:g}) m · '
                      f'Hs={hs:g} m, Tp={tp:g} s, gamma={gamma:g}')
fig_histories.savefig(output_dir / 'station_time_series.png', dpi=140)

# %% [markdown]
# ## Precompute frames using the same wave components
#
# For the directional surface, cos(A+B) = cos(A)cos(B) - sin(A)sin(B)
# turns the sum into small matrix products, following the MATLAB reference.
# This avoids a large (frequency × direction × y × x × time) array.
# Surface arrays are shaped (y,x), matching np.meshgrid(x,y).

# %%
profile_frames = np.array([
    amplitude @ np.cos(long_kx * x[None, :] + long_ky * station_y
                       - omega[:, None] * t + profile_phase[0, :, None])
    for t in frame_times
])
surface_frames = np.zeros((len(frame_times), len(y), len(x)))
for j in range(len(directions)):
    y_phase = ky[:, j, None] * y[None, :]
    cos_y, sin_y = np.cos(y_phase), np.sin(y_phase)
    for index, t in enumerate(frame_times):
        x_phase = (kx[:, j, None] * x[None, :] - omega[:, None] * t
                   + directional_phase[:, j, None])
        surface_frames[index] += (
            (directional_amplitude[:, j, None] * np.cos(x_phase)).T @ cos_y
            - (directional_amplitude[:, j, None] * np.sin(x_phase)).T @ sin_y
        ).T
profile_station_frames = station_history(
    frame_times, profile_phase[0, :, None], amplitude[:, None], long_kx, long_ky)
surface_station_frames = station_history(
    frame_times, directional_phase, directional_amplitude, kx, ky)
print(f'Prepared {len(frame_times)} frames; '
      f'playback speed = {playback_fps * animation_time_step:g} × real time.')

# %% [markdown]
# ## Verify energy and space-time consistency
#
# A spatial slice and a station history must agree when evaluated at the
# same coordinates and time. Compare the factored 3D sum with a direct sum
# at an off-axis grid point to check signs, directions and array orientation.

# %%
assert np.isclose(0.5 * np.sum(amplitude**2), (hs / 4)**2)
assert np.isclose(0.5 * np.sum(directional_amplitude**2), (hs / 4)**2)
assert np.all(np.isfinite(surface_frames))
ix, iy, iframe = len(x) // 3, len(y) // 3, len(frame_times) // 2
direct_elevation = np.sum(directional_amplitude * np.cos(
    kx * x[ix] + ky * y[iy] - omega[:, None] * frame_times[iframe] + directional_phase))
assert np.isclose(surface_frames[iframe, iy, ix], direct_elevation, atol=1e-12)
if np.any(np.isclose(x, station_x)):
    station_ix = np.argmin(abs(x - station_x))
    assert np.allclose(profile_frames[:, station_ix], profile_station_frames, atol=1e-12)
    if np.any(np.isclose(y, station_y)):
        station_iy = np.argmin(abs(y - station_y))
        assert np.allclose(surface_frames[:, station_iy, station_ix],
                           surface_station_frames, atol=1e-12)
print('Passed: spectral energy, finite surfaces, direct sum, and station/slice consistency.')

# %% [markdown]
# ## 2D animation: a long-crested wave profile
#
# Keep the FuncAnimation objects alive. In a GUI backend plt.show() plays
# them; the saved HTML players also work from VS Code, Spyder and headless runs.
# In Jupyter, optionally display HTML(animation_2d.to_jshtml(fps=playback_fps)).

# %%
elevation_limit = 1.1 * max(hs, np.max(abs(profile_frames)), np.max(abs(surface_frames)),
                          np.max(abs(profile_station_frames)), np.max(abs(surface_station_frames)))
fig_2d, ax_2d = plt.subplots(figsize=(9, 4), layout='constrained', dpi=85)
profile_line, = ax_2d.plot(x, profile_frames[0], color='tab:blue', linewidth=2)
station_dot, = ax_2d.plot([station_x], [profile_station_frames[0]], 'o', color='crimson',
                         label='Fixed station')
ax_2d.axhline(0, color='grey', linewidth=0.8)
ax_2d.set(xlim=(x[0], x[-1]), ylim=(-elevation_limit, elevation_limit),
          xlabel='x [m]', ylabel='Elevation eta [m]')
ax_2d.legend(loc='upper right')
title_2d = ax_2d.set_title('')


def update_2d(index):
    profile_line.set_ydata(profile_frames[index])
    station_dot.set_ydata([profile_station_frames[index]])
    title_2d.set_text(f'Long-crested JONSWAP · y={station_y:g} m · t={frame_times[index]:.2f} s')
    return profile_line, station_dot, title_2d


animation_2d = FuncAnimation(fig_2d, update_2d, frames=len(frame_times),
                             interval=1000 / playback_fps, blit=False)

# %% [markdown]
# ## 3D animation: a sea surface with directional spreading
#
# Colour and elevation limits stay fixed between frames. The z dimension is
# visually exaggerated so the surface is readable; x, y and eta are in metres.
# Set all direction weights except the central one to zero for long crests.

# %%
xx, yy = np.meshgrid(x, y)
fig_3d = plt.figure(figsize=(9, 5.5), layout='constrained', dpi=85)
ax_3d = fig_3d.add_subplot(111, projection='3d')
colour_norm = Normalize(-elevation_limit, elevation_limit)


def draw_surface(index):
    return ax_3d.plot_surface(xx, yy, surface_frames[index], cmap='Blues_r',
                              norm=colour_norm, rcount=len(y), ccount=len(x),
                              linewidth=0, antialiased=False)


surface_artist = draw_surface(0)
station_marker, = ax_3d.plot([station_x], [station_y], [surface_station_frames[0]],
                            'o', color='crimson', markersize=5, label='Fixed station')
ax_3d.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
          zlim=(-elevation_limit, elevation_limit),
          xlabel='x [m]', ylabel='y [m]', zlabel='Elevation eta [m]')
ax_3d.set_box_aspect((2.0, 1.0, 0.55))
ax_3d.view_init(elev=25, azim=-65)
ax_3d.legend(loc='upper left')
title_3d = ax_3d.set_title('')
fig_3d.colorbar(surface_artist, ax=ax_3d, shrink=0.65, pad=0.07, label='Elevation [m]')


def update_3d(index):
    global surface_artist
    surface_artist.remove()
    surface_artist = draw_surface(index)
    station_marker.set_data_3d([station_x], [station_y], [surface_station_frames[index]])
    title_3d.set_text(f'Directional JONSWAP · t={frame_times[index]:.2f} s\n'
                      f'Hs={hs:g} m, Tp={tp:g} s · vertical scale exaggerated')
    return surface_artist, station_marker, title_3d


animation_3d = FuncAnimation(fig_3d, update_3d, frames=len(frame_times),
                             interval=1000 / playback_fps, blit=False)

# %% [markdown]
# ## Save standalone playable HTML and data
#
# Browser players contain their own frame images and JavaScript. Exporting
# the 3D frames takes longer than calculating the wave field. The end-to-start
# replay jump is expected: this clip is shorter than the realization's period.
# For MP4 instead, install FFmpeg and use animation_3d.save('waves.mp4', fps=...).

# %%
if export_html:
    for name, animation in [('waves_2d', animation_2d), ('waves_3d', animation_3d)]:
        print(f'Exporting {name} HTML player...', flush=True)
        player = animation.to_jshtml(fps=playback_fps, default_mode='once')
        # Matplotlib's player uses Font Awesome icons from a CDN. Replace its
        # stylesheet with local text icons so the controls also work offline.
        player = player.replace(
            '<link rel="stylesheet"\n'
            'href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">',
            '<style>.fa{font-style:normal}.fa:before{content:attr(aria-label)}'
            '.fa-play:before{content:"▶"}.fa-pause:before{content:"Ⅱ"}'
            '.fa-flip-horizontal:before{content:"◀"}'
            '.fa-backward:before{content:"◀"}.fa-forward:before{content:"▶"}'
            '.fa-fast-backward:before{content:"|◀"}.fa-fast-forward:before{content:"▶|"}'
            '.fa-step-backward:before{content:"|◁"}.fa-step-forward:before{content:"▷|"}'
            '.fa-minus:before{content:"−"}.fa-plus:before{content:"+"}</style>')
        document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                    f'<title>{name} · JONSWAP waves</title>'
                    '<body style="font-family:system-ui;max-width:1100px;margin:24px auto">'
                    '<p><a href="index.html">← Wave generation overview</a></p>'
                    '<p>Use play/pause or drag the frame slider. Time is shown in the plot.</p>'
                    + player + '</body></html>')
        (output_dir / f'{name}.html').write_text(document, encoding='utf-8')
    (output_dir / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>06 · JONSWAP wave generation</title>'
        '<body style="font-family:system-ui;max-width:1000px;margin:32px auto;line-height:1.5">'
        '<h1>06 · JONSWAP wave generation</h1>'
        f'<p>Hs = {hs:g} m · Tp = {tp:g} s · gamma = {gamma:g} · seed = {seed}</p>'
        '<p>Linear deep-water waves with random phases. The 2D profile is long-crested; '
        f'the 3D surface distributes energy over {len(directions)} directions. '
        'Both players work offline.</p>'
        '<p><a href="waves_2d.html">Play 2D wave profile</a> · '
        '<a href="waves_3d.html">Play 3D sea surface</a> · '
        '<a href="station_time_series.csv">Download station histories (CSV)</a></p>'
        '<p>Red markers follow the fixed station. Vertical scale is exaggerated in 3D. '
        'Use the slider to inspect a frame, or play the sequence. Replaying jumps back '
        'to the start of the clip.</p>'
        '<img src="spectrum.png" alt="JONSWAP spectrum" style="width:100%">'
        '<img src="station_time_series.png" alt="Fixed-station wave histories" style="width:100%">'
        '</body></html>', encoding='utf-8')
    print('Open this file in a browser:', output_dir / 'index.html')

if show_figures:
    plt.show()
else:
    plt.close('all')
print('Exercise: vary Hs, Tp, gamma, seed, and direction weights; '
      'then refine the frequency cutoff and spatial/time grids.')
