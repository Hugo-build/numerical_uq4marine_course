# %% [markdown]
# # 06 · Short-term extremes of a linear dynamic response
# 
# **Goal:** generate a stationary linear Gaussian response, collect independent record maxima,
# fit a Gumbel model, and compare approximation methods. The quantity is the **maximum positive displacement**
# $M_T=\max_{0\le t<T}x(t)$, not $\max|x(t)|$.
# 
# Use a synthetic narrow-band Gaussian **force** PSD, a linear oscillator transfer function,
# and normally distributed Fourier coefficients. Each record is periodic with duration $T$; this is a finite-grid
# stationary synthesis, not a transient integration from rest. Independent records use independent coefficients.
# No JONSWAP wave-to-force conversion is implied. Rayleigh peak approximations require narrow-band behavior.
# References: [Gumbel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html),
# [GEV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, integrate, optimize, linalg
np.set_printoptions(precision=5, suppress=True)
plt.rcParams.update({'figure.figsize': (8, 4), 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 11})
rng = np.random.default_rng(2026)
print('Imports ready. Random seed: 2026. Units: SI unless stated otherwise.')

# %% [markdown]
# ## Define PSD transfer and synthesis
# 
# For a one-sided PSD in Hz, $S_x(f)=|H(2\pi f)|^2S_F(f)$. The variance is the integral of $S_x$ over positive frequencies.

# %%
def response_spectrum(force_rms, peak_hz, bandwidth_hz, duration, dt,
                      mass=1000., natural_hz=.15, damping=.08):
    # One-sided PSD per Hz, not per rad/s. No DC or Nyquist component.
    nt = int(round(duration/dt))
    assert nt%2==0 and np.isclose(nt*dt,duration)
    f = np.fft.rfftfreq(nt,dt); df = 1/duration
    shape = np.exp(-.5*((f-peak_hz)/bandwidth_hz)**2)
    shape[0] = 0.; shape[-1] = 0.
    Sf = force_rms**2*shape/(shape.sum()*df)
    omega = 2*np.pi*f; wn = 2*np.pi*natural_hz
    H = 1/(mass*(wn**2-omega**2+2j*damping*wn*omega))
    Sx = abs(H)**2*Sf
    return f, Sf, Sx, H

def gaussian_history(Sx, duration, dt, generator):
    nt = int(round(duration/dt)); df = 1/duration
    # Normal sine/cosine coefficients create an exactly Gaussian ensemble.
    a = generator.normal(size=Sx.size); b = generator.normal(size=Sx.size)
    coefficients = (nt/2)*np.sqrt(Sx*df)*(a-1j*b)
    coefficients[0] = 0.; coefficients[-1] = 0.
    return np.fft.irfft(coefficients,n=nt), coefficients
print('PSD transfer and stationary Gaussian FFT synthesis functions defined.')

# %% [markdown]
# ## Inspect excitation and response
# 
# The discrete sums match the synthesis convention. $m_2=\int(2\pi f)^2S_x(f)df$ is velocity variance; the mean zero-upcrossing rate is $\nu_0=(2\pi)^{-1}\sqrt{m_2/m_0}$.

# %%
T, dt, force_rms, peak_hz, bandwidth = 600., .1, 100., .14, .012
f,Sf,Sx,H = response_spectrum(force_rms,peak_hz,bandwidth,T,dt)
df = 1/T; m0 = Sx.sum()*df; m2 = ((2*np.pi*f)**2*Sx).sum()*df
sigma = np.sqrt(m0); nu0 = np.sqrt(m2/m0)/(2*np.pi)
npeaks = max(1,int(round(T*nu0)))
print('Response RMS [m]:',sigma,'upcrossings per second:',nu0,'approximate peak count:',npeaks)
assert np.isclose(Sf.sum()*df,force_rms**2)
fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot(f,Sf); axes[0].set(xlim=(0,.4),xlabel='f [Hz]',ylabel='force PSD [N²/Hz]')
axes[1].plot(f,Sx); axes[1].set(xlim=(0,.4),xlabel='f [Hz]',ylabel='response PSD [m²/Hz]')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Generate independent records
# 
# For a teaching run use 60 records; the default is 300. A single history need not have exactly the ensemble variance, because its Fourier amplitudes are random.

# %%
R = 300; maxima = np.empty(R); record_variances = np.empty(R)
upcrossings = np.empty(R)
for i in range(R):
    history, coefficients = gaussian_history(Sx,T,dt,rng)
    maxima[i] = history.max(); record_variances[i] = np.var(history)
    upcrossings[i] = np.sum((history[:-1]<0)&(history[1:]>=0))
print('Records:',R,'duration [s]:',T,'time step [s]:',dt)
print('Mean record variance/theoretical:',record_variances.mean(),m0)
print('Mean zero-upcrossing rate/theoretical:',upcrossings.mean()/T,nu0)
time = np.arange(history.size)*dt
fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].plot(time,history); axes[0].set(xlim=(0,120),xlabel='time [s]',ylabel='response [m]')
axes[1].hist(maxima,bins=25,density=True); axes[1].set(xlabel='600 s maximum [m]',ylabel='density [1/m]')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Fit and compare maximum distributions
# 
# Independent Rayleigh peaks give $F_R(u)^{n_p}$. Rice/Poisson approximation gives $\exp[-T\nu_0\exp(-u^2/(2m_0))]$ at high positive thresholds. These are approximations for this process. Raising the Gaussian marginal CDF to the number of time samples incorrectly assumes sample independence.

# %%
location, scale = stats.gumbel_r.fit(maxima)
gumbel = stats.gumbel_r(loc=location,scale=scale)
grid = np.linspace(.8*maxima.min(),1.15*maxima.max(),500)
rayleigh_cdf = stats.rayleigh(scale=sigma).cdf(grid)**npeaks
rice_cdf = np.exp(-T*nu0*np.exp(-grid**2/(2*m0)))
iid_cdf = stats.norm(scale=sigma).cdf(grid)**history.size
ordered = np.sort(maxima); ecdf = np.arange(1,R+1)/R
print('Gumbel location/scale:',location,scale)
print('Empirical q50/q90/q99:',np.quantile(maxima,[.5,.9,.99]))
print('Gumbel q50/q90/q99:',gumbel.ppf([.5,.9,.99]))
fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].step(ordered,ecdf,where='post',label='record maxima')
axes[0].plot(grid,gumbel.cdf(grid),label='Gumbel fit')
axes[0].plot(grid,rayleigh_cdf,'--',label='independent Rayleigh peaks')
axes[0].plot(grid,rice_cdf,':',label='Rice / Poisson')
axes[0].plot(grid,iid_cdf,'-.',label='IID time samples (incorrect assumption)')
axes[0].set(xlabel='maximum [m]',ylabel='CDF'); axes[0].legend(fontsize=8)
q = (np.arange(1,R+1)-.5)/R
axes[1].plot(gumbel.ppf(q),ordered,'o',ms=3)
axes[1].plot([ordered.min(),ordered.max()],[ordered.min(),ordered.max()],'k--')
axes[1].set(xlabel='Gumbel quantiles [m]',ylabel='observed quantiles [m]')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Uncertainty and time-grid check
# 
# Bootstrap independent record maxima, not correlated time samples. Zero-pad Fourier coefficients to evaluate the same continuous record on a finer time grid.

# %%
boot_q99 = np.empty(200)
for i in range(boot_q99.size):
    loc_b, scale_b = stats.gumbel_r.fit(rng.choice(maxima,size=R,replace=True))
    boot_q99[i] = stats.gumbel_r.ppf(.99,loc=loc_b,scale=scale_b)
print('Gumbel q99 bootstrap 95% interval [m]:',np.quantile(boot_q99,[.025,.975]))
nt = history.size
fine_coefficients = np.zeros(nt+1,dtype=complex)
fine_coefficients[:coefficients.size] = 2*coefficients
fine_history = np.fft.irfft(fine_coefficients,n=2*nt)
assert np.allclose(fine_history[::2],history)
print('Same-record max at dt / dt/2 [m]:',history.max(),fine_history.max())
plt.hist(boot_q99,bins=25); plt.xlabel('fitted q99 [m]'); plt.ylabel('bootstrap count'); plt.show()
print('Exercise: double bandwidth and record duration separately. Which approximation changes most?')
print('One-record grid agreement does not establish convergence for every record or in the tail.')
