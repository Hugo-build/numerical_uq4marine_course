# %% [markdown]
# # 07 · Long-term response mixtures and annual extremes
# 
# **Goal:** distinguish a long-term distribution of instantaneous response, a mixture of block maxima,
# and an annual maximum distribution. Four artificial environmental states have time-occurrence probabilities $w_j$.
# They change the Gaussian force spectrum seen by the same linear oscillator. These states are examples, not site data.
# 
# All blocks last $T=600$ seconds. The annual model assumes independent blocks and independently selected states.
# Real weather persistence can violate both assumptions. Tail extrapolation from a few hundred simulated records
# is demonstrated for teaching and is not validated as a design return level.

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
# ## Set up the linear spectral model
# 
# This notebook repeats the spectral functions so it runs independently.

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
# ## Environmental states and conditional moments
# 
# Weights must sum to one. Time-occurrence weights also weight blocks here only because every block has equal duration.

# %%
T, dt = 600., .1
weights = np.array([.45,.30,.20,.05])
force_rms = np.array([40.,75.,120.,180.])
peak_hz = np.array([.08,.10,.12,.15])
assert np.isclose(weights.sum(),1)
spectra = []; sigmas = []
for j in range(4):
    f,Sf,Sx,H = response_spectrum(force_rms[j],peak_hz[j],.018,T,dt)
    spectra.append(Sx); sigmas.append(np.sqrt(Sx.sum()/T))
sigmas = np.array(sigmas)
print('state | time weight | force RMS [N] | peak [Hz] | response RMS [m]')
for j in range(4): print(j,weights[j],force_rms[j],peak_hz[j],sigmas[j])
plt.bar(np.arange(4),weights); plt.xlabel('state'); plt.ylabel('time-occurrence probability'); plt.show()

# %% [markdown]
# ## Long-term instantaneous distribution
# 
# The mixture is $F_X(x)=\sum_jw_jF_{X|j}(x)$. A mixture of centered Gaussians is generally not Gaussian, even though its variance is $\sum_jw_j\sigma_j^2$.

# %%
grid_x = np.linspace(-5*sigmas.max(),5*sigmas.max(),600)
conditional_pdf = np.array([stats.norm(scale=s).pdf(grid_x) for s in sigmas])
pdf_long = weights@conditional_pdf
variance_long = weights@(sigmas**2)
state_draws = rng.choice(4,size=50000,p=weights)
instantaneous = rng.normal(size=state_draws.size)*sigmas[state_draws]
print('Long-term standard deviation theory / MC [m]:',np.sqrt(variance_long),instantaneous.std(ddof=1))
plt.hist(instantaneous,bins=100,density=True,alpha=.3,label='mixture samples')
plt.plot(grid_x,pdf_long,label='Gaussian mixture')
plt.plot(grid_x,stats.norm(scale=np.sqrt(variance_long)).pdf(grid_x),'--',label='same-variance Gaussian')
plt.xlabel('instantaneous displacement [m]'); plt.ylabel('density [1/m]'); plt.legend(); plt.show()

# %% [markdown]
# ## Conditional block maxima and mixture
# 
# Generate independent records in each state. Fit one Gumbel distribution per state, then mix CDFs. A single Gumbel fit to the pooled maxima need not represent the mixture.

# %%
R = 250
maxima = np.empty((4,R)); fitted = []
for j,Sx in enumerate(spectra):
    for i in range(R):
        history,_ = gaussian_history(Sx,T,dt,rng)
        maxima[j,i] = history.max()
    loc,scale = stats.gumbel_r.fit(maxima[j])
    fitted.append(stats.gumbel_r(loc=loc,scale=scale))
    print('State',j,'Gumbel loc/scale:',loc,scale,'empirical q90:',np.quantile(maxima[j],.9))
def block_cdf(x):
    return sum(w*rv.cdf(x) for w,rv in zip(weights,fitted))
def block_sf(x):
    return sum(w*rv.sf(x) for w,rv in zip(weights,fitted))
grid_m = np.linspace(0,maxima.max()*1.4,700)
plt.plot(grid_m,block_cdf(grid_m),'k',lw=2,label='mixture block CDF')
for j,rv in enumerate(fitted): plt.plot(grid_m,rv.cdf(grid_m),label=f'state {j}')
plt.xlabel('600 s positive maximum [m]'); plt.ylabel('CDF'); plt.legend(); plt.show()
q90_block = optimize.brentq(lambda x:block_cdf(x)-.9,0,grid_m[-1])
print('Mixture block q90 [m]:',q90_block)

# %% [markdown]
# ## Annual extremes and return levels
# 
# For $B$ independent identically distributed blocks in a year, $F_A(x)=F_B(x)^B$. A return level with return period $R_y$ satisfies $P(A>x)=1/R_y$. Use survival probabilities and `log1p` to avoid cancellation.

# %%
seconds_per_year = 365*24*3600
B = seconds_per_year/T
assert B == int(B)
def annual_cdf(x):
    sf = np.clip(block_sf(x),0,1-np.finfo(float).eps)
    return np.exp(B*np.log1p(-sf))
def return_level(years):
    target_block_sf = -np.expm1(np.log1p(-1/years)/B)
    lo, hi = 0., max(rv.ppf(.999999999999) for rv in fitted)
    return optimize.brentq(lambda x:block_sf(x)-target_block_sf,lo,hi)
periods = np.array([2.,5.,10.,20.,50.,100.])
levels = np.array([return_level(y) for y in periods])
print('Blocks/year:',int(B))
for years,level in zip(periods,levels): print('Return period [years]:',years,'return level [m]:',level)
assert np.all(np.diff(levels)>0)
assert np.isclose(annual_cdf(return_level(50)),.98)
fig, axes = plt.subplots(1,2,figsize=(11,4))
grid_a = np.linspace(fitted[-1].ppf(.99),levels[-1]*1.1,600)
axes[0].plot(grid_a,annual_cdf(grid_a)); axes[0].set(xlabel='annual positive maximum [m]',ylabel='CDF')
axes[1].semilogx(periods,levels,'o-'); axes[1].set(xlabel='return period [years]',ylabel='return level [m]')
plt.tight_layout(); plt.show()
print('These annual levels extrapolate the fitted tails far beyond the simulated quantiles.')

# %% [markdown]
# ## Check the annual CDF by inverse sampling
# 
# Sample annual maxima directly by transforming a uniform variable: $F_B(A)=U^{1/B}$. Solve through the survival function for numerical stability. This checks the aggregation calculation, not the fitted tail model.

# %%
annual_samples = np.empty(600)
uniform = rng.uniform(size=annual_samples.size)
hi = 2*levels[-1]
for i,U in enumerate(uniform):
    target_sf = -np.expm1(np.log(U)/B)
    annual_samples[i] = optimize.brentq(lambda x:block_sf(x)-target_sf,0,hi)
ordered = np.sort(annual_samples)
plt.step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',label='inverse samples')
plt.plot(grid_a,annual_cdf(grid_a),'--',label='annual formula')
plt.xlabel('annual positive maximum [m]'); plt.ylabel('CDF'); plt.legend(); plt.show()
print('Annual median sample / formula [m]:',np.median(annual_samples),return_level(2))
print('Exercise: increase rare-state weight, renormalize weights, then rerun subsequent cells.')
print('Extension: weather persistence needs a state-sequence model and dependence-aware aggregation.')
