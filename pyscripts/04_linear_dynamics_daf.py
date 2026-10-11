# %% [markdown]
# # 04 · Linear dynamics and dynamic amplification
# 
# **Goal:** solve a forced oscillator, identify resonance, and verify time integration against a frequency-domain result.
# For $m\ddot x+c\dot x+kx=F_0\sin(\omega t)$, the steady displacement amplitude is
# $(F_0/k)D(r)$ with $D(r)=[(1-r^2)^2+(2\zeta r)^2]^{-1/2}$ and $r=\omega/\omega_n$.
# DAF here is the steady harmonic **displacement** amplification relative to static deflection.

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
# ## Parameters and natural frequency
# 
# Use SI units. Damping ratio $\zeta=c/(2m\omega_n)$ is dimensionless.

# %%
m, k, zeta, F0 = 1000., 10000., .05, 100.
wn = np.sqrt(k/m); fn = wn/(2*np.pi); c = 2*zeta*m*wn
static_deflection = F0/k
print('Natural frequency [rad/s, Hz]:', wn, fn)
print('Damping [N s/m]:', c, 'static displacement [m]:', static_deflection)

# %% [markdown]
# ## Plot DAF and phase
# 
# The displacement lags the force by $\phi=\operatorname{atan2}(2\zeta r,1-r^2)$. For nonzero damping the displacement resonance occurs slightly below $r=1$.

# %%
r = np.linspace(0,3,1200)
fig, axes = plt.subplots(1,2,figsize=(10,4))
for damping in [.02,.05,.1,.2]:
    daf = 1/np.sqrt((1-r*r)**2+(2*damping*r)**2)
    axes[0].plot(r,daf,label=f'ζ={damping}')
    axes[1].plot(r,np.degrees(np.arctan2(2*damping*r,1-r*r)),label=f'ζ={damping}')
axes[0].set(xlabel='frequency ratio r',ylabel='displacement DAF')
axes[1].set(xlabel='frequency ratio r',ylabel='phase lag [degrees]')
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()
print('DAF at r=1:', 1/(2*zeta), 'resonance ratio:', np.sqrt(1-2*zeta*zeta))

# %% [markdown]
# ## Time-domain solution
# 
# Integrate from rest with an adaptive solver. The initial transient must decay before estimating steady-state amplitude.

# %%
omega = .95*wn
def rhs(t, y):
    x, v = y
    return [v, (F0*np.sin(omega*t)-c*v-k*x)/m]
t = np.linspace(0,120,12001)
solution = integrate.solve_ivp(rhs,[t[0],t[-1]],[0.,0.],t_eval=t,rtol=1e-8,atol=1e-10)
assert solution.success
x, v = solution.y
print('Solver message:', solution.message, 'RHS evaluations:', solution.nfev)
fig, axes = plt.subplots(2,1,figsize=(9,6),sharex=True)
axes[0].plot(t,F0*np.sin(omega*t)); axes[0].set(ylabel='force [N]')
axes[1].plot(t,x); axes[1].set(xlabel='time [s]',ylabel='displacement [m]')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Verify steady amplitude and phase
# 
# Fit sine and cosine components over the final 30 seconds. This avoids requiring sample times to land exactly on the response peaks.

# %%
mask = t>90
design = np.column_stack([np.sin(omega*t[mask]),np.cos(omega*t[mask]),np.ones(mask.sum())])
coef = np.linalg.lstsq(design,x[mask],rcond=None)[0]
amplitude = np.hypot(coef[0],coef[1]); phase = np.arctan2(-coef[1],coef[0])
ratio = omega/wn
theory = static_deflection/np.sqrt((1-ratio**2)**2+(2*zeta*ratio)**2)
print('Computed/theoretical amplitude [m]:', amplitude,theory)
print('Computed/theoretical phase [rad]:', phase,np.arctan2(2*zeta*ratio,1-ratio**2))
assert abs(amplitude/theory-1)<1e-4
plt.plot(t[mask],x[mask],label='time solution')
plt.plot(t[mask],theory*np.sin(omega*t[mask]-phase),'--',label='steady harmonic solution')
plt.xlabel('time [s]'); plt.ylabel('displacement [m]'); plt.legend(); plt.show()

# %% [markdown]
# ## Free decay and energy
# 
# With no forcing, mechanical energy decreases because $\dot E=-c\dot x^2$.

# %%
decay = integrate.solve_ivp(lambda t,y: [y[1],(-c*y[1]-k*y[0])/m],
                          [0,40],[.1,0],t_eval=np.linspace(0,40,4001),rtol=1e-8,atol=1e-10)
energy = .5*k*decay.y[0]**2+.5*m*decay.y[1]**2
assert energy[-1]<energy[0]
print('Initial/final energy [J]:',energy[0],energy[-1])
plt.plot(decay.t,energy); plt.xlabel('time [s]'); plt.ylabel('mechanical energy [J]'); plt.show()
print('Exercise: double damping, then compare DAF and the time needed for transients to decay.')
