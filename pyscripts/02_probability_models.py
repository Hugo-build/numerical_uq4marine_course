# %% [markdown]
# # 02 · Probability models and basic statistics
# 
# **Goal:** connect PDF, CDF, survival probability, quantiles, sampling, and fitting.
# Models: normal, Rayleigh, maximum of $n$ Rayleigh variables (called “Rayleigh-n” here), Gumbel for maxima,
# Weibull, lognormal, and exponential. “Rayleigh-n” is ambiguous terminology; here it means an order statistic,
# not the chi distribution with $n$ degrees of freedom.
# 
# Sources: [SciPy statistics](https://docs.scipy.org/doc/scipy/reference/stats.html),
# [Rayleigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html),
# [Weibull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html),
# [GEV conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).

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
# ## Define probability models
# 
# `loc` shifts and `scale` rescales. A lognormal uses `s` as the standard deviation of the underlying normal log-variable; its scale is $e^{\mu_{\log}}$.

# %%
models = {'Normal': stats.norm(loc=0, scale=1),
          'Rayleigh': stats.rayleigh(scale=1),
          'Gumbel maxima': stats.gumbel_r(loc=1.5, scale=0.5),
          'Weibull 2 params': stats.weibull_min(c=2.5, scale=2),
          'Weibull 3 params': stats.weibull_min(c=2.5, loc=0.5, scale=2),
          'Lognormal': stats.lognorm(s=0.4, scale=np.exp(0.3)),
          'Exponential': stats.expon(scale=1.5)}
print(f"{'Model':18s} {'mean':>9s} {'std':>9s} {'median':>9s} {'q99':>9s}")
for name, rv in models.items():
    print(f'{name:18s} {rv.mean():9.3f} {rv.std():9.3f} {rv.median():9.3f} {rv.ppf(.99):9.3f}')

# %% [markdown]
# ## PDF, CDF, and survival
# 
# For continuous variables, probability is area under the PDF. For example, $P(a<X<b)=F(b)-F(a)$. Use `sf` for small exceedance probabilities.

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, (name, rv) in zip(axes.flat, models.items()):
    grid = np.linspace(rv.ppf(.001), rv.ppf(.999), 400)
    ax.plot(grid, rv.pdf(grid), label='PDF')
    ax.plot(grid, rv.cdf(grid), '--', label='CDF')
    ax.set(title=name, xlabel='x', ylabel='density / probability'); ax.legend()
plt.tight_layout(); plt.show()
normal = models['Normal']
print('P(-1 < X < 1):', normal.cdf(1)-normal.cdf(-1))
print('P(X > 5):', normal.sf(5), '99% quantile:', normal.ppf(.99))

# %% [markdown]
# ## Rayleigh-n: exact independent maximum
# 
# If $R_i$ are independent Rayleigh variables with scale $\sigma$, then $F_M(x)=[1-e^{-x^2/(2\sigma^2)}]^n$ for $x\geq0$. The PDF is $nf_R F_R^{n-1}$. This formula does not apply directly to correlated time samples.

# %%
sigma, n = 1., 50
rayleigh = stats.rayleigh(scale=sigma)
grid = np.linspace(0, 5, 500)
cdf_max = rayleigh.cdf(grid)**n
pdf_max = n*rayleigh.pdf(grid)*rayleigh.cdf(grid)**(n-1)
max_samples = rng.rayleigh(sigma, size=(10000, n)).max(axis=1)
q = np.array([.5, .9, .99])
# expm1 avoids cancellation in 1 - q**(1/n).
quantiles = sigma*np.sqrt(-2*np.log(-np.expm1(np.log(q)/n)))
print('Probabilities:', q, 'exact maximum quantiles:', quantiles)
print('Monte Carlo quantiles:', np.quantile(max_samples, q))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(max_samples, bins=50, density=True, alpha=.4, label='MC maxima')
axes[0].plot(grid, pdf_max, label='exact PDF'); axes[0].legend()
axes[1].plot(grid, cdf_max, label='maximum CDF')
axes[1].plot(grid, rayleigh.cdf(grid), '--', label='single Rayleigh CDF'); axes[1].legend()
for ax in axes: ax.set(xlabel='x', ylabel='density / probability')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Sampling and fitting
# 
# Fit data generated from a known Weibull model. Fixing `loc=0` is a physical support assumption. A fitted curve is not proof that the distribution is correct.

# %%
truth = models['Weibull 2 params']; sample = truth.rvs(size=2000, random_state=rng)
shape, location, scale = stats.weibull_min.fit(sample, floc=0)
fitted = stats.weibull_min(shape, loc=location, scale=scale)
print('True shape/scale: 2.5/2; fitted:', shape, scale)
print('Sample mean/std:', np.mean(sample), np.std(sample, ddof=1))
print('Sample covariance of two variables:\n', np.cov(np.vstack([sample, sample+rng.normal(0,.3,sample.size)])))
q = np.linspace(.01, .99, 99)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
grid = np.linspace(0, 5, 400)
axes[0].hist(sample, bins=40, density=True, alpha=.4)
axes[0].plot(grid, truth.pdf(grid), label='truth'); axes[0].plot(grid, fitted.pdf(grid), '--', label='fit')
axes[0].set(xlabel='x', ylabel='PDF'); axes[0].legend()
axes[1].plot(fitted.ppf(q), np.quantile(sample,q), 'o', ms=3)
limits = [0, max(fitted.ppf(q).max(), np.quantile(sample,q).max())]
axes[1].plot(limits, limits, 'k--'); axes[1].set(xlabel='fitted quantiles', ylabel='sample quantiles', title='Q–Q plot')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Bootstrap uncertainty and model identities
# 
# A confidence interval concerns estimation uncertainty. It is different from an interval covering future observations. Weibull shape 2 with scale $\sqrt{2}\sigma$ equals Rayleigh scale $\sigma$.

# %%
boot_means = np.mean(rng.choice(sample, size=(400, sample.size), replace=True), axis=1)
print('Mean estimate:', sample.mean(), 'bootstrap 95% interval:', np.quantile(boot_means,[.025,.975]))
error = np.max(np.abs(stats.weibull_min(2,scale=np.sqrt(2)).cdf(grid)-stats.rayleigh().cdf(grid)))
assert error < 1e-12
print('Weibull–Rayleigh CDF identity error:', error)
print('Exercise: change n to 5 and 500; compare how the maximum quantiles change.')
