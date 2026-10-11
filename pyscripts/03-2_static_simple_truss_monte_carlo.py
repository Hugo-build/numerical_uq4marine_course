# %% [markdown]
# # 05 · Probability propagation through static finite elements
# 
# **Goal:** propagate uncertain load, elastic modulus, and member area through the 2D truss.
# The model is linear in displacement but the input-to-response map is nonlinear when stiffness is uncertain.
# Use independent, positive lognormal inputs. This exercise studies a displacement serviceability limit;
# it does not represent a full structural safety assessment.

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
# ## Standard local stiffness and rotation for a 2D bar
# 
# This notebook includes its own truss functions; it does not depend on running another notebook.
# Local x follows the member; local y is perpendicular. The local DOFs
# are [u_i, v_i, u_j, v_j]. Only the axial translations have stiffness.
# R=[[c,s],[-s,c]] maps global translations to local translations.
# With T=diag(R,R), u_local=T u_global and k_global=T^T k_local T.

# %%
def local_bar_stiffness(E, area, length):
    assert length > 0, 'A member must connect distinct positions.'
    return E*area/length*np.array([
        [ 1., 0., -1., 0.],
        [ 0., 0.,  0., 0.],
        [-1., 0.,  1., 0.],
        [ 0., 0.,  0., 0.],
    ])

def bar_transformation(point_i, point_j):
    delta = point_j-point_i
    length = np.linalg.norm(delta)
    assert length > 0, 'A member must connect distinct positions.'
    c, s = delta/length
    R = np.array([[c, s], [-s, c]])
    T = np.zeros((4, 4))
    T[:2, :2] = R
    T[2:, 2:] = R
    return length, T

print('Standard local 4×4 bar stiffness and global-to-local rotation defined.')

# %% [markdown]
# ## Define the deterministic FE map
#
# Evaluate each member's local matrix, rotate it into global coordinates,
# then add it to the global DOFs [ux_i, uy_i, ux_j, uy_j]. Each Monte Carlo
# realization uses these same functions with its sampled modulus and area.

# %%
def assemble_truss(nodes, elements, E, areas):
    # Two translations per node; axial-only members.
    K = np.zeros((2*len(nodes), 2*len(nodes)))
    for (i, j), area in zip(elements, areas):
        length, T = bar_transformation(nodes[i], nodes[j])
        k_local = local_bar_stiffness(E, area, length)
        k_global = T.T @ k_local @ T
        dofs = np.array([2*i, 2*i+1, 2*j, 2*j+1])
        K[np.ix_(dofs,dofs)] += k_global
    return K

def solve_truss(nodes, elements, E, areas, force, fixed):
    K = assemble_truss(nodes, elements, E, areas)
    free = np.setdiff1d(np.arange(len(force)), fixed)
    u = np.zeros_like(force, dtype=float)
    u[free] = np.linalg.solve(K[np.ix_(free,free)], force[free])
    reaction = K@u-force
    stress = []
    for i,j in elements:
        delta = nodes[j]-nodes[i]; L = np.linalg.norm(delta)
        strain = (u[2*j:2*j+2]-u[2*i:2*i+2])@(delta/L)/L
        stress.append(E*strain)
    return u, reaction, np.array(stress), K
print('Truss assembly and solution functions defined; two displacement DOFs per node.')

# %% [markdown]
# ## Input distributions
# 
# Convert physical mean $\mu$ and coefficient of variation $v$ into lognormal parameters: $s=\sqrt{\ln(1+v^2)}$, scale $=\mu/\sqrt{1+v^2}$. One uncertain area applies to all three members, so member areas are perfectly correlated.

# %%
nodes = np.array([[0.,0.],[2.,0.],[1.,1.5]])
elements = np.array([[0,1],[0,2],[1,2]])
fixed = np.array([0,1,3])
def positive_model(mean, cov):
    s = np.sqrt(np.log1p(cov**2))
    return stats.lognorm(s=s,scale=mean/np.sqrt(1+cov**2))
models = [positive_model(10000.,.2),positive_model(210e9,.05),positive_model(3e-4,.1)]
print('Input names: load P [N], modulus E [Pa], common area A [m²]')
for name, rv in zip(['P','E','A'],models): print(name,'mean:',rv.mean(),'CoV:',rv.std()/rv.mean())
force = np.array([0.,0.,0.,0.,0.,-10000.])
u0,_,_,_ = solve_truss(nodes,elements,210e9,np.full(3,3e-4),force,fixed)
baseline = -u0[-1]
print('Baseline downward apex displacement [mm]:',baseline*1000)

# %% [markdown]
# ### Probability density and cumulative distribution of the inputs
#
# The PDF $f_X(x)$ describes probability density; probability over an interval
# is the area under its curve. The CDF $F_X(x)=\Pr(X\leq x)$ gives the
# probability of being at or below a value. All three inputs are positive
# lognormal variables. The horizontal axes use kN, GPa, and mm² for readability.
# If the plotted variable is $Y=kX$, its density is $f_Y(y)=f_X(y/k)/k$;
# its CDF is $F_Y(y)=F_X(y/k)$. The curves end at the 99.9th percentile,
# so a small upper tail is outside the displayed range.

# %%
input_labels = ['Load P [kN]', 'Modulus E [GPa]', 'Common area A [mm²]']
input_factors = [1e-3, 1e-9, 1e6]  # Convert SI values into the displayed units.
input_density_units = ['1/kN', '1/GPa', '1/mm²']
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j, (rv, label, factor, density_unit) in enumerate(
        zip(models, input_labels, input_factors, input_density_units)):
    input_grid = np.linspace(0., rv.ppf(.999), 500)
    axes[0, j].plot(input_grid*factor, rv.pdf(input_grid)/factor)
    axes[0, j].set(xlabel=label, ylabel=f'Density [{density_unit}]', title='Input PDF')
    axes[1, j].plot(input_grid*factor, rv.cdf(input_grid))
    axes[1, j].set(xlabel=label, ylabel='Cumulative probability', ylim=(0, 1), title='Input CDF')
fig.suptitle('Independent lognormal random inputs')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## One solved case: original and deformed truss
#
# Plot the baseline solution at the mean inputs: $P=10$ kN, $E=210$ GPa,
# and $A=300$ mm² for every member. This is one deterministic case, not the
# mean deformed shape from Monte Carlo. Reshape the six displacement DOFs
# into three $(u_x,u_y)$ pairs, then plot $x_{deformed}=x+500u$.
# Displacements are magnified 500 times to make the deformation visible;
# the reported apex displacement is the actual, unscaled value.
# The left support fixes both translations; the right support fixes only
# vertical translation, allowing horizontal movement.

# %%
deformation_scale = 500.
deformed_nodes = nodes + deformation_scale*u0.reshape(-1, 2)
fig, ax = plt.subplots(figsize=(8, 5))
for member, (i, j) in enumerate(elements):
    ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], 'o--', color='0.5',
            label='Original' if member == 0 else None)
    ax.plot(deformed_nodes[[i, j], 0], deformed_nodes[[i, j], 1], 'o-',
            color='tab:blue', label=f'Deformed ({deformation_scale:g}×)' if member == 0 else None)
ax.scatter(nodes[0, 0], nodes[0, 1], marker='^', s=150, color='black',
           zorder=3, label='Pinned support')
ax.scatter(nodes[1, 0], nodes[1, 1], marker='o', s=100, facecolors='white',
           edgecolors='black', zorder=3, label='Roller support')
ax.annotate('P = 10 kN', xy=nodes[2], xytext=(1., 1.95), ha='center',
            arrowprops={'arrowstyle': '->', 'color': 'tab:red'}, color='tab:red')
ax.text(0.03, 0.97, f'Actual downward apex displacement:\n{baseline*1000:.5f} mm',
        transform=ax.transAxes, va='top', fontsize=10)
ax.set(xlabel='x [m]', ylabel='y [m]', xlim=(-0.3, 2.4), ylim=(-0.3, 2.3),
       title='Baseline FE solution: displacement magnified 500×')
ax.set_aspect('equal')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Monte Carlo with actual FE solves
# 
# Each row of the sample table is one physical realization. Start with 500 samples when teaching; the default uses 5,000 small solves.

# %%
N = 5000
samples = np.column_stack([rv.rvs(size=N,random_state=rng) for rv in models])
displacement = np.empty(N)
for i,(P,E,A) in enumerate(samples):
    force = np.zeros(6); force[-1] = -P
    u,_,_,_ = solve_truss(nodes,elements,E,np.full(3,A),force,fixed)
    displacement[i] = -u[-1]
print('Completed FE realizations:',N)
print('Displacement mean/std [mm]:',displacement.mean()*1000,displacement.std(ddof=1)*1000)
print('Displacement quantiles [mm]:',np.quantile(displacement,[.5,.9,.99])*1000)
fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].hist(displacement*1000,bins=50,density=True); axes[0].set(xlabel='downward displacement [mm]',ylabel='density [1/mm]')
axes[1].scatter(samples[:,0]/1000,displacement*1000,s=4,alpha=.3)
axes[1].set(xlabel='load [kN]',ylabel='downward displacement [mm]')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Analytical benchmark for this special model
# 
# ### 1. Factor the FE stiffness and load
#
# Each member contributes $K_e=(EA/L_e)B_e^T B_e$, where
# $B_e=[-c,-s,c,s]$ contains its direction cosines. The geometry is fixed,
# and every member uses the same modulus $E$ and common area $A$.
# Therefore $K=EA\,\bar K$, where $\bar K$ depends only on geometry.
# The applied force is $f=P\bar f$, with a downward unit load at the apex.
# With zero prescribed support displacements, the free DOFs satisfy
#
# $$EA\,\bar K_{ff}u_f=P\bar f_f
# \quad\Longrightarrow\quad
# u_f=\frac{P}{EA}\bar K_{ff}^{-1}\bar f_f.$$
#
# The downward apex displacement is $d=-u_{\mathrm{apex},y}$, so
#
# $$\boxed{d=C\frac{P}{EA}}.$$
#
# The geometry constant $C$ has units of length. The code below obtains it
# from the baseline FE result: $C=d_0E_0A_0/P_0$, stored as constant.
# Only one deterministic solve is needed to find $C$.
#
# ### 2. Derive C directly from truss mechanics
#
# Let $a=1$ m be the half-span, $h=1.5$ m the height, and
# $L=\sqrt{a^2+h^2}$ the length of each diagonal. The bottom member has
# length $2a$. Equilibrium gives compression of magnitude $PL/(2h)$
# in each diagonal and tension $Pa/(2h)$ in the bottom member.
#
# The axial strain energy and Castigliano's theorem give
#
# $$U=\sum_e\frac{N_e^2L_e}{2EA}
# =\frac{P^2(L^3+a^3)}{4EA h^2},
# \qquad d=\frac{\partial U}{\partial P}
# =\frac{P}{EA}\frac{L^3+a^3}{2h^2}.$$
#
# Thus $C=(L^3+a^3)/(2h^2)\approx1.52423$ m, agreeing with the FE-derived
# constant. At the mean inputs, $d_0\approx0.24194$ mm. This baseline
# response is not generally the mean of the uncertain displacement.
#
# ### 3. Propagate the independent lognormal inputs
#
# For each input $X$ with physical mean $\mu_X$ and coefficient of variation
# $v_X$, positive_model uses
#
# $$s_X^2=\ln(1+v_X^2),\qquad
# m_X=\ln\mu_X-\frac{s_X^2}{2},\qquad
# \ln X\sim\mathcal N(m_X,s_X^2).$$
#
# SciPy's scale parameter is $\exp(m_X)$, the median, rather than the mean.
# Logarithms here refer to numerical values expressed in the stated SI units.
# Taking the logarithm of the response formula gives
#
# $$\ln d=\ln C+\ln P-\ln E-\ln A.$$
#
# A linear combination of independent normal variables is normal. Hence
# the displacement is exactly lognormal with
#
# $$m_d=\ln C+m_P-m_E-m_A,\qquad
# s_d^2=s_P^2+s_E^2+s_A^2.$$
#
# Variances add even for the negative terms because their coefficients are
# squared. Independence removes covariance terms. The code stores $m_d$
# as mu_log and $s_d$ as sigma_log. The exact distribution is constructed
# with stats.lognorm(s=sigma_log, scale=np.exp(mu_log)).
#
# Its population mean and standard deviation are
#
# $$\mathbb E[d]=\exp(m_d+s_d^2/2),\qquad
# \operatorname{SD}(d)=\mathbb E[d]\sqrt{\exp(s_d^2)-1}.$$
#
# For these inputs they are approximately 0.24497 mm and 0.05641 mm.
# Monte Carlo estimates fluctuate around these population values.
#
# ### 4. Compare the analytical map with Monte Carlo FE solves
#
# The direct array evaluates $CP/(EA)$ for the same samples as the FE loop.
# The relative_error assertion checks sample-by-sample agreement up to
# floating-point error. The histogram estimates the density given by exact.pdf.
#
# This shortcut relies on fixed geometry, linear elastic behavior, the same
# scalar $E$ and $A$ for every member, and a fixed load pattern scaled by $P$.
# Independent member areas generally prevent factoring $K$ as $EA\bar K$.
# General FE uncertainty does not usually produce an exactly lognormal response.

# %%
constant = baseline*210e9*3e-4/10000.
mu_log = np.log(constant)+np.log(models[0].kwds['scale'])-np.log(models[1].kwds['scale'])-np.log(models[2].kwds['scale'])
sigma_log = np.sqrt(sum(rv.kwds['s']**2 for rv in models))
exact = stats.lognorm(s=sigma_log,scale=np.exp(mu_log))
direct = constant*samples[:,0]/(samples[:,1]*samples[:,2])
relative_error = np.max(np.abs(displacement-direct)/direct)
assert relative_error<1e-10
print('Maximum FE/scaling relative error:',relative_error)
print('Exact mean/std [mm]:',exact.mean()*1000,exact.std()*1000)

# %% [markdown]
# ### Displacement PDF and CDF: analytical solution versus Monte Carlo
#
# The density histogram estimates the analytical lognormal PDF. The empirical
# CDF is $\widehat F_N(d)=N^{-1}\sum_{i=1}^N\mathbf{1}(d_i\leq d)$:
# sort the sampled displacements and increase the cumulative probability by
# $1/N$ at each sample. The smooth analytical CDF is
# $F_d(d)=\Phi((\ln d-m_d)/s_d)$ for $d>0$, with $F_d(0)=0$.
# At a displacement limit, $1-F_d(d_{lim})$ is the exceedance probability.
# The PDF is converted from density per metre to density per millimetre;
# cumulative probabilities are dimensionless and need no vertical scaling.

# %%
grid_end = max(exact.ppf(.999), displacement.max()*1.01)
grid = np.linspace(0., grid_end, 500)
sorted_displacement = np.sort(displacement)
empirical_cdf = np.arange(1, N+1)/N

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(displacement*1000, bins=50, density=True, alpha=.4, label='FE Monte Carlo')
axes[0].plot(grid*1000, exact.pdf(grid)/1000, label='Analytical lognormal')
axes[0].set(xlabel='Downward displacement [mm]', ylabel='Density [1/mm]', title='Displacement PDF')
axes[1].step(np.r_[0., sorted_displacement, grid_end]*1000,
             np.r_[0., empirical_cdf, 1.], where='post', label='FE Monte Carlo ECDF')
axes[1].plot(grid*1000, exact.cdf(grid), '--', label='Analytical lognormal')
axes[1].set(xlabel='Downward displacement [mm]', ylabel='Cumulative probability',
            ylim=(0, 1), title='Displacement CDF')
for ax in axes:
    ax.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Exceedance and Monte Carlo convergence
# 
# Define $g=d_{lim}-d$. Failure here means $g<0$. Report a Wilson binomial confidence interval, rather than treating the Monte Carlo estimate as exact.
#
# For the analytical distribution, the exceedance probability is
#
# $$p_f=\Pr(d>d_{\mathrm{lim}})
# =1-\Phi\!\left(\frac{\ln d_{\mathrm{lim}}-m_d}{s_d}\right),$$
#
# where $\Phi$ is the standard normal CDF. The call exact.sf(limit) evaluates
# this upper-tail probability directly. At $d_{\mathrm{lim}}=1.3d_0$,
# it is approximately 0.11254 (11.25%). The Monte Carlo estimate is the
# fraction of samples exceeding the limit; sampling uncertainty decreases
# as the sample count $N$ grows.

# %%
limit = 1.3*baseline
failure = displacement>limit; failures = int(failure.sum()); pf = failure.mean()
z = stats.norm.ppf(.975); denom = 1+z*z/N
center = (pf+z*z/(2*N))/denom
half = z*np.sqrt(pf*(1-pf)/N+z*z/(4*N*N))/denom

print('Limit [mm]:',limit*1000, 'exceedances:',failures,'of',N)
print('Estimated probability:',pf,'95% Wilson interval:',(center-half,center+half))
print('Exact probability:',exact.sf(limit),'mean standard error [mm]:',displacement.std(ddof=1)/np.sqrt(N)*1000)

counts = np.unique(np.geomspace(50,N,30).astype(int))
estimates = np.cumsum(failure)[counts-1]/counts
plt.semilogx(counts,estimates,'o-',label='MC exceedance probability')
plt.axhline(exact.sf(limit),color='k',ls='--',label='exact')
plt.xlabel('number of samples'); plt.ylabel('probability'); plt.legend(); plt.show()
print('Exercise: give each member an independent area. Which analytical shortcut stops working?')
