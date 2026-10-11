# %% [markdown]
# # 03 · Static finite elements: 2D bars and beams
# 
# **Goal:** understand element stiffness, assembly, boundary conditions, solution, and verification.
# Small displacements, linear elastic material, SI units. Truss members carry axial force only.
# The beam is an Euler–Bernoulli bending element in the $x$–$y$ plane, with transverse displacement and rotation.
# This is a teaching model; buckling, geometric nonlinearity, shear deformation, and yielding are excluded.

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
# ## A bar stiffness matrix
# 
# For a 2-node axial bar, $k_e=(EA/L)[[1,-1],[-1,1]]$. An unconstrained bar has a rigid translation mode.

# %%
def axial_bar_stiffness(E, area, length):
    return E*area/length*np.array([[1., -1.], [-1., 1.]])

E, area, length = 210e9, 3e-4, 2.
ke = axial_bar_stiffness(E, area, length)
print('Bar stiffness [N/m]:\n', ke, 'eigenvalues:', np.linalg.eigvalsh(ke))
load = 10000.
elongation = load/ke[1,1]
print('Fixed-left bar elongation [mm]:', 1000*elongation, 'analytical PL/(EA):', load*length/(E*area))

# %% [markdown]
# ## Standard local stiffness and rotation for a 2D bar
# 
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
# ## Rotate and assemble a 2D truss
#
# Evaluate each member's local matrix, rotate it into global coordinates,
# then add it to the global DOFs [ux_i, uy_i, ux_j, uy_j].
# Solve only the free degrees of freedom; fixed displacements are zero.

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
# ## Solve and check the triangle
# 
# Node 0 is pinned; node 1 is a vertical roller. A vertical force acts on node 2. Check force balance and the residual on free DOFs.

# %%
nodes = np.array([[0.,0.],[2.,0.],[1.,1.5]])
elements = np.array([[0,1],[0,2],[1,2]])
areas = np.full(3, 3e-4); force = np.array([0.,0.,0.,0.,0.,-10000.])
fixed = np.array([0,1,3])
u, reaction, stress, K = solve_truss(nodes,elements,E,areas,force,fixed)
free = np.setdiff1d(np.arange(6),fixed)
print('Nodal displacements [mm]:\n', u.reshape(-1,2)*1000)
print('Support reactions [N]:', reaction[fixed], 'stress [MPa]:', stress/1e6)
balance = (reaction+force).reshape(-1,2).sum(axis=0)
assert np.linalg.norm(reaction[free]) < 1e-8
assert np.linalg.norm(balance) < 1e-8
assert np.all(np.linalg.eigvalsh(K[np.ix_(free,free)])>0)
print('Force balance [N]:', balance, 'free DOF residual:', np.linalg.norm(reaction[free]))
fig, ax = plt.subplots()
deformed = nodes + 300*u.reshape(-1,2)
for e,(i,j) in enumerate(elements):
    ax.plot(nodes[[i,j],0],nodes[[i,j],1], 'k--', label='undeformed' if e==0 else None)
    ax.plot(deformed[[i,j],0],deformed[[i,j],1], 'o-', label='deformed ×300' if e==0 else None)
for i,p in enumerate(nodes): ax.text(*p, f'  node {i}')
ax.set(xlabel='x [m]', ylabel='y [m]', title='2D truss'); ax.axis('equal'); ax.legend(); plt.show()

# %% [markdown]
# ## Euler–Bernoulli beam assembly
# 
# Each node has $(w,\theta)$; the element DOFs are $[w_i,\theta_i,w_j,\theta_j]$. A cantilever with tip force has $w(L)=PL^3/(3EI)$ and $\theta(L)=PL^2/(2EI)$.
# beam_stiffness returns the standard local bending matrix. This beam lies
# along global x, so local and global axes coincide and T is the identity.
# An inclined beam requires a full 2D frame element with axial DOFs as well
# as bending DOFs; this bending-only model does not support that rotation.

# %%
def beam_stiffness(EI, L):
    return EI/L**3*np.array([[12,6*L,-12,6*L],
        [6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],
        [6*L,2*L**2,-6*L,4*L**2]])

Lbeam, EI, P, ne = 3., 210e9*8e-6, -1000., 6
xb = np.linspace(0,Lbeam,ne+1); ndof = 2*(ne+1)
Kb = np.zeros((ndof,ndof))
for e in range(ne):
    dofs = np.arange(2*e,2*e+4)
    k_local = beam_stiffness(EI,xb[e+1]-xb[e])
    T = np.eye(4)  # Straight horizontal beam: local axes = global axes.
    k_global = T.T @ k_local @ T
    Kb[np.ix_(dofs,dofs)] += k_global
fb = np.zeros(ndof); fb[-2] = P
ub = np.zeros(ndof); ub[2:] = np.linalg.solve(Kb[2:,2:],fb[2:])
rb = Kb@ub-fb
exact_tip = P*Lbeam**3/(3*EI)
assert np.isclose(ub[-2], exact_tip, rtol=1e-9)
print('Beam tip FE / exact [mm]:', ub[-2]*1000, exact_tip*1000)
print('Tip rotation [rad]:', ub[-1], 'support shear/moment [N, N m]:', rb[:2])

# %% [markdown]
# ## Plot the beam with cubic interpolation
# 
# Hermite shape functions reconstruct displacement between nodes. Joining only nodal values with straight segments hides the beam curvature.

# %%
xplot, wplot = [], []
for e in range(ne):
    h = xb[e+1]-xb[e]; r = np.linspace(0,1,30)
    N = np.array([1-3*r**2+2*r**3, h*(r-2*r**2+r**3), 3*r**2-2*r**3, h*(-r**2+r**3)])
    xplot.extend(xb[e]+h*r); wplot.extend(ub[2*e:2*e+4]@N)
xplot = np.array(xplot); wplot = np.array(wplot)
exact = P*xplot**2*(3*Lbeam-xplot)/(6*EI)
print('Maximum interpolation error [m]:', np.max(np.abs(wplot-exact)))
plt.plot(xplot,1000*wplot,label='FE Hermite interpolation')
plt.plot(xplot,1000*exact,'--',label='analytical')
plt.scatter(xb,1000*ub[::2],label='nodes')
plt.xlabel('x [m]'); plt.ylabel('transverse displacement [mm]'); plt.legend(); plt.show()
print('Exercise: vary ne from 1 to 12. Why is this tip-load solution exact even with one element?')
