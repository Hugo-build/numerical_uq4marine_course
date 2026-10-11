# %% [markdown]
# # 03 · Space truss: static response, modes, and time integration
#
# **Goal:** extend the 2D truss lesson to three translations per node,
# assemble stiffness and consistent mass, and compare Newmark-beta with RK4.
# Small displacements, linear elasticity, axial-only members, and SI units.
# The CSV coordinates are in mm; they are converted to metres on loading.
# Material and section values below are teaching assumptions, not measured
# properties of the laboratory structure shown in the reference image.
#
# This companion to testFE40barTruss_v2.py keeps all model equations visible.
# The supplied 40-member geometry has mechanisms as an axial-only truss.
# We first identify them, then explicitly add eight braces for the solves.
# A frame model with bending stiffness would be a different physical model.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

np.set_printoptions(precision=5, suppress=True)
plt.rcParams.update({'figure.figsize': (8, 4), 'axes.grid': True,
                     'grid.alpha': 0.25, 'font.size': 11})
print('Imports ready. Units: m, N, kg, s; plots also report displacement in mm.')

# %% [markdown]
# ## Load geometry and map node labels to array indices
#
# CSV node labels start at 1. Array indices start at 0; each node has
# DOFs [ux, uy, uz]. Mapping labels explicitly avoids assuming row order.
# Run the file from any directory, or run its cells from the course root
# or scripts folder in an interactive editor.

# %%
if '__file__' in globals():
    data_folder = Path(__file__).resolve().parent / '40barTruss'
else:
    data_folder = Path('scripts/40barTruss')
    if not data_folder.is_dir():
        data_folder = Path('40barTruss')

node_table = np.genfromtxt(data_folder / 'nodes.csv', delimiter=',', names=True)
element_table = np.genfromtxt(data_folder / 'elements.csv', delimiter=',', names=True)
node_ids = node_table['node_id'].astype(int)
node_index = {label: index for index, label in enumerate(node_ids)}
nodes = np.column_stack([node_table[key] for key in ['x', 'y', 'z']])*1e-3
original_elements = np.array([
    [node_index[int(row['node_i'])], node_index[int(row['node_j'])]]
    for row in element_table
])
ndof = 3*len(nodes)

def node_dofs(label):
    return 3*node_index[label] + np.arange(3)

fixed = np.concatenate([node_dofs(label) for label in [1, 2, 3, 4]])
free = np.setdiff1d(np.arange(ndof), fixed)
print('Nodes / original members / free DOFs:', len(nodes), len(original_elements), len(free))
print('Coordinate bounds [m]:', nodes.min(axis=0), nodes.max(axis=0))

# %% [markdown]
# ## Standard element matrices in local coordinates
#
# Local x points along the member. The DOF order is
# [u1, v1, w1, u2, v2, w2]. Only u1 and u2 contribute to axial stiffness;
# local transverse translations carry inertia but have no bar stiffness.
# The consistent translational mass includes all three directions.

# %%
def local_bar_matrices(E, area, rho, length):
    assert length > 0, 'A member must connect distinct positions.'
    k_local = E*area/length*np.array([
        [ 1., 0., 0., -1., 0., 0.],
        [ 0., 0., 0.,  0., 0., 0.],
        [ 0., 0., 0.,  0., 0., 0.],
        [-1., 0., 0.,  1., 0., 0.],
        [ 0., 0., 0.,  0., 0., 0.],
        [ 0., 0., 0.,  0., 0., 0.],
    ])
    m_local = rho*area*length/6*np.array([
        [2., 0., 0., 1., 0., 0.],
        [0., 2., 0., 0., 1., 0.],
        [0., 0., 2., 0., 0., 1.],
        [1., 0., 0., 2., 0., 0.],
        [0., 1., 0., 0., 2., 0.],
        [0., 0., 1., 0., 0., 2.],
    ])
    return k_local, m_local

print('Standard local 6×6 bar stiffness and consistent mass matrices defined.')

# %% [markdown]
# ## Rotation from global to local coordinates
#
# The rows of R are the local unit axes expressed in global coordinates.
# Choose a reference axis least parallel to local x, then use cross products
# to construct a right-handed orthonormal basis, including for vertical bars.
# T=diag(R, R) gives u_local=T u_global. Therefore
# k_global=T^T k_local T and m_global=T^T m_local T.
# The isotropic translational mass retains its form after rotation.

# %%
def bar_transformation(point_i, point_j):
    delta = point_j-point_i
    length = np.linalg.norm(delta)
    assert length > 0, 'A member must connect distinct positions.'
    ex = delta/length
    reference = np.eye(3)[np.argmin(np.abs(ex))]
    ey = np.cross(reference, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.vstack([ex, ey, ez])
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, 3:] = R
    return length, T

print('Member rotation defined: local displacement = T @ global displacement.')

# %% [markdown]
# ## Rotate each element, then assemble
#
# Obtain length and rotation from the two node positions, evaluate the
# predefined local matrices, rotate both matrices, and add their entries
# to the global DOFs [ux_i, uy_i, uz_i, ux_j, uy_j, uz_j].

# %%
def assemble_truss(nodes, elements, E, area, rho):
    K = np.zeros((3*len(nodes), 3*len(nodes)))
    M = np.zeros_like(K)
    for i, j in elements:
        length, T = bar_transformation(nodes[i], nodes[j])
        k_local, m_local = local_bar_matrices(E, area, rho, length)
        ke = T.T @ k_local @ T
        me = T.T @ m_local @ T
        dofs = np.r_[3*i+np.arange(3), 3*j+np.arange(3)]
        K[np.ix_(dofs, dofs)] += ke
        M[np.ix_(dofs, dofs)] += me
    return K, M

E, area, rho = 210e9, 1e-4, 7850.
K_original, M_original = assemble_truss(nodes, original_elements, E, area, rho)
K_original_ff = K_original[np.ix_(free, free)]
rank = np.linalg.matrix_rank(K_original_ff)
print('Young modulus [GPa] / area [mm²] / density [kg/m³]:', E/1e9, area*1e6, rho)
print('Original free stiffness rank:', rank, 'of', len(free))
print('Number of mechanisms:', len(free)-rank)

# %% [markdown]
# ## Make the teaching truss stable
#
# Add one diagonal on each of the two y–z side faces at each storey.
# These eight members supplement the original x–z side bracing. Orange
# lines identify the added members. Results below apply to this 48-member
# teaching extension, rather than the unmodified laboratory geometry.
# Setting added_braces to an empty list makes the stability check fail.

# %%
added_braces = []
for level in range(4):
    base = 4*level
    added_braces.extend([(base+1, base+8), (base+2, base+7)])  # CSV labels
braces = np.array([[node_index[i], node_index[j]] for i, j in added_braces],
                  dtype=int).reshape(-1, 2)
elements = np.vstack([original_elements, braces])


K, M = assemble_truss(nodes, elements, E, area, rho)
K_ff = K[np.ix_(free, free)]
M_ff = M[np.ix_(free, free)]
assert np.linalg.matrix_rank(K_ff) == len(free), 'The supported truss has mechanisms.'
assert np.allclose(K, K.T) and np.allclose(M, M.T)
linalg.cholesky(K_ff)
linalg.cholesky(M_ff)
print('Members after bracing:', len(elements), 'free stiffness rank:', len(free))

def plot_structure(displacement=None, scale=1., title='Space truss', labels=False):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for member, (i, j) in enumerate(elements):
        added = member >= len(original_elements)
        label = ('Original members' if member == 0 else
                 'Added braces' if member == len(original_elements) else None)
        ax.plot(*nodes[[i, j]].T, color='tab:orange' if added else '0.5',
                linewidth=1, label=label)
    positions = nodes
    if displacement is not None:
        positions = nodes + scale*displacement.reshape(-1, 3)
        for member, (i, j) in enumerate(elements):
            ax.plot(*positions[[i, j]].T, color='tab:blue',
                    label=f'Deformed ({scale:g}×)' if member == 0 else None)
    supports = [node_index[label] for label in [1, 2, 3, 4]]
    ax.scatter(*nodes[supports].T, marker='^', color='black', s=60, label='Fixed supports')
    if labels:
        for label, position in zip(node_ids, nodes):
            ax.text(*position, str(label), fontsize=8)
    # Include both configurations so magnified displacements are not clipped.
    bounds = np.vstack([nodes, positions])
    midpoint = (bounds.max(axis=0)+bounds.min(axis=0))/2
    radius = .55*np.max(np.ptp(bounds, axis=0))
    ax.set_xlim(midpoint[0]-radius, midpoint[0]+radius)
    ax.set_ylim(midpoint[1]-radius, midpoint[1]+radius)
    ax.set_zlim(midpoint[2]-radius, midpoint[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=title)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

plot_structure(title='Original geometry with eight added braces', labels=True)

# %% [markdown]
# ## Static displacement, reactions, and member stress
#
# Apply 1 kN in x at node 10, as in the original script. Solve K_ff u_f=f_f
# with zero support displacements. Check the free residual and force balance.
# Positive axial stress means tension; negative stress means compression.

# %%
force = np.zeros(ndof)
force[node_dofs(10)[0]] = 1000.
u = np.zeros(ndof)
u[free] = linalg.solve(K_ff, force[free], assume_a='pos')
reaction = K@u-force
stress = np.empty(len(elements))
for member, (i, j) in enumerate(elements):
    delta = nodes[j]-nodes[i]
    length = np.linalg.norm(delta)
    extension = (u[3*j:3*j+3]-u[3*i:3*i+3])@(delta/length)
    stress[member] = E*extension/length
relative_residual = np.linalg.norm(reaction[free])/np.linalg.norm(force[free])
balance = (force+reaction).reshape(-1, 3).sum(axis=0)
assert relative_residual < 1e-10
assert np.linalg.norm(balance)/np.linalg.norm(force) < 1e-10
assert np.isclose(.5*u@K@u, .5*force@u)
print('Node 10 displacement [mm]:', u[node_dofs(10)]*1000)
print('Largest nodal displacement magnitude [mm]:', np.linalg.norm(u.reshape(-1, 3), axis=1).max()*1000)
print('Free relative residual / force balance [N]:', relative_residual, balance)
print('Min/max axial stress [MPa]:', stress.min()/1e6, stress.max()/1e6)
plot_structure(u, scale=500., title='Static displacement magnified 500×')

# %% [markdown]
# ## (Additional) Natural frequencies and mode shapes
#
# Solve the symmetric generalized problem K_ff phi = omega² M_ff phi
# directly. Inverting M and passing M^-1 K to a symmetric eigensolver is
# generally invalid because that product need not be symmetric.
# Each eigenpair is one mode; there is no automatic grouping in threes.
# SciPy returns mass-normalized modes: phi^T M phi=I.

# %%
eigenvalues, modes = linalg.eigh(K_ff, M_ff)
assert np.all(eigenvalues > 0)
omega_n = np.sqrt(eigenvalues)
frequencies = omega_n/(2*np.pi)
assert np.allclose(modes.T@M_ff@modes, np.eye(len(free)), atol=1e-10)
modal_residual = K_ff@modes-(M_ff@modes)*eigenvalues
assert np.linalg.norm(modal_residual)/np.linalg.norm(K_ff@modes) < 1e-10
print('First six natural frequencies [Hz]:', frequencies[:6])
modes_full = np.zeros((ndof, len(free)))
modes_full[free] = modes
for mode in range(3):
    shape = modes_full[:, mode]
    scale = .03/np.max(np.abs(shape))
    plot_structure(shape, scale=scale,
                   title=f'Mode {mode+1}: {frequencies[mode]:.2f} Hz (arbitrary amplitude)')

# %% [markdown]
# ## Modal energy check
#
# For the same modal amplitude, peak strain energy is .5 phi^T K phi
# and peak kinetic energy is .5 omega² phi^T M phi. They must agree.
# These peaks occur at different times in an undamped free oscillation;
# adding them would not give its instantaneous total energy.

# %%
peak_strain = .5*np.sum(modes*(K_ff@modes), axis=0)
peak_kinetic = .5*eigenvalues*np.sum(modes*(M_ff@modes), axis=0)
assert np.allclose(peak_strain, peak_kinetic, rtol=1e-10)
print('Largest relative modal energy mismatch:', np.max(np.abs(peak_strain/peak_kinetic-1)))
plt.plot(np.arange(1, 7), peak_kinetic[:6]/peak_strain[:6], 'o')
plt.axhline(1., color='k', ls='--')
plt.xlabel('Mode number'); plt.ylabel('Peak kinetic / peak strain energy')
plt.title('Modal energy identity'); plt.show()

# %% [markdown]
# ## (Additional) Harmonic forcing and Rayleigh damping
#
# Set C=alpha M+beta_R K to give 2% damping at modes 1 and 6.
# Then zeta_j=alpha/(2 omega_j)+beta_R omega_j/2; other modes have
# different damping. Start from rest with a sinusoidal 1 kN z load at
# node 20, at 0.95 times the first natural frequency.
# RK4 is explicit: unconditional Newmark stability does not imply RK4
# stability. Select dt using the full damped system's fastest eigenvalue,
# then also resolve the forcing period. The step is deliberately small.

# %%
zeta = .02
damping_system = np.column_stack([1/(2*omega_n[[0, 5]]), omega_n[[0, 5]]/2])
alpha, beta_R = linalg.solve(damping_system, np.full(2, zeta))
C_ff = alpha*M_ff + beta_R*K_ff
modal_damping = alpha/(2*omega_n)+beta_R*omega_n/2
force_amplitude = 1000.
force_frequency = .95*frequencies[0]
omega_force = 2*np.pi*force_frequency
force_global = node_dofs(20)[2]
force_dof = int(np.flatnonzero(free == force_global)[0])
force_pattern = np.zeros(len(free))
force_pattern[force_dof] = force_amplitude
period = 1/force_frequency
state_matrix = np.block([[np.zeros_like(K_ff), np.eye(len(free))],
                         [-linalg.solve(M_ff, K_ff, assume_a='pos'),
                          -linalg.solve(M_ff, C_ff, assume_a='pos')]])
max_rate = np.max(np.abs(linalg.eigvals(state_matrix)))
dt_limit = min(period/200, .2/max_rate)
nsteps = int(np.ceil(20*period/dt_limit))
time = np.linspace(0., 20*period, nsteps+1)
print('Rayleigh alpha [1/s], beta_R [s]:', alpha, beta_R)
print('First six modal damping ratios:', modal_damping[:6])
print('Force frequency [Hz] / duration [s] / dt [s]:', force_frequency, time[-1], time[1])
print('Time steps:', nsteps)

# %% [markdown]
# ## (Additional) Newmark-beta: average acceleration
#
# With beta_N=1/4 and gamma=1/2, form displacement and velocity predictors.
# Solve (M+gamma dt C+beta_N dt² K) a_next
# = f_next-C v_pred-K u_pred, then correct displacement and velocity.
# This acceleration form includes both damping terms and factors the
# constant effective matrix once for this uniform time grid.

# %%
def solve_newmark(M, C, K, time, force_pattern, omega_force):
    dt = time[1]-time[0]
    assert np.allclose(np.diff(time), dt), 'Use a uniform time grid.'
    beta_N, gamma = .25, .5
    factor = linalg.cho_factor(M+gamma*dt*C+beta_N*dt**2*K)
    u = np.zeros((len(M), len(time)))
    v = np.zeros_like(u)
    a = np.zeros_like(u)  # From rest; f(0)=0, so a(0)=0.
    for step in range(1, len(time)):
        u_pred = u[:, step-1]+dt*v[:, step-1]+(.5-beta_N)*dt**2*a[:, step-1]
        v_pred = v[:, step-1]+(1-gamma)*dt*a[:, step-1]
        force_next = force_pattern*np.sin(omega_force*time[step])
        a[:, step] = linalg.cho_solve(factor, force_next-C@v_pred-K@u_pred)
        u[:, step] = u_pred+beta_N*dt**2*a[:, step]
        v[:, step] = v_pred+gamma*dt*a[:, step]
    return u, v, a

u_newmark, v_newmark, a_newmark = solve_newmark(
    M_ff, C_ff, K_ff, time, force_pattern, omega_force)
force_history = force_pattern[:, None]*np.sin(omega_force*time)
dynamic_residual = M_ff@a_newmark+C_ff@v_newmark+K_ff@u_newmark-force_history
assert np.linalg.norm(dynamic_residual)/np.linalg.norm(force_history) < 1e-10
print('Newmark peak node 20 z displacement [mm]:', np.max(np.abs(u_newmark[force_dof]))*1000)

# %% [markdown]
# ## (Additional) RK4 comparison and time-step refinement
#
# Write y=[u, v], so y_dot=A y+[0, M^-1 f(t)]. Compute the constant
# acceleration operators with linear solves, avoiding an explicit inverse.
# Compare the entire displacement history, then halve the Newmark step.
# Agreement between methods and refinement are numerical checks, not
# validation of the physical assumptions or laboratory measurements.

# %%
def solve_rk4(state_matrix, M, time, force_pattern, omega_force):
    size = len(M)
    load_state = np.r_[np.zeros(size), linalg.solve(M, force_pattern, assume_a='pos')]
    history = np.zeros((2*size, len(time)))

    def rhs(t, state):
        return state_matrix@state+load_state*np.sin(omega_force*t)

    for step in range(1, len(time)):
        t, dt = time[step-1], time[step]-time[step-1]
        state = history[:, step-1]
        k1 = rhs(t, state)
        k2 = rhs(t+dt/2, state+dt*k1/2)
        k3 = rhs(t+dt/2, state+dt*k2/2)
        k4 = rhs(t+dt, state+dt*k3)
        history[:, step] = state+dt*(k1+2*k2+2*k3+k4)/6
    return history[:size], history[size:]

u_rk4, v_rk4 = solve_rk4(state_matrix, M_ff, time, force_pattern, omega_force)
assert np.all(np.isfinite(u_rk4))
relative_difference = np.linalg.norm(u_newmark-u_rk4)/np.linalg.norm(u_rk4)
fine_time = np.linspace(time[0], time[-1], 2*nsteps+1)
u_fine, _, _ = solve_newmark(M_ff, C_ff, K_ff, fine_time, force_pattern, omega_force)
refinement_error = np.linalg.norm(u_newmark-u_fine[:, ::2])/np.linalg.norm(u_fine[:, ::2])
assert relative_difference < .005
assert refinement_error < .005
print('Newmark/RK4 relative history difference:', relative_difference)
print('Newmark dt / dt/2 relative history difference:', refinement_error)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(time, force_history[force_dof])
axes[0].set(ylabel='Node 20 z force [N]')
axes[1].plot(time, u_newmark[force_dof]*1000, label='Newmark-beta')
axes[1].plot(time, u_rk4[force_dof]*1000, '--', label='RK4')
axes[1].set(xlabel='Time [s]', ylabel='Node 20 z displacement [mm]')
axes[1].legend()
plt.tight_layout(); plt.show()
print('Exercise: change area or density and predict the frequency change before rerunning.')
print('Exercise: change force frequency and damping; distinguish transient from steady response.')
print('Exercise: remove a brace and inspect the smallest stiffness eigenvalues before solving.')
