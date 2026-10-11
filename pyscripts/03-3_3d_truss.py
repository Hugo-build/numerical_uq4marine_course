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


