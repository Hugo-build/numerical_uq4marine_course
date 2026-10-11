# %% [markdown]
# # 03-4 · Static analysis of a hollow cylinder with flat shell elements
#
# **Goal:** mesh a hollow cylinder's middle surface with plane quadrilaterals,
# give each panel a physical thickness, and assemble a three-dimensional shell.
# Each node has [ux, uy, uz, rx, ry, rz]; rotations are in radians.
# A panel combines plane-stress membrane action, Mindlin plate bending, and
# MITC4 assumed transverse shear. Local matrices are rotated into global axes.
# The cylinder is open at both ends. There are no caps or internal pressure.
# Small displacement, isotropic linear elasticity, and SI units are assumed.
# Buckling, yielding, contact, and curved-element geometry are excluded.
#
# The main example is a cantilever with a clamped lower rim and distributed
# axial and transverse forces on its upper rim. An additional uniform-tension
# patch test has an exact solution and deliberately permits Poisson contraction.
# All code is in this file; run cells in order or execute the whole script.
# MITC4 background: https://doi.org/10.1002/nme.1620210213

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=5, suppress=True)
print('Hollow-cylinder shell model. Units: m, N, Pa; six DOFs per node.')

# %% [markdown]
# ## Geometry, material, and loading
#
# Radius refers to the outer surface. Flat panels approximate the middle
# surface at R = outer_radius - thickness/2. Increasing n_circumference
# improves this polygonal approximation. F_transverse acts in global +x;
# F_axial acts in +z. Both are total rim forces, not forces per node.
#
# With $n_c$ circumferential panels and $n_z$ axial layers,
# $$R_m=R_o-t/2,\quad \theta_j=2\pi j/n_c,\quad z_k=kL/n_z,$$
# $$\mathbf{x}_{k,j}=[R_m\cos\theta_j,R_m\sin\theta_j,z_k]^T.$$
# A flat panel has width $b=2R_m\sin(\pi/n_c)$ and height $h=L/n_z$.
# The polygonal wall area is $A_h=n_cbt$; the smooth circular wall area is
# $A_c=\pi[R_o^2-(R_o-t)^2]$. Both are cross-sectional material areas,
# not middle-surface areas. The shell has $6n_{nodes}$ global unknowns.

# %%
E, nu = 210e9, 0.30
outer_radius, thickness, length = 0.20, 0.005, 2.0
n_circumference, n_axial = 24, 8
F_axial, F_transverse = 100_000.0, 10_000.0
assert E > 0 and -1 < nu < 0.5
assert 0 < thickness < outer_radius and length > 0
assert n_circumference >= 8 and n_axial >= 1
radius = outer_radius - thickness/2
angles = np.arange(n_circumference)*2*np.pi/n_circumference
nodes = np.array([[radius*np.cos(a), radius*np.sin(a), z]
                  for z in np.linspace(0, length, n_axial+1) for a in angles])
elements = []
for level in range(n_axial):
    for j in range(n_circumference):
        next_j = (j+1) % n_circumference
        elements.append([level*n_circumference+j,
                         level*n_circumference+next_j,
                         (level+1)*n_circumference+next_j,
                         (level+1)*n_circumference+j])
elements = np.array(elements, dtype=int)
bottom = np.arange(n_circumference)
top = n_axial*n_circumference + np.arange(n_circumference)
ndof = 6*len(nodes)
polygon_perimeter = 2*n_circumference*radius*np.sin(np.pi/n_circumference)
area_mesh = polygon_perimeter*thickness
area_circle = np.pi*(outer_radius**2-(outer_radius-thickness)**2)

print('Nodes / shell panels / DOFs:', len(nodes), len(elements), ndof)
print('Outer radius / thickness / length [m]:', outer_radius, thickness, length)
print('Mesh / circular wall area [m²]:', area_mesh, area_circle)

# %% [markdown]
# ## See one shell element before deriving it
#
# The blue face is the flat middle surface. The transparent box shows the
# physical wall thickness, with faces at local $s=\pm t/2$. Only the four
# middle-surface corners are FE nodes: the thickness faces add no nodes.
# The left sketch uses local coordinates, not the cylinder's global axes.
# The right sketch maps the same node order to the reference square and
# marks the four edge midpoints used for assumed-shear interpolation.
# Local node labels 1–4 correspond to connectivity entries 0–3 in Python.
# Every corner carries $[u,v,w,r_x,r_y,r_z]$ (24 DOFs per element).




# %%
panel_width = 2*radius*np.sin(np.pi/n_circumference)
panel_height = length/n_axial
panel_corners = np.array([[0., 0., 0.], [panel_width, 0., 0.],
                          [panel_width, panel_height, 0.], [0., panel_height, 0.]])
lower_face = panel_corners-np.array([0., 0., thickness/2])
upper_face = panel_corners+np.array([0., 0., thickness/2])
thickness_faces = [lower_face, upper_face]
for a in range(4):
    b = (a+1) % 4
    thickness_faces.append(np.array([lower_face[a], lower_face[b], upper_face[b], upper_face[a]]))
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.add_collection3d(Poly3DCollection(thickness_faces, facecolors='lightgrey',
                                    edgecolors='grey', alpha=0.25, linewidths=0.7))
ax.add_collection3d(Poly3DCollection([panel_corners], facecolors='cornflowerblue',
                                    edgecolors='navy', alpha=0.7))
ax.scatter(*panel_corners.T, color='navy', s=35)
label_offsets = np.array([[-0.006, -0.010, 2*thickness], [0.004, -0.010, 2*thickness],
                          [0.004, 0.007, 2*thickness], [-0.006, 0.007, 2*thickness]])
for a, point in enumerate(panel_corners+label_offsets):
    ax.text(*point, f'{a+1}', color='navy', fontsize=12, zorder=10)
ax.set(xlim=(-0.02, panel_width+0.02), ylim=(-0.03, panel_height+0.03),
       zlim=(-0.015, 0.015), xlabel='local x [m]', ylabel='local y [m]',
       zlabel='local s [m]', title='One flat Q4 shell panel with thickness')
ax.set_box_aspect((panel_width+0.04, panel_height+0.06, 0.03), zoom=1.0)
ax.set_axis_off()
frame_origin = np.array([-0.015, -0.025, -0.012])
for label, vector in [('x', [0.04, 0., 0.]), ('y', [0., 0.06, 0.]), ('s', [0., 0., 0.022])]:
    ax.quiver(*frame_origin, *vector, color='black', arrow_length_ratio=0.15)
    ax.text(*(frame_origin+np.array(vector)), f' {label}', fontsize=11, zorder=10)
ax.text2D(0.10, 0.08, 'Blue: middle surface; grey: wall faces at s = ±t/2',
          transform=ax.transAxes, fontsize=10)
ax.view_init(elev=25, azim=-55)
ax = fig.add_subplot(122)
reference = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
closed = np.vstack([reference, reference[0]])
ax.fill(reference[:, 0], reference[:, 1], color='cornflowerblue', alpha=0.15)
ax.plot(*closed.T, 'o-', color='navy')
for a, (xi, eta) in enumerate(reference):
    ax.text(1.12*xi, 1.12*eta, f'{a+1}: ({xi:g}, {eta:g})',
            ha='center', va='center', fontsize=10)
for label, (xi, eta) in zip(['A', 'B', 'C', 'D'], [(0., -1.), (1., 0.), (0., 1.), (-1., 0.)]):
    ax.plot(xi, eta, 's', color='darkorange')
    ax.annotate(label, (xi, eta), xytext=(6, 6), textcoords='offset points', color='darkorange')
ax.set(xlim=(-1.7, 1.7), ylim=(-1.5, 1.5), xlabel=r'$\xi$', ylabel=r'$\eta$',
       title='Reference square; orange points tie shear')
ax.set_aspect('equal')
ax.grid(alpha=0.2)
fig.text(0.5, 0.02, f'b = {panel_width:.4f} m; h = {panel_height:.4f} m; t = {thickness:g} m. '
         'Four nodes on the middle surface, six DOFs per node.', ha='center', fontsize=10)
fig.tight_layout(rect=(0., 0.08, 1., 0.95))
plt.show()
print('Q4 element sketch: local nodes 1–4 and assumed-shear tying points A–D.')

# %% [markdown]
# ## Q4 interpolation and strain-displacement matrices
#
# Counter-clockwise local nodes correspond to (-1,-1), (1,-1), (1,1),
# (-1,1). Local x follows the rim, local y follows the cylinder axis, and
# local z points outwards. Engineering shear strains are used.
# Membrane strain = [u_x, v_y, u_y+v_x].
# Curvature = [ry_x, -rx_y, ry_y-rx_x].
# Transverse shear = [w_x+ry, w_y-rx]. These signs use physical rotations:
# a rigid rotation about local y gives w=-ry*x and zero shear strain.
# The weak drilling term penalizes rz - (v_x-u_y)/2, not rigid rotation.
#
# **Interpolation and geometry.** With corner signs $(\xi_a,\eta_a)$,
# $$N_a(\xi,\eta)=\tfrac14(1+\xi_a\xi)(1+\eta_a\eta),\quad
# \sum_aN_a=1,\quad N_a(\xi_b,\eta_b)=\delta_{ab}.$$
# Coordinates and each displacement/rotation component use the same shapes:
# $\mathbf{x}(\xi,\eta)=\sum_aN_a\mathbf{x}_a$ and
# $\mathbf{q}(\xi,\eta)=\sum_aN_a\mathbf{q}_a$.
# The geometry Jacobian and row-gradient transformation used in the code are
# $$J=\begin{bmatrix}x_{,\xi}&x_{,\eta}\\y_{,\xi}&y_{,\eta}\end{bmatrix},
# \quad [N_{a,x},N_{a,y}]=[N_{a,\xi},N_{a,\eta}]J^{-1},
# \quad dA=\det J\,d\xi\,d\eta.$$
# A comma denotes differentiation; $(u,v,w)$ are local translations.
#
# **Strains.** Stack the corner DOFs into $\mathbf{q}_e\in\mathbb{R}^{24}$.
# Write $p=N_{a,x}$, $q=N_{a,y}$, $N=N_a$. The six-column blocks for node a are
# $$B_{m,a}=\begin{bmatrix}p&0&0&0&0&0\\0&q&0&0&0&0\\q&p&0&0&0&0\end{bmatrix},$$
# $$B_{b,a}=\begin{bmatrix}0&0&0&0&p&0\\0&0&0&-q&0&0\\0&0&0&-p&q&0\end{bmatrix},$$
# $$B_{s,a}=\begin{bmatrix}0&0&p&0&N&0\\0&0&q&-N&0&0\end{bmatrix},
# \quad B_{d,a}=[q/2,-p/2,0,0,0,N].$$
# Concatenate all four node blocks to obtain the full B matrices. Then
# $\boldsymbol\epsilon_m=B_m\mathbf{q}_e$, $\boldsymbol\kappa=B_b\mathbf{q}_e$,
# $\boldsymbol\gamma_s=B_s\mathbf{q}_e$, and $d=B_d\mathbf{q}_e$.
# Through the thickness, $u(x,y,s)=u(x,y)+s r_y(x,y)$,
# $v(x,y,s)=v(x,y)-s r_x(x,y)$, and $w$ is independent of s. Therefore
# $\boldsymbol\epsilon(x,y,s)=\boldsymbol\epsilon_m+s\boldsymbol\kappa$.
#
# **Assumed transverse shear.** For the rectangular panels in this mesh,
# $$\widetilde\gamma_{xs}=\tfrac{1-\eta}{2}\gamma_{xs}(0,-1)
# +\tfrac{1+\eta}{2}\gamma_{xs}(0,1),$$
# $$\widetilde\gamma_{ys}=\tfrac{1-\xi}{2}\gamma_{ys}(-1,0)
# +\tfrac{1+\xi}{2}\gamma_{ys}(1,0).$$
# Apply these interpolations to the rows of $B_s$ to form $\widetilde B_s$.
# These are the orange points A,C and D,B, respectively, in the sketch.

# %%
def q4_shape(xi, eta, xy):
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    N = (1+signs[:, 0]*xi)*(1+signs[:, 1]*eta)/4
    dnatural = np.column_stack([
        signs[:, 0]*(1+signs[:, 1]*eta)/4,
        signs[:, 1]*(1+signs[:, 0]*xi)/4,
    ])
    J = xy.T @ dnatural
    detJ = np.linalg.det(J)
    assert detJ > 0, 'Shell node order or geometry gives a nonpositive Jacobian.'
    gradient = dnatural @ np.linalg.inv(J)
    return N, gradient, detJ


def shell_B(xi, eta, xy):
    N, gradient, detJ = q4_shape(xi, eta, xy)
    membrane, bending = np.zeros((3, 24)), np.zeros((3, 24))
    shear, drilling = np.zeros((2, 24)), np.zeros((1, 24))
    for a, (dx, dy) in enumerate(gradient):
        i = 6*a
        membrane[0, i] = dx
        membrane[1, i+1] = dy
        membrane[2, i:i+2] = [dy, dx]
        bending[0, i+4] = dx
        bending[1, i+3] = -dy
        bending[2, i+3:i+5] = [-dx, dy]
        shear[0, i+2:i+5] = [dx, 0., N[a]]
        shear[1, i+2:i+4] = [dy, -N[a]]
        drilling[0, i:i+2] = [dy/2, -dx/2]
        drilling[0, i+5] = N[a]
    return membrane, bending, shear, drilling, detJ


def mitc4_shear(xi, eta, xy):
    # For these rectangular panels, interpolate the x shear from the two
    # horizontal edge midpoints and the y shear from vertical midpoints.
    # This rectangular MITC4 formula must not be reused for distorted quads
    # without a covariant shear transformation.
    lower = shell_B(0., -1., xy)[2][0]
    upper = shell_B(0., 1., xy)[2][0]
    left = shell_B(-1., 0., xy)[2][1]
    right = shell_B(1., 0., xy)[2][1]
    return np.vstack([(1-eta)/2*lower+(1+eta)/2*upper,
                      (1-xi)/2*left+(1+xi)/2*right])


def shell_frame(points):
    ex = points[1]-points[0]
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, points[3]-points[0])
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])  # Global translations/rotations -> local.
    local = (points-points[0]) @ R.T
    assert np.max(np.abs(local[:, 2])) < 1e-12
    xy = local[:, :2]
    assert np.allclose(xy, [[0., 0.], [xy[1, 0], 0.],
                           [xy[1, 0], xy[3, 1]], [0., xy[3, 1]]], atol=1e-12)
    return xy, np.kron(np.eye(8), R)

print('Q4 interpolation, shell strains, assumed shear, and rotations defined.')

# %% [markdown]
# ## Integrate the local stiffness and assemble the global matrix
#
# D is the plane-stress material matrix; G=E/[2(1+nu)]. Membrane stiffness
# scales with t, bending with t³/12, and transverse shear with (5/6)Gt.
# All terms use 2×2 Gauss integration. The assumed shear avoids the severe
# thin-plate shear locking of direct Q4 shear interpolation.
# A small drilling factor ties the otherwise unsupported local rz DOF to
# in-plane spin. It is a numerical parameter: check sensitivity when changing
# the mesh or load, especially for bending. This is an educational flat-panel
# shell, not a general-purpose curved-shell element implementation.
#
# **Constitutive law.** The local in-plane stress is $\boldsymbol\sigma=D\boldsymbol\epsilon$:
# $$D=\frac{E}{1-\nu^2}\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix},
# \quad G=\frac{E}{2(1+\nu)},\quad k_s=5/6.$$
# Thickness integration gives membrane force $\mathbf{n}=tD\boldsymbol\epsilon_m$
# [N/m], bending moment $\mathbf{m}=t^3D\boldsymbol\kappa/12$ [N], and
# transverse shear force $\mathbf{Q}=k_sGt\widetilde{\boldsymbol\gamma}_s$ [N/m].
# The element strain energy is $U_e=\tfrac12\mathbf{q}_e^Tk_e\mathbf{q}_e$, with
# $$k_e=\int_{A_e}\left(t B_m^TDB_m+\frac{t^3}{12}B_b^TDB_b
# +k_sGt\widetilde B_s^T\widetilde B_s+\alpha Gt B_d^TB_d\right)dA.$$
# Here $\alpha=$ drilling_factor is dimensionless. The last term is stabilization.
# Numerically, $k_e\approx\sum_{i,j=1}^2 H(\xi_i,\eta_j)\det J(\xi_i,\eta_j)$,
# where H is the integrand, $\xi_i,\eta_j=\pm1/\sqrt3$, and each Gauss weight is 1.

# %%
D = E/(1-nu**2)*np.array([[1., nu, 0.], [nu, 1., 0.],
                          [0., 0., (1-nu)/2]])
G = E/(2*(1+nu))
drilling_factor = 1e-3
gauss = [-1/np.sqrt(3), 1/np.sqrt(3)]


def local_shell_stiffness(xy, E, nu, thickness, drilling_factor):
    material = E/(1-nu**2)*np.array([[1., nu, 0.], [nu, 1., 0.],
                                   [0., 0., (1-nu)/2]])
    shear_modulus = E/(2*(1+nu))
    ke = np.zeros((24, 24))
    for xi in gauss:
        for eta in gauss:
            Bm, Bb, _, Bd, detJ = shell_B(xi, eta, xy)
            Bs = mitc4_shear(xi, eta, xy)
            ke += detJ*(thickness*Bm.T @ material @ Bm
                        + thickness**3/12*Bb.T @ material @ Bb
                        + (5/6)*shear_modulus*thickness*Bs.T @ Bs
                        + drilling_factor*shear_modulus*thickness*Bd.T @ Bd)
    return ke


print('Plane-stress material D [Pa]:\n', D)
print('Local shell stiffness function defined, including thickness integration.')

# %% [markdown]
# ## Rotate, assemble, and check rigid motions
#
# Let the rows of $R_e$ be the local unit axes in global coordinates.
# Each translation and rotation transforms with the same R, giving
# $T_e=\operatorname{diag}(R_e,\ldots,R_e)$ with eight 3×3 blocks.
# If $L_e$ extracts an element's 24 global DOFs from the system displacement,
# $$\mathbf{q}_e=T_eL_e\mathbf{u},\quad
# K=\sum_eL_e^TT_e^Tk_eT_eL_e.$$
# The Python dofs array represents $L_e$ without constructing that matrix.
# Shared DOFs receive contributions from every adjoining element; duplicate
# sparse entries are summed when COO is converted to CSR.
# For rigid translation $\mathbf{u}_a=\mathbf{c}$, rotations are zero.
# For rigid rotation $\mathbf{u}_a=\boldsymbol\omega\times\mathbf{x}_a$,
# all nodal rotations equal $\boldsymbol\omega$. Both must give $k_e\mathbf{q}_e=0$.
# Element symmetry follows from the quadratic strain energy.

# %%
rows, columns, values, element_data = [], [], [], []
for connectivity in elements:
    xy, T = shell_frame(nodes[connectivity])
    ke = local_shell_stiffness(xy, E, nu, thickness, drilling_factor)
    dofs = (6*connectivity[:, None]+np.arange(6)).ravel()
    rows.extend(np.repeat(dofs, 24))
    columns.extend(np.tile(dofs, 24))
    values.extend((T.T @ ke @ T).ravel())
    element_data.append((xy, T, dofs))
K = coo_matrix((values, (rows, columns)), shape=(ndof, ndof)).tocsr()
assert np.max(np.abs((K-K.T).data), initial=0.) < 1e-12*np.max(np.abs(K.data))

# Check the six rigid motions of a single panel before applying supports.
xy, T, dofs = element_data[0]
ke = local_shell_stiffness(xy, E, nu, thickness, drilling_factor)
points = np.column_stack([xy, np.zeros(4)])
for direction in np.eye(3):
    translation = np.tile(np.r_[direction, np.zeros(3)], 4)
    rotation = np.column_stack([np.cross(direction, points),
                               np.tile(direction, (4, 1))]).ravel()
    for rigid in [translation, rotation]:
        assert np.linalg.norm(ke @ rigid) < 1e-10*np.linalg.norm(ke)*np.linalg.norm(rigid)
print('Sparse stiffness assembled; symmetry and six element rigid motions checked.')

# %% [markdown]
# ## Solve with prescribed displacements and verify equilibrium
#
# Partition K into free and prescribed DOFs. Nonzero prescribed values are
# included in the right-hand side, so this also supports the patch test.
# Reaction = K u - f. The global moment balance includes both r × force
# and nodal reaction couples because shell nodes have rotational DOFs.
#
# **Consistent rim loading.** The constant line load is
# $\boldsymbol\ell=[F_x,0,F_z]^T/(n_cb)$ [N/m]. Each top panel edge gives
# $\int_{edge}N_a\boldsymbol\ell\,ds=b\boldsymbol\ell/2$ to each endpoint.
# Two edges meet at each rim node, hence $\mathbf{f}_a=[F_x,0,F_z]^T/n_c$;
# no nodal couples are applied. The lower-rim clamp prescribes all six DOFs.
#
# **Partition and solve.** Subscripts f and c denote free and prescribed DOFs:
# $$\begin{bmatrix}K_{ff}&K_{fc}\\K_{cf}&K_{cc}\end{bmatrix}
# \begin{bmatrix}\mathbf{u}_f\\\mathbf{u}_c\end{bmatrix}
# =\begin{bmatrix}\mathbf{f}_f\\\mathbf{f}_c+\mathbf{r}_c\end{bmatrix},$$
# $$K_{ff}\mathbf{u}_f=\mathbf{f}_f-K_{fc}\mathbf{u}_c,\qquad
# \mathbf{r}=K\mathbf{u}-\mathbf{f}.$$
# In the main case $\mathbf{u}_c=0$. The patch case below prescribes nonzero contraction.
# Check the free residual, $\sum_a(\mathbf{f}_a+\mathbf{r}_a)=0$ for forces, and
# $\sum_a[\mathbf{x}_a\times(\mathbf{f}_a+\mathbf{r}_a)+\mathbf{m}_a]=0$
# for moments; $\mathbf{m}_a$ contains nodal applied/reaction couples.
# For the zero-displacement clamp, $U=\tfrac12\mathbf{u}^TK\mathbf{u}
# =\tfrac12\mathbf{u}^T\mathbf{f}$. Expected support resultants are
# $[-F_x,0,-F_z]$ and $[0,-LF_x,0]$ about the origin.

# %%
def solve_shell(force, fixed, prescribed=None):
    free = np.setdiff1d(np.arange(ndof), fixed)
    displacement = np.zeros(ndof)
    if prescribed is not None:
        displacement[fixed] = prescribed
    rhs = force[free]-K[free][:, fixed] @ displacement[fixed]
    displacement[free] = spsolve(K[free][:, free], rhs)
    assert np.all(np.isfinite(displacement)), 'The supported model is singular.'
    reaction = K @ displacement-force
    relative_residual = np.linalg.norm(reaction[free])/max(np.linalg.norm(force), 1.)
    assert relative_residual < 1e-8
    total = (force+reaction).reshape(-1, 6)
    force_balance = total[:, :3].sum(axis=0)
    moment_balance = (np.cross(nodes, total[:, :3])+total[:, 3:]).sum(axis=0)
    force_scale = max(np.linalg.norm(force.reshape(-1, 6)[:, :3], axis=1).sum(), 1.)
    assert np.linalg.norm(force_balance) < 1e-7*force_scale
    assert np.linalg.norm(moment_balance) < 1e-7*force_scale*max(length, outer_radius)
    return displacement, reaction, relative_residual, force_balance, moment_balance


fixed = (6*bottom[:, None]+np.arange(6)).ravel()
force = np.zeros(ndof)
force[6*top] = F_transverse/n_circumference
force[6*top+2] = F_axial/n_circumference
u, reaction, residual, force_balance, moment_balance = solve_shell(force, fixed)
tip_mean = u.reshape(-1, 6)[top, :3].mean(axis=0)
support = reaction.reshape(-1, 6)[bottom]
support_moment = (np.cross(nodes[bottom], support[:, :3])+support[:, 3:]).sum(axis=0)
assert np.isclose(0.5*u @ (K @ u), 0.5*u @ force, rtol=1e-8)
print('Total applied [Fx, Fy, Fz] [N]:', force.reshape(-1, 6)[:, :3].sum(axis=0))
print('Mean upper-rim displacement [ux, uy, uz] [mm]:', 1000*tip_mean)
print('Support resultant [N]:', support[:, :3].sum(axis=0))
print('Support moment about origin [N m]:', support_moment)
print('Free residual / force balance / moment balance:', residual, force_balance, moment_balance)
print('Elastic strain energy [J]:', 0.5*u @ (K @ u))

# %% [markdown]
# ## Recover membrane stress and bending stress on both wall faces
#
# At the panel centre, sigma(±t/2) = D (Bm u ± t/2 Bb u).
# These are local [sigma_x, sigma_y, tau_xy] plane-stress components.
# Local y is axial, so sigma_y is the cylinder axial stress.
# The reported surface von Mises stress uses those three components;
# transverse shear is represented separately as a plate shear resultant.
# Centre samples are not a resolved maximum near the clamp.
#
# First extract and rotate: $\mathbf{q}_e=T_eL_e\mathbf{u}$. Evaluate B at
# $(\xi,\eta)=(0,0)$, then
# $$\boldsymbol\sigma^{\pm}=D(B_m\mathbf{q}_e\pm\tfrac{t}{2}B_b\mathbf{q}_e),
# \quad\mathbf{Q}=k_sGt\widetilde B_s\mathbf{q}_e.$$
# For each face's $[\sigma_x,\sigma_y,\tau_{xy}]$, the plane-stress equivalent is
# $$\sigma_{vm}^{\pm}=\sqrt{\sigma_x^2+\sigma_y^2-\sigma_x\sigma_y+3\tau_{xy}^2}.$$
# The panel colour uses $\max(\sigma_{vm}^-,\sigma_{vm}^+)$.
# Surface plane stress sets the normal-through-thickness stress to zero;
# the separate Q resultant is not included in this wall-face equivalent stress.

# %%
def recover_shell(displacement):
    membrane, faces, shear_resultant = [], [], []
    for xy, T, dofs in element_data:
        local_u = T @ displacement[dofs]
        Bm, Bb, _, _, _ = shell_B(0., 0., xy)
        mid = D @ (Bm @ local_u)
        bend = thickness/2*D @ (Bb @ local_u)
        membrane.append(mid)
        faces.append([mid-bend, mid+bend])
        shear_resultant.append((5/6)*G*thickness*mitc4_shear(0., 0., xy) @ local_u)
    membrane, faces = np.array(membrane), np.array(faces)
    sx, sy, tau = faces[..., 0], faces[..., 1], faces[..., 2]
    vm = np.sqrt(sx**2+sy**2-sx*sy+3*tau**2)
    return membrane, faces, vm.max(axis=1), np.array(shear_resultant)


membrane_stress, face_stress, von_mises, shear_resultant = recover_shell(u)
print('Axial membrane stress range [MPa]:', membrane_stress[:, 1].min()/1e6,
      membrane_stress[:, 1].max()/1e6)
print('Largest sampled wall-face von Mises stress [MPa]:', von_mises.max()/1e6)
print('Largest centre shear resultant magnitude [N/m]:', np.linalg.norm(shear_resultant, axis=1).max())

# %% [markdown]
# ## Exact uniform-tension patch test and approximate beam comparison
#
# Uniform tension sigma=F/A_mesh has u_z=sigma*z/E and
# [u_x,u_y]=-nu*sigma/E*[x,y]. Prescribe this lateral contraction on the
# lower rim, with zero rotations, and apply only axial rim forces. This
# exact flat-panel membrane test differs from the fully clamped main case.
# The circular Euler–Bernoulli estimate is only a slender-cantilever guide:
# clamping, shear deformation, shell ovalisation, and polygonal geometry
# cause differences. It is not an asserted exact benchmark for the main case.
#
# In the exact membrane test, $\sigma_0=P/A_h$ and $\epsilon_0=\sigma_0/E$:
# $$\mathbf{u}_{exact}(x,y,z)=[-\nu\epsilon_0x,-\nu\epsilon_0y,\epsilon_0z]^T,
# \quad\mathbf{r}_{rotation}=0.$$
# Its local strains are $[-\nu\epsilon_0,\epsilon_0,0]$, so D produces
# $[0,\sigma_0,0]$ with zero curvature and shear. The asserted relative error is
# $e_u=\|\mathbf{u}_{FE}-\mathbf{u}_{exact}\|_2/\|\mathbf{u}_{exact}\|_2$.
# For the circular cantilever estimates,
# $$I_c=\tfrac\pi4[R_o^4-(R_o-t)^4],\quad
# u_x(L)\approx\frac{F_xL^3}{3EI_c},\quad u_z(L)\approx\frac{F_zL}{EA_c}.$$
# These estimates neglect shear, local end effects, and cross-section distortion.

# %%
patch_load = 100_000_0.0
patch_force = np.zeros(ndof)
patch_force[6*top+2] = patch_load/n_circumference
sigma_exact = patch_load/area_mesh
exact_u = np.zeros((len(nodes), 6))
exact_u[:, :2] = -nu*sigma_exact/E*nodes[:, :2]
exact_u[:, 2] = sigma_exact/E*nodes[:, 2]
patch_u, _, patch_residual, _, _ = solve_shell(patch_force, fixed, exact_u.ravel()[fixed])
patch_error = np.linalg.norm(patch_u-exact_u.ravel())/np.linalg.norm(exact_u)
patch_membrane, patch_faces, _, _ = recover_shell(patch_u)
assert patch_error < 1e-7
assert np.allclose(patch_membrane, [0., sigma_exact, 0.], atol=1e-6*sigma_exact)
assert np.allclose(patch_faces, [0., sigma_exact, 0.], atol=1e-6*sigma_exact)
inertia_circle = np.pi/4*(outer_radius**4-(outer_radius-thickness)**4)
beam_tip_x = F_transverse*length**3/(3*E*inertia_circle)
rod_tip_z = F_axial*length/(E*area_circle)
print('Exact tension tip extension [mm]:', 1000*sigma_exact*length/E)
print('Patch displacement relative error / residual:', patch_error, patch_residual)
print('Main FE / circular beam tip ux [mm]:', 1000*tip_mean[0], 1000*beam_tip_x)
print('Main FE / circular rod tip uz [mm]:', 1000*tip_mean[2], 1000*rod_tip_z)

# %% [markdown]
# ## Plot the shell mesh, amplified displacement, and wall-face stress
#
# Colours show the larger von Mises value on the two wall faces of each
# panel. Displacement amplification changes only the display geometry.
# Try increasing n_circumference and n_axial, or reducing drilling_factor
# by ten, and compare the tip displacement and centre stress samples.
#
# Display coordinates are $\mathbf{x}_{plot}=\mathbf{x}+s_d\mathbf{u}_{translation}$,
# where $s_d=0.12L/\max_a\|\mathbf{u}_a\|_2$ (or 1 for zero displacement).
# Thus the largest drawn shift is 12% of the original length; the numerical
# solution and stress colours are never multiplied by this display factor.

# %%
def plot_shell(ax, positions, title, stress=None):
    collection = Poly3DCollection(positions[elements], edgecolors='0.35', linewidths=0.4)
    if stress is None:
        collection.set_facecolor('lightsteelblue')
    else:
        norm = colors.Normalize(vmin=0., vmax=max(stress.max()/1e6, 1e-12))
        collection.set_facecolor(plt.colormaps['viridis'](norm(stress/1e6)))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
                     shrink=0.65, label='Wall-face von Mises [MPa]')
    ax.add_collection3d(collection)
    ax.scatter(*positions[bottom].T, color='black', marker='^', s=12)
    ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=title)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.labelpad = 18
    low, high = positions.min(axis=0), positions.max(axis=0)
    span = high-low
    ax.set_xlim(low[0]-0.1*span[0], high[0]+0.1*span[0])
    ax.set_ylim(low[1]-0.1*span[1], high[1]+0.1*span[1])
    ax.set_zlim(low[2]-0.02*span[2], high[2]+0.05*span[2])
    ax.set_box_aspect(np.maximum(span, 1e-8), zoom=0.80)


displacement = u.reshape(-1, 6)[:, :3]
maximum = np.linalg.norm(displacement, axis=1).max()
scale = 500 #0.30*length/maximum if maximum > 0 else 1.
fig = plt.figure(figsize=(12, 6))
plot_shell(fig.add_subplot(121, projection='3d'), nodes, 'Flat shell mesh; lower rim clamped')
plot_shell(fig.add_subplot(122, projection='3d'), nodes+scale*displacement,
           f'Deformed shell ({scale:.0f}×)', von_mises)
fig.tight_layout(rect=(0., 0.02, 1., 0.92))
plt.show()
print('Exercise: set either force to zero and predict the support forces and moments.')

# %%
