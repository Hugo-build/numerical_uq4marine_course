# %% [markdown]
# # 03-5 · Static analysis of a filled cylinder with 3D solid elements
#
# **Goal:** fill the entire cylinder volume with four-node tetrahedra (T4),
# assemble 3D elastic stiffness, and calculate displacement and stress.
# Each node has [ux, uy, uz]; rotations follow from the displacement field.
# This is a filled, homogeneous solid, with no central cavity or shell wall.
# Small displacement, isotropic linear elasticity, and SI units are assumed.
# Buckling, yielding, contact, and geometric nonlinearity are excluded.
#
# The main cylinder is clamped on the lower face. Uniform tractions on the
# upper face give total axial and transverse forces, as in the shell example.
# A separate uniform-tension patch test verifies an exact 3D elastic solution.
# All functions and equations are visible here; no other course script is needed.

# %%
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=5, suppress=True)
print('Filled-cylinder solid model. Units: m, N, Pa; three DOFs per node.')

# %% [markdown]
# ## Mesh the circular cross-section, then extrude and split into tetrahedra
#
# A centre node and concentric rings form a triangular disk mesh. Each
# triangular prism between axial levels is split into three tetrahedra.
# Sorting the cross-section node labels before splitting ensures neighbours
# choose the same diagonal on their shared rectangular faces.
# The outer surface is polygonal: more circumferential divisions improve
# geometry, while radial and axial divisions refine the volume mesh.
#
# Set $r_i=iR/n_r$, $\theta_j=2\pi j/n_c$, and $z_k=kL/n_z$.
# Ring nodes are $[r_i\cos\theta_j,r_i\sin\theta_j,z_k]^T$; add one centre
# node on each level. There are $(1+n_rn_c)(n_z+1)$ nodes, each with 3 DOFs.
# The polygonal cross-sectional area is $A_h=(n_c/2)R^2\sin(2\pi/n_c)$,
# compared with the smooth circular area $A_c=\pi R^2$.
# For a triangle with sorted lower labels a<b<c and upper copies A,B,C,
# the three tetrahedra are (a,b,c,C), (a,b,B,C), and (a,A,B,C).
# This common ordering gives the same shared-face diagonals in adjacent prisms.
# Reorder any inverted tetrahedron so its signed volume is positive; then
# verify $\sum_eV_e=A_hL$. No material is removed during mesh generation.

# %%
E, nu = 210e9, 0.30
radius, length = 0.20, 2.0
n_circumference, n_radial, n_axial = 32, 4, 24
F_axial, F_transverse = 100_000.0, 10_000.0
assert E > 0 and -1 < nu < 0.5
assert radius > 0 and length > 0
assert n_circumference >= 8 and n_radial >= 1 and n_axial >= 1


def cylinder_mesh(radius, length, n_circumference, n_radial, n_axial):
    angles = np.arange(n_circumference)*2*np.pi/n_circumference
    disk_nodes = [[0., 0.]]
    for r in np.linspace(radius/n_radial, radius, n_radial):
        disk_nodes.extend(np.column_stack([r*np.cos(angles), r*np.sin(angles)]))
    disk_nodes = np.array(disk_nodes)
    triangles = []
    for j in range(n_circumference):
        triangles.append([0, 1+j, 1+(j+1) % n_circumference])
    for ring in range(1, n_radial):
        inner = 1+(ring-1)*n_circumference
        outer = 1+ring*n_circumference
        for j in range(n_circumference):
            k = (j+1) % n_circumference
            triangles.extend([[inner+j, outer+j, outer+k],
                              [inner+j, outer+k, inner+k]])
    triangles = np.array(triangles, dtype=int)
    n_disk = len(disk_nodes)
    nodes = np.array([[x, y, z] for z in np.linspace(0., length, n_axial+1)
                      for x, y in disk_nodes])
    tetrahedra = []
    for level in range(n_axial):
        for triangle in triangles:
            a, b, c = np.sort(triangle)+level*n_disk
            A, B, C = np.array([a, b, c])+n_disk
            tetrahedra.extend([[a, b, c, C], [a, b, B, C], [a, A, B, C]])
    tetrahedra = np.array(tetrahedra, dtype=int)
    for tet in tetrahedra:
        if np.linalg.det((nodes[tet[1:]]-nodes[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
    return nodes, tetrahedra, disk_nodes, triangles


nodes, elements, disk_nodes, disk_triangles = cylinder_mesh(
    radius, length, n_circumference, n_radial, n_axial)
n_disk = len(disk_nodes)
bottom = np.arange(n_disk)
top = n_axial*n_disk+np.arange(n_disk)
ndof = 3*len(nodes)
disk_edges_1 = disk_nodes[disk_triangles[:, 1]]-disk_nodes[disk_triangles[:, 0]]
disk_edges_2 = disk_nodes[disk_triangles[:, 2]]-disk_nodes[disk_triangles[:, 0]]
triangle_areas = (disk_edges_1[:, 0]*disk_edges_2[:, 1]
                  - disk_edges_1[:, 1]*disk_edges_2[:, 0])/2
assert np.all(triangle_areas > 0)
area_mesh = triangle_areas.sum()
area_circle = np.pi*radius**2
print('Nodes / tetrahedra / DOFs:', len(nodes), len(elements), ndof)
print('Mesh / circular area [m²]:', area_mesh, area_circle)
print('Polygonal area error [%]:', 100*(area_mesh/area_circle-1))

# %% [markdown]
# ## See the four-node solid element before deriving it
#
# A T4 is a volume bounded by four triangular faces and six edges. It has
# four corner nodes, no middle or face nodes, and 12 translation DOFs.
# The left sketch is the reference tetrahedron used to describe its linear
# shapes. The right is an actual tetrahedron from the filled-cylinder mesh.
# Local labels 1–4 refer to the order in that element's connectivity, while
# the accompanying global labels identify rows of the nodes array.
# A tetrahedron's triangular face is not a separate shell element.

# %%
reference_tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
example_connectivity = elements[0]
example_tet = nodes[example_connectivity]
tet_face_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
fig = plt.figure(figsize=(12, 5))
for column, points, title in [(1, reference_tet, 'Reference T4: four nodes, six edges'),
                               (2, example_tet, 'One actual tetrahedron from the volume mesh')]:
    ax = fig.add_subplot(1, 2, column, projection='3d')
    ax.add_collection3d(Poly3DCollection(points[tet_face_indices], facecolors='cornflowerblue',
                                        edgecolors='navy', alpha=0.18, linewidths=1.1))
    ax.scatter(*points.T, color='navy', s=35)
    span = np.ptp(points, axis=0)
    for a, point in enumerate(points):
        offset = 0.025*span
        if column == 2:
            offset = np.array([[-0.002, -0.001, -0.004], [0.004, -0.002, -0.004],
                               [0.002, 0.003, 0.004], [0.002, 0.001, 0.004]])[a]
        ax.text(*(point+offset), f'{a+1}', color='navy', fontsize=12)
    low, high = points.min(axis=0)-0.12*span, points.max(axis=0)+0.18*span
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]), title=title)
    ax.set_box_aspect(span.copy(), zoom=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.view_init(elev=24, azim=-55 if column == 1 else -65)
    if column == 1:
        ax.set(xlabel=r'$\xi$', ylabel=r'$\eta$', zlabel=r'$\zeta$')
    else:
        ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')
        # Coordinates are printed below; omit crowded ticks in this sketch.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.yaxis.labelpad = 12
        ax.zaxis.labelpad = 12
fig.text(0.5, 0.025, 'Each node carries [ux, uy, uz]. Transparent triangular faces enclose a solid volume.',
         ha='center', fontsize=10)
fig.text(0.5, 0.065, 'Actual element, local → global labels: '
         + ', '.join(f'{a+1} → {label}' for a, label in enumerate(example_connectivity)),
         ha='center', fontsize=10)
fig.tight_layout(rect=(0., 0.08, 1., 0.95))
plt.show()
print('Example T4 connectivity (local nodes 1–4 -> global node rows):', example_connectivity)
print('Example T4 node coordinates [m]:\n', example_tet)

# %% [markdown]
# ## Three-dimensional elasticity and a constant-strain tetrahedron
#
# Strain order is [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz].
# Stress order is [sxx, syy, szz, tau_xy, tau_yz, tau_xz].
# With Lamé constants lambda and G, the normal block has lambda everywhere
# plus 2G on its diagonal; the engineering-shear diagonal is G.
# N_i=a_i+b_i*x+c_i*y+d_i*z comes from inverting [1,x_i,y_i,z_i].
# Its gradients and B are constant, so k_e = volume * B.T @ D @ B is exact.
# These T4 elements tend to be stiff in bending; refine before trusting a
# cantilever displacement. Nearly incompressible material (nu near 0.5)
# requires a different formulation to avoid volumetric locking.
#
# **Shapes and mapping.** In the reference tetrahedron,
# $$N_1=1-\xi-\eta-\zeta,\quad N_2=\xi,\quad N_3=\eta,\quad N_4=\zeta,$$
# with $\xi,\eta,\zeta\ge0$ and $\xi+\eta+\zeta\le1$.
# They satisfy $\sum_aN_a=1$ and $N_a(\mathbf{x}_b)=\delta_{ab}$.
# Geometry and displacement interpolate as $\mathbf{x}=\sum_aN_a\mathbf{x}_a$
# and $\mathbf{u}(\mathbf{x})=\sum_aN_a\mathbf{u}_a$.
# An equivalent physical-coordinate formula, used in the code, is
# $$C=\begin{bmatrix}1&x_1&y_1&z_1\\1&x_2&y_2&z_2\\
# 1&x_3&y_3&z_3\\1&x_4&y_4&z_4\end{bmatrix},\quad
# [N_1,N_2,N_3,N_4]=[1,x,y,z]C^{-1},\quad V_e=\det C/6>0.$$
# Rows 2–4 of $C^{-1}$ hold the constant shape gradients.
#
# **Strain.** For $\mathbf{u}=[u,v,w]^T$, the engineering-strain vector is
# $$\boldsymbol\epsilon=[u_{,x},v_{,y},w_{,z},u_{,y}+v_{,x},
# v_{,z}+w_{,y},u_{,z}+w_{,x}]^T=B\mathbf{u}_e.$$
# Stack $\mathbf{u}_e=[\mathbf{u}_1^T,\ldots,\mathbf{u}_4^T]^T$ and form
# $B=[B_1,B_2,B_3,B_4]$ (6×12), where for $p=N_{a,x}$, $q=N_{a,y}$, $r=N_{a,z}$,
# $$B_a=\begin{bmatrix}p&0&0\\0&q&0\\0&0&r\\q&p&0\\0&r&q\\r&0&p\end{bmatrix}.$$
# B is constant because T4 shape functions are affine.
#
# **Material.** The 3D isotropic law is $\boldsymbol\sigma=D\boldsymbol\epsilon$, with
# $$G=\frac{E}{2(1+\nu)},\quad\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},$$
# $$D=\begin{bmatrix}
# \lambda+2G&\lambda&\lambda&0&0&0\\
# \lambda&\lambda+2G&\lambda&0&0&0\\
# \lambda&\lambda&\lambda+2G&0&0&0\\
# 0&0&0&G&0&0\\0&0&0&0&G&0\\0&0&0&0&0&G
# \end{bmatrix}.$$
# The shear diagonal is G because the strain vector uses engineering shear
# $\gamma_{ij}=2\epsilon_{ij}$, not tensor shear $\epsilon_{ij}$.
#
# **Element stiffness.** Virtual work gives
# $\delta W_{int}=\int_{V_e}\delta\boldsymbol\epsilon^T\boldsymbol\sigma\,dV$.
# With the interpolation above, $\delta W_{int}=\delta\mathbf{u}_e^Tk_e\mathbf{u}_e$:
# $$k_e=\int_{V_e}B^TDB\,dV=V_eB^TDB,\quad
# U_e=\tfrac12\mathbf{u}_e^Tk_e\mathbf{u}_e.$$
# No numerical quadrature is needed for constant E, nu, and B. Coordinates
# and B already use global x,y,z, so no separate element rotation is required.

# %%
G = E/(2*(1+nu))
lame = E*nu/((1+nu)*(1-2*nu))
D = np.zeros((6, 6))
D[:3, :3] = lame
D[np.arange(3), np.arange(3)] += 2*G
D[3:, 3:] = G*np.eye(3)


def tetrahedron_B_volume(points):
    coordinates = np.column_stack([np.ones(4), points])
    volume = np.linalg.det(coordinates)/6
    assert volume > 0, 'A tetrahedron has nonpositive volume.'
    gradient = np.linalg.inv(coordinates)[1:, :].T
    B = np.zeros((6, 12))
    for a, (dx, dy, dz) in enumerate(gradient):
        i = 3*a
        B[0, i] = dx
        B[1, i+1] = dy
        B[2, i+2] = dz
        B[3, i:i+2] = [dy, dx]
        B[4, i+1:i+3] = [dz, dy]
        B[5, [i, i+2]] = [dz, dx]
    return B, volume


print('3D elastic material D [Pa]:\n', D)
print('Constant-strain tetrahedral B and volume function defined.')

# %% [markdown]
# ## Assemble the volume elements and check geometry and rigid motions
#
# If $L_e$ extracts the 12 element DOFs from the system vector,
# $$\mathbf{u}_e=L_e\mathbf{u},\quad K=\sum_eL_e^Tk_eL_e.$$
# The dofs array is the index representation of $L_e$; the rows and columns
# collect contributions to shared nodes. COO-to-CSR conversion sums duplicate
# entries. Positive volumes and $\sum_eV_e=A_hL$ verify filled volume.
# Energy makes each $k_e$ and K symmetric. For $\mathbf{u}_a=\mathbf{c}$ or
# $\mathbf{u}_a=\boldsymbol\omega\times\mathbf{x}_a$, $B\mathbf{u}_e=0$.
# These six rigid motions produce no strain; support constraints remove them
# from the assembled solve. Solids need no independent rotational DOFs.

# %%
rows, columns, values, element_data = [], [], [], []
for connectivity in elements:
    B, volume = tetrahedron_B_volume(nodes[connectivity])
    ke = volume*B.T @ D @ B
    dofs = (3*connectivity[:, None]+np.arange(3)).ravel()
    rows.extend(np.repeat(dofs, 12))
    columns.extend(np.tile(dofs, 12))
    values.extend(ke.ravel())
    element_data.append((B, volume, dofs))
K = coo_matrix((values, (rows, columns)), shape=(ndof, ndof)).tocsr()
volumes = np.array([data[1] for data in element_data])
assert np.isclose(volumes.sum(), area_mesh*length, rtol=1e-12)
assert np.max(np.abs((K-K.T).data), initial=0.) < 1e-12*np.max(np.abs(K.data))

# Rigid translations and rotations must have zero element strain.
B, _, _ = element_data[0]
points = nodes[elements[0]]
for direction in np.eye(3):
    assert np.linalg.norm(B @ np.tile(direction, 4)) < 1e-10
    assert np.linalg.norm(B @ np.cross(direction, points).ravel()) < 1e-10
print('Stiffness, positive volumes, total volume, symmetry, and rigid motions checked.')

# %% [markdown]
# ## Integrate the end-face traction and identify exterior faces
#
# A constant traction on a T3 face gives area*traction/3 to each of its
# three nodes. This keeps the resultant independent of mesh node count.
# Exterior triangle faces occur once in the tetrahedron connectivity;
# interior faces occur twice. We check the expected exterior count to
# catch incompatible prism splits that would leave internal cracks.
#
# The applied constant traction is $\boldsymbol\tau=[F_x,0,F_z]^T/A_h$ [N/m²].
# On a triangular end face with vertices $\mathbf{x}_1,\mathbf{x}_2,\mathbf{x}_3$,
# $$A_f=\tfrac12\|(\mathbf{x}_2-\mathbf{x}_1)\times(\mathbf{x}_3-\mathbf{x}_1)\|,
# \quad\mathbf{f}_{a,f}=\int_{A_f}N_a\boldsymbol\tau\,dA
# =\frac{A_f}{3}\boldsymbol\tau.$$
# Add the three vertex contributions to global loads; shared vertices receive
# contributions from several faces. Then $\sum_a\mathbf{f}_a=[F_x,0,F_z]^T$.
# In this example body force is zero; a constant body force $\mathbf{b}$ [N/m³]
# would give $V_e\mathbf{b}/4$ per tetrahedral vertex.
#
# For topology, let $c(f)$ count tetrahedra containing a given sorted face.
# An exterior face has $c(f)=1$, an interior face has $c(f)=2$.
# The expected exterior count is $2n_{disk\ triangles}+2n_cn_z$ (two ends
# plus the triangulated side surface); extra count-one faces indicate cracks.

# %%
def end_face_force(total_force):
    traction = np.asarray(total_force)/area_mesh
    force = np.zeros((len(nodes), 3))
    for triangle, area in zip(disk_triangles, triangle_areas):
        force[top[triangle]] += area/3*traction
    assert np.allclose(force.sum(axis=0), total_force)
    return force.ravel()


face_counter = Counter()
face_owner = {}
for e, tet in enumerate(elements):
    for local in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
        face = tuple(sorted(tet[local]))
        face_counter[face] += 1
        face_owner[face] = e
assert all(count in [1, 2] for count in face_counter.values())
surface_faces = np.array([face for face, count in face_counter.items() if count == 1])
surface_owner = np.array([face_owner[tuple(face)] for face in surface_faces])
expected_faces = 2*len(disk_triangles)+2*n_circumference*n_axial
assert len(surface_faces) == expected_faces, 'The volume mesh has internal exposed faces.'
print('Conforming exterior mesh checked; exterior triangles:', len(surface_faces))

# %% [markdown]
# ## Clamp the lower face, solve, and calculate support reactions
#
# Every translation on the lower face is zero. There are no independent
# nodal rotations in a solid element. Moment balance uses r × nodal force.
# Reaction = K u - f; the entries on free DOFs should be negligible.
#
# Let f denote free DOFs and c constrained DOFs. The lower-face clamp has
# $\mathbf{u}_c=0$ in the main case. Partition the equilibrium equations:
# $$\begin{bmatrix}K_{ff}&K_{fc}\\K_{cf}&K_{cc}\end{bmatrix}
# \begin{bmatrix}\mathbf{u}_f\\\mathbf{u}_c\end{bmatrix}
# =\begin{bmatrix}\mathbf{f}_f\\\mathbf{f}_c+\mathbf{r}_c\end{bmatrix},$$
# $$K_{ff}\mathbf{u}_f=\mathbf{f}_f-K_{fc}\mathbf{u}_c,\quad
# \mathbf{r}=K\mathbf{u}-\mathbf{f}.$$
# The same partition handles nonzero prescribed contraction in the patch test.
# Checks are $\mathbf{r}_f\approx0$,
# $\sum_a(\mathbf{f}_a+\mathbf{r}_a)=0$, and
# $\sum_a\mathbf{x}_a\times(\mathbf{f}_a+\mathbf{r}_a)=0$.
# Expected support force and moment about the origin are $[-F_x,0,-F_z]$
# and $[0,-LF_x,0]$. Since $\mathbf{u}_c=0$,
# $U=\tfrac12\mathbf{u}^TK\mathbf{u}=\tfrac12\mathbf{u}^T\mathbf{f}$.
#
# The mean tip translation is a face integral, not an unweighted node mean:
# $$\overline{\mathbf{u}}_{top}=\frac1{A_h}\int_{top}\mathbf{u}\,dA
# =\sum_a w_a\mathbf{u}_a,\quad
# w_a=\frac1{A_h}\sum_{f\ni a}\frac{A_f}{3},\quad\sum_aw_a=1.$$

# %%
def solve_solid(force, fixed, prescribed=None):
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
    total = (force+reaction).reshape(-1, 3)
    force_balance = total.sum(axis=0)
    moment_balance = np.cross(nodes, total).sum(axis=0)
    force_scale = max(np.linalg.norm(force.reshape(-1, 3), axis=1).sum(), 1.)
    assert np.linalg.norm(force_balance) < 1e-7*force_scale
    assert np.linalg.norm(moment_balance) < 1e-7*force_scale*max(length, radius)
    return displacement, reaction, relative_residual, force_balance, moment_balance


fixed = (3*bottom[:, None]+np.arange(3)).ravel()
force = end_face_force([F_transverse, 0., F_axial])
u, reaction, residual, force_balance, moment_balance = solve_solid(force, fixed)
# Area weights give a face average, rather than a node-count-weighted average.
top_weights = end_face_force([0., 0., 1.]).reshape(-1, 3)[top, 2]
tip_mean = top_weights @ u.reshape(-1, 3)[top]
support = reaction.reshape(-1, 3)[bottom]
support_moment = np.cross(nodes[bottom], support).sum(axis=0)
assert np.isclose(0.5*u @ (K @ u), 0.5*u @ force, rtol=1e-8)
print('Total applied [Fx, Fy, Fz] [N]:', force.reshape(-1, 3).sum(axis=0))
print('Mean upper-face displacement [ux, uy, uz] [mm]:', 1000*tip_mean)
print('Support resultant [N]:', support.sum(axis=0))
print('Support moment about origin [N m]:', support_moment)
print('Free residual / force balance / moment balance:', residual, force_balance, moment_balance)
print('Elastic strain energy [J]:', 0.5*u @ (K @ u))

# %% [markdown]
# ## Recover the constant stress in each tetrahedron
#
# sigma = D B u_e. The von Mises expression includes all three normal
# and shear stresses. Stress is discontinuous between T4 elements; we
# retain the element values instead of smoothing them into nodal values.
# A sharp clamped edge can produce mesh-sensitive stresses. The maximum
# here is an element sample, not a validated design stress at that edge.
#
# Extract $\mathbf{u}_e=L_e\mathbf{u}$; then
# $\boldsymbol\epsilon_e=B_e\mathbf{u}_e$ and
# $\boldsymbol\sigma_e=D\boldsymbol\epsilon_e$ are constant over each tetrahedron.
# For the six stress components in the declared order,
# $$\sigma_{vm}=\sqrt{\tfrac12[(\sigma_x-\sigma_y)^2+(\sigma_y-\sigma_z)^2
# +(\sigma_z-\sigma_x)^2]+3(\tau_{xy}^2+\tau_{yz}^2+\tau_{xz}^2)}.$$
# Surface and cutaway triangles are coloured by their owning tetrahedron's
# $\sigma_{vm}$. No new stress solution is calculated on the displayed faces.

# %%
def recover_solid(displacement):
    stress = np.array([D @ B @ displacement[dofs] for B, _, dofs in element_data])
    sx, sy, sz, txy, tyz, txz = stress.T
    von_mises = np.sqrt(0.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)
                        + 3*(txy**2+tyz**2+txz**2))
    return stress, von_mises


stress, von_mises = recover_solid(u)
print('Axial stress range [MPa]:', stress[:, 2].min()/1e6, stress[:, 2].max()/1e6)
print('Largest element von Mises stress [MPa]:', von_mises.max()/1e6)

# %% [markdown]
# ## Exact uniform-tension patch test and approximate beam comparison
#
# Uniform sigma_zz=F/A_mesh has u_z=sigma*z/E and
# [u_x,u_y]=-nu*sigma/E*[x,y]. Prescribe those lateral values on the lower
# face, allowing the exact Poisson contraction, and load only in +z.
# T4 interpolation exactly represents this affine displacement field.
# This checks a different support condition from the clamped main case.
# Euler–Bernoulli bending uses I=pi*R⁴/4 and is an approximate slender-beam
# comparison, not an exact 3D benchmark. Coarse T4 meshes can be too stiff.
#
# Set $\sigma_0=P/A_h$ and $\epsilon_0=\sigma_0/E$:
# $$\mathbf{u}_{exact}=[-\nu\epsilon_0x,-\nu\epsilon_0y,\epsilon_0z]^T.$$
# Then $\boldsymbol\epsilon=[-\nu\epsilon_0,-\nu\epsilon_0,\epsilon_0,0,0,0]^T$
# and $D\boldsymbol\epsilon=[0,0,\sigma_0,0,0,0]^T$. Impose this exact field
# on the lower face and apply uniform axial traction on the top. Because
# T4 exactly represents an affine field, the nodal error
# $e_u=\|\mathbf{u}_{FE}-\mathbf{u}_{exact}\|_2/\|\mathbf{u}_{exact}\|_2$
# should be near floating-point solver accuracy. This is not a bending test.
# For the circular cantilever guide,
# $$A_c=\pi R^2,\quad I_c=\pi R^4/4,\quad
# u_x(L)\approx\frac{F_xL^3}{3EI_c},\quad u_z(L)\approx\frac{F_zL}{EA_c}.$$
# Differences include mesh stiffness, polygon geometry, transverse shear,
# and the local effect of preventing Poisson contraction at the clamp.

# %%
patch_load = 100_000.0
sigma_exact = patch_load/area_mesh
exact_u = np.zeros_like(nodes)
exact_u[:, :2] = -nu*sigma_exact/E*nodes[:, :2]
exact_u[:, 2] = sigma_exact/E*nodes[:, 2]
patch_force = end_face_force([0., 0., patch_load])
patch_u, _, patch_residual, _, _ = solve_solid(patch_force, fixed, exact_u.ravel()[fixed])
patch_error = np.linalg.norm(patch_u-exact_u.ravel())/np.linalg.norm(exact_u)
patch_stress, _ = recover_solid(patch_u)
assert patch_error < 1e-8
assert np.allclose(patch_stress, [0., 0., sigma_exact, 0., 0., 0.], atol=1e-6*sigma_exact)
inertia_circle = np.pi*radius**4/4
beam_tip_x = F_transverse*length**3/(3*E*inertia_circle)
rod_tip_z = F_axial*length/(E*area_circle)
print('Exact tension tip extension [mm]:', 1000*sigma_exact*length/E)
print('Patch displacement relative error / residual:', patch_error, patch_residual)
print('Main FE / circular beam tip ux [mm]:', 1000*tip_mean[0], 1000*beam_tip_x)
print('Main FE / circular rod tip uz [mm]:', 1000*tip_mean[2], 1000*rod_tip_z)

# %% [markdown]
# ## Display the surface and a cut through the volume mesh
#
# The cut removes tetrahedra whose centroids have positive y. Its triangles
# show the stair-step boundary of the retained tetrahedra, not an exact
# planar slice. This exposes the filled interior. Colours use the stress
# of the owning tetrahedron without interpolation or smoothing.
# Try refining all three mesh directions and track bending displacement.
#
# **Which tetrahedra are visible?** For each solved element, compute its centroid
# $$\overline{\mathbf{x}}_e=\tfrac14\sum_{a=1}^4\mathbf{x}_{e,a}.$$
# The picture retains the index set $\mathcal{S}=\{e:\overline y_e<0\}$.
# Count faces within this subset and draw faces appearing once. These include
# the original outer surface and newly exposed internal triangular faces.
# Since entire tetrahedra are retained or hidden, the boundary does not lie
# exactly on $y=0$. This is a whole-element cutaway, not a geometric plane slice.
# Crucially, K, the loading, the solve, and the stresses use **all elements**;
# the cut changes only the picture. The model remains a filled cylinder.
#
# **Amplified deformation.** Plot $\mathbf{x}_{plot}=\mathbf{x}+s_d\mathbf{u}$,
# with $s_d=0.12L/\max_a\|\mathbf{u}_a\|_2$ (or 1 for zero displacement).
# Stress is never scaled by $s_d$. The cutaway uses original, undeformed coordinates.

# %%
centres = nodes[elements].mean(axis=1)
retained = np.flatnonzero(centres[:, 1] < 0.)
cut_counter, cut_owner = Counter(), {}
for e in retained:
    tet = elements[e]
    for local in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]:
        face = tuple(sorted(tet[local]))
        cut_counter[face] += 1
        cut_owner[face] = e
cut_faces = np.array([face for face, count in cut_counter.items() if count == 1])
cut_elements = np.array([cut_owner[tuple(face)] for face in cut_faces])


def plot_solid(ax, positions, faces, title, face_stress=None):
    collection = Poly3DCollection(positions[faces], edgecolors='0.4', linewidths=0.25)
    if face_stress is None:
        collection.set_facecolor('lightsteelblue')
    else:
        norm = colors.Normalize(vmin=0., vmax=max(von_mises.max()/1e6, 1e-12))
        collection.set_facecolor(plt.colormaps['viridis'](norm(face_stress/1e6)))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
                     shrink=0.6, label='Element von Mises [MPa]')
    ax.add_collection3d(collection)
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


displacement = u.reshape(-1, 3)
maximum = np.linalg.norm(displacement, axis=1).max()
scale = 0.12*length/maximum if maximum > 0 else 1.
fig = plt.figure(figsize=(15, 6))
plot_solid(fig.add_subplot(131, projection='3d'), nodes, surface_faces, 'Solid cylinder: exterior mesh')
plot_solid(fig.add_subplot(132, projection='3d'), nodes+scale*displacement,
           surface_faces, f'Deformed solid ({scale:.0f}×)', von_mises[surface_owner])
plot_solid(fig.add_subplot(133, projection='3d'), nodes, cut_faces,
           'Cutaway: tetrahedra fill the interior', von_mises[cut_elements])
fig.tight_layout(rect=(0., 0.02, 1., 0.92))
plt.show()
print('Exercise: compare the shell and solid under the same outer radius and total forces.')
