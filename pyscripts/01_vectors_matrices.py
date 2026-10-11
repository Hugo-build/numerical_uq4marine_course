# %% [markdown]
# # 01 · Vectors and matrices
# 
# **Goal:** build arrays, understand shapes, and solve linear systems. Prerequisite: basic Python.
# Run cells from top to bottom. A one-dimensional NumPy array is neither explicitly a row nor a column.
# Use `@` for matrix multiplication and `*` for entrywise multiplication. Never form an inverse just to solve a system.

# %%
import numpy as np                                            # NumPy provides vectors, matrices, and linear algebra operations.
import matplotlib.pyplot as plt                               # Matplotlib displays the fitted line and the vector transformation.
from scipy import stats, signal, integrate, optimize, linalg  # Import scientific tools; these SciPy modules are not used in this lesson.
np.set_printoptions(precision=5, suppress=True)               # Display array values with five decimal digits and suppress small-value scientific notation.
plt.rcParams.update({'figure.figsize': (8, 4), 'axes.grid': True,  # Set default plot size and show coordinate grids.
                     'grid.alpha': 0.25, 'font.size': 11})                     # Set grid transparency and label size; no mathematical operation.
rng = np.random.default_rng(2026)                                              # Fix the random seed so the simulated noise is reproducible.
print('Imports ready. Random seed: 2026. Units: SI unless stated otherwise.')  # Report the setup and default unit convention.

# %% [markdown]
# ## Create and inspect arrays
# 
# Compare lists, arrays, row vectors, column vectors, and useful constructors.

# %%
v = np.array([1., 2., 3.])            # v = (1, 2, 3); shape (3,) is a 1D vector.
row, column = v[None, :], v[:, None]  # Represent v as a 1 x 3 row and a 3 x 1 column by adding an axis.
A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])  # Define the symmetric 3 x 3 matrix A, with A_ij = A_ji.

print('Vector:', v, 'shape:', v.shape,'\n')               # Show v and its number of entries.
print('Row:', row, 'shape:', row.shape, '\n')             # Show the row-vector representation (1 x 3).
print('Column:\n', column, 'shape:', column.shape,'\n')   # Show the column-vector representation (3 x 1).
print('Matrix:\n', A,'\n')                                # Show the entries of A.
print('Identity:\n', np.eye(3),'\n')                                    # I_3 has ones on the diagonal and zeros elsewhere; I_3 v = v.
print('Zeros:', np.zeros(3), 'linspace:', np.linspace(0, 2.0, 5),'\n')  # Create (0, 0, 0) and five equally spaced values: 0, 0.5, 1, 1.5, 2.
print('Second row:', A[1, :], 'top-left block:\n', A[:2, :2],'\n')      # Zero-based indexing selects row 2 and the first two rows/columns of A.

# %% [markdown]
# ## Linear operations
# 
# A dot product gives a scalar; an outer product gives a matrix. Broadcasting is useful only when shapes express your intention.

# %%
w = np.array([2., -1., 4.])  # Define w = (2, -1, 4), another 3D vector.

print('v:', v, 'w:', w, '\n')                                                    # Display the two vectors used below.
print('v + w:', v+w, '2v:', 2*v, '\n')                                           # (v + w)_i = v_i + w_i; (2v)_i = 2v_i.
print('Entrywise v*w:', v*w, 'dot v@w:', v@w, '\n')                              # (v * w)_i = v_i w_i; v dot w = sum_i v_i w_i = 12.
print('Outer product:\n', np.outer(v, w), 'shape:', np.outer(v, w).shape, '\n')  # v w^T is a 3 x 3 matrix with entry (i, j) = v_i w_j.
print('A@v:', A@v, 'transpose:\n', A.T, 'shape:', A.T.shape, '\n')               # (Av)_i = sum_j A_ij v_j; (A^T)_ij = A_ji, and here A^T = A.
print('Norm:', np.linalg.norm(v), 'unit vector:', v/np.linalg.norm(v), '\n')     # ||v||_2 = sqrt(sum_i v_i^2) = sqrt(14); v / ||v||_2 has length 1.
print('Row scaling A * column:\n', A*column, 'shape:', (A*column).shape, '\n')   # Broadcast the column: result_ij = v_i A_ij, equivalent to diag(v) @ A.
print('Cross product:', np.cross(v, w), 'shape:', np.cross(v, w).shape, '\n')    # v cross w = (v_2 w_3 - v_3 w_2, v_3 w_1 - v_1 w_3, v_1 w_2 - v_2 w_1).



# %% [markdown]
# ## Solve and verify
# 
# Solve $Ax=b$. The residual measures how accurately the computed solution satisfies the equations; conditioning measures sensitivity to input perturbations.

# %%
b = np.array([1., 2., 3.])  # Define the right-hand side b in the system Ax = b.
x = np.linalg.solve(A, b)  # Compute x satisfying Ax = b using a linear solver.
residual = np.linalg.norm(A@x-b)  # Measure ||Ax - b||_2; a value near zero means the equations are satisfied.
print('Solution:', x, 'residual:', residual)  # Display the solution and its residual norm.
print('Rank:', np.linalg.matrix_rank(A), 'condition number:', np.linalg.cond(A))  # Rank counts independent columns; kappa_2(A) = sigma_max / sigma_min measures sensitivity.
assert residual < 1e-12  # Require the residual norm to be below the numerical tolerance 10^(-12).
eigenvalues, eigenvectors = np.linalg.eigh(A)  # For symmetric A, compute real lambda_j and orthonormal columns u_j with A u_j = lambda_j u_j.
print('Symmetric eigenvalues:', eigenvalues)  # Display the eigenvalues lambda_j.
print('Eigenvector residual:', np.linalg.norm(A@eigenvectors-eigenvectors*eigenvalues))  # Check ||AU - U diag(lambda)||_F; broadcasting scales each column u_j by lambda_j.

# %% [markdown]
# ## Least squares and a geometric plot
# 
# Fit a straight line to noisy observations. Overdetermined systems are solved with least squares.

# %%
t = np.linspace(0, 2, 30)  # Create 30 equally spaced samples t_i in [0, 2], with spacing 2/29.
y = 1.5 + 2*t + rng.normal(0, 0.2, t.size)  # Generate y_i = 1.5 + 2t_i + epsilon_i, with independent N(0, 0.2^2) noise.
design = np.column_stack([np.ones_like(t), t])  # Build D with row i = [1, t_i], so (D beta)_i = beta_0 + beta_1 t_i.
beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)  # Find beta minimizing ||D beta - y||_2^2; discard the other solver outputs.

print('Intercept and slope:', beta, 'RMSE:', np.sqrt(np.mean((design@beta-y)**2)))  # Report beta_0, beta_1 and RMSE = sqrt((1/n) sum_i ((D beta)_i - y_i)^2).

fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create two panels for the fitted line and the geometric mapping.
axes[0].scatter(t, y, label='observations')  # Plot the observed coordinate pairs (t_i, y_i).
axes[0].plot(t, design@beta, label='least squares')  # Plot the fitted line y_hat_i = beta_0 + beta_1 t_i.
axes[0].set(xlabel='t', ylabel='y'); axes[0].legend()  # Label the axes and identify the data and fitted line.
q = np.array([1., 0.5]); Aq = A[:2,:2]@q  # Apply the top-left 2 x 2 matrix to q = (1, 0.5), giving Aq = (4.5, 2.5).
for vec, label in [(q,'q'), (Aq,'Aq')]:  # Draw both the input vector q and its transformed vector Aq.
    axes[1].quiver(0, 0, *vec, angles='xy', scale_units='xy', scale=1, label=label)  # Draw an arrow from (0, 0) to (vec_1, vec_2) in coordinate units.
axes[1].set(xlim=(-1, 6), ylim=(-1, 4), xlabel='x', ylabel='y', title='Linear mapping')  # Set the visible coordinate range and label the linear transformation.
axes[1].set_aspect('equal'); axes[1].legend(); plt.tight_layout(); plt.show()  # Use equal axis scales to preserve vector geometry, arrange panels, and display.

