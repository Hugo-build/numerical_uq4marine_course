# Numerical engineering: a notebook-first teaching course

Seven standalone lessons connect Python, linear algebra, probability, finite elements, and dynamics.
Every code cell prints a result or displays a plot. No command-line argument parsers, entry-point guards,
or hidden model code. Functions are defined visibly inside the relevant notebook.

## Open and run

1. Extract this archive.
2. Install Python 3.10+ and the dependencies listed in `requirements.txt` using your preferred environment manager.
3. Open `notebooks/01_vectors_matrices.ipynb` in JupyterLab, Jupyter Notebook, or VS Code.
4. Restart the kernel and run all cells from top to bottom. Each notebook runs independently.
5. Matching `scripts/*.py` files use `# %%` cells for VS Code or Spyder. They contain the same teaching code.

### Set up with uv

For installation steps and a comparison with Miniconda, see
[Install Python with uv or Miniconda](documents/python_install_uv_miniconda.md).

From the repository directory, create `.venv` and install the locked course dependencies:

```sh
uv sync --locked
uv run jupyter lab
```

`pyproject.toml` defines the dependencies, `uv.lock` records their resolved versions,
and `.python-version` selects Python 3.12. The local `.venv` directory is hidden
and ignored by Git. `requirements.txt` is retained for other environment managers;
keep its dependency ranges aligned with `pyproject.toml` when changing them.

For later sessions, run `uv run jupyter lab` again. To activate the environment
in your shell manually, run `source .venv/bin/activate`.
In VS Code, select `.venv/bin/python` as the Python interpreter and notebook kernel.
Run `deactivate` to leave the environment.

The delivered notebooks already contain results and plots. Browser-readable HTML previews are in `previews/`.
The previews display the lesson text, source code, printed results, and figures without needing Python.
Random seeds are fixed. Exact sampled values can differ slightly across scientific-library versions.

## Curriculum

| Lesson | Main content | Verification | Suggested class time |
|---|---|---|---|
| 01 | Array shapes, vector algebra, matrix products, linear solves, eigenpairs, least squares | Residual and eigenpair checks | 60–90 min |
| 02 | Normal, Rayleigh, Rayleigh-n maxima, Gumbel, Weibull, lognormal, exponential | Exact maximum quantiles and distribution identity | 90 min |
| 03 | Axial bars, 2D truss assembly, Euler–Bernoulli beam bending | Equilibrium and cantilever analytical solution | 120 min |
| 04 | SDOF dynamics, displacement DAF, phase, transients, energy | Time-domain amplitude against harmonic formula | 90 min |
| 05 | Static FE Monte Carlo, positive uncertain parameters, exceedance and convergence | Exact lognormal response benchmark | 90–120 min |
| 06 | Stationary Gaussian dynamic response, short-record maxima, Gumbel, peak approximations | PSD variance, upcrossings, bootstrap, one-record time refinement | 120 min |
| 07 | State-conditioned response, long-term mixture, block maxima, annual maxima, return periods | Inverse sampling and annual CDF identity | 120 min |

Suggested pace: seven sessions, each combining explanation, execution, parameter experiments, and discussion.
Prerequisite: variables, loops, functions, and basic differentiation. Matrices and probability are introduced here.
All mechanics examples use SI units and linear elastic assumptions. Assertions are lightweight mathematical checks,
not a replacement for understanding the physical model.

An additional script, [03-3 · Space truss dynamics](scripts/03-3_space_truss_FE_model.py),
adapts `testFE40barTruss_v2.py` to the same teaching-cell format. It loads the
geometry in `scripts/40barTruss`, checks stability, solves static response and
natural modes, and compares Newmark-beta with RK4 and a refined time step.
Standard local element stiffness and mass matrices are defined in a separate
function and rotated into global coordinates before assembly.
The original axial-only 40-member model has eight mechanisms with nodes 1–4
fixed. The script explicitly adds eight side braces for a stable 48-member
teaching extension and uses assumed steel properties in SI units. It is a
standalone script supplement; no matching notebook or HTML preview is included.
Run its cells in order or execute `uv run python scripts/03-3_space_truss_FE_model.py`.

Two further standalone scripts compare static response of hollow and filled
cylinders. Both use a 0.20 m outer radius, 2 m length, steel elasticity, a
clamped lower end, and distributed upper-end loads of 100 kN axial and
10 kN transverse:

- [03-4 · Hollow cylinder with flat shell elements](scripts/03-4_static_hollow_cylinder_shell.py)
  meshes the middle surface with plane Q4 panels and a 5 mm wall thickness.
  Each node has three translations and three rotations. The visible element
  equations combine plane-stress membrane action, Mindlin bending, MITC4
  transverse shear for rectangular panels, and a weak drilling constraint.
  Plots show the shell mesh, amplified displacement, and wall-face stress.
- [03-5 · Filled cylinder with solid elements](scripts/03-5_static_solid_cylinder.py)
  fills the volume with conforming four-node tetrahedra, with three
  translations per node. End-face tractions are integrated into consistent
  nodal forces. Plots show the exterior, amplified displacement, and a
  cutaway exposing the tetrahedral interior and element stress.

Both scripts check rigid motions, stiffness symmetry, force and moment
equilibrium, strain energy, and an exact uniform-tension patch test. The patch
test permits Poisson contraction at the lower end; it verifies a separate
load case from the fully clamped cantilever. Circular beam formulas provide
approximate comparisons for the cantilever, not exact 3D benchmarks. Refine
the mesh before interpreting bending displacement or clamp stresses;
linear tetrahedra can be overly stiff in bending. The shell script also
provides a drilling-stiffness sensitivity exercise. The cylinders have
polygonal boundaries, use linear elasticity, and exclude pressure,
buckling, and plasticity. No new dependencies, notebooks, or HTML previews
are needed for these script supplements.

Before the element equations, each script plots a labelled element sketch:
the shell panel with thickness and its Q4 reference square, or the T4
reference tetrahedron beside an actual volume-mesh element. The mathematical
notes cover geometry, interpolation, Jacobians, strain matrices, material
laws, element stiffness, assembly, consistent loading, constrained solves,
reactions, stress recovery, and verification. The solid cutaway notes also
explain how whole tetrahedra are hidden only for display; the analysis always
uses the complete filled cylinder.

Run their cells in order, or run:

```sh
uv run python scripts/03-4_static_hollow_cylinder_shell.py
uv run python scripts/03-5_static_solid_cylinder.py
```

Shell formulation background: Bathe and Dvorkin,
[A four-node plate bending element based on Mindlin/Reissner plate theory and a mixed interpolation](https://doi.org/10.1002/nme.1620210213)
(1985). The script uses its rectangular assumed-shear approach within an
educational flat-panel shell, rather than implementing a general curved shell.

## Teaching conventions

### JONSWAP wave generation script

[06_wave_generation_jonswap.py](pyscripts/06_wave_generation_jonswap.py) generates
three fixed-station elevation records, a long-crested 2D wave-profile animation,
and a directional 3D sea-surface animation. The script uses the course dependencies
and can be run in teaching cells or with:

```sh
uv run python pyscripts/06_wave_generation_jonswap.py
```

Open [the wave-generation preview](previews/06_wave_generation_jonswap/index.html)
in a browser for offline animation players with play/pause and frame sliders,
spectrum and time-history plots, and CSV station data. The fourth CSV history
is the directional realization shown in the 3D animation. No FFmpeg is required.
Change sea-state, random seed, direction weights, grids and export settings near
the top of the script, then rerun all cells. HTML export renders every frame and
can take a minute. For batch runs, set `show_figures = False`.

The one-sided spectrum uses Hz and is normalized over its retained band to
`Hm0 = 4 sqrt(m0) = Hs`. This is a linear deep-water model. Finite directional
records can have a measured `4 std(eta)` different from the spectral Hs; phases
from different directions can reinforce or cancel at a station. The clip is
shorter than the realization's repeat period, so replay jumps back to its start.

Each section follows: physical/mathematical question → readable code → visible result → interpretation or exercise.
Parameter assignments stay near the computation that uses them. Encourage students to predict a result before running it.
After modifying a parameter, rerun every downstream cell, or restart and run all.
Rerunning only a random sampling cell advances the RNG state; reproducibility requires rerunning the setup cell too.

“Rayleigh-n” means the maximum of n independent Rayleigh variates in this course. If the intended meaning is a chi
distribution with n degrees of freedom, use `scipy.stats.chi(df=n)` as a separate lesson model.
The ordinary Weibull here is `weibull_min`; it is not interchangeable with the bounded-tail GEV family.
SciPy's GEV shape `c` has the opposite sign to the common extreme-value parameter xi.

## Three different distributions

- Instantaneous response: one random response value, potentially conditioned on environment.
- Short-term maximum: maximum over a specified record duration, preserving temporal dependence.
- Annual maximum: maximum over an annual sequence of environmental blocks under an explicit aggregation model.

Shuffling a time history does not remove dependence from its joint law or justify independent sampling from its marginal
to predict original record maxima. Normal response marginals do not imply normally distributed maxima.
Rayleigh peaks are a narrow-band approximation; time samples are not independent Rayleigh peaks.
Gumbel fitting is a model choice and should be checked, not assumed universal.

## Scope and extensions

The beam is planar bending with two DOFs per node, not a full 2D frame element (which also includes axial DOFs).
The truss uses axial-only members. Neither model includes geometric instability or material nonlinearity.
Static uncertainty uses a common area multiplier; independent member areas change the response map.
Dynamics use a single oscillator. A useful next lesson is modal reduction of a beam with a consistent mass matrix.

The spectral examples use an artificial Gaussian force PSD, not measured environmental data or a wave-force model.
Normally distributed Fourier coefficients give a Gaussian ensemble; fixed-amplitude random-phase synthesis would not
be exactly Gaussian. The FFT synthesis is periodic with the record duration and approximates a continuous spectrum.
Vary duration, time step, spectral bandwidth, and number of records to assess the numerical and statistical effects.
The time-grid check in lesson 06 is one illustrative record, not a full convergence study.

Long-term states are synthetic. Independent equal-duration blocks are a teaching assumption. Weather persistence,
serial dependence, unequal state durations, nonstationarity, and model uncertainty need additional treatment.
The annual tail extrapolation is not validated by the inverse-sampling check. A robust extension compares Gumbel and GEV,
bootstraps within each state, studies dependence, and uses substantially more tail evidence.

## Instructor prompts

1. Explain `*`, `@`, outer products, and broadcasting before computing.
2. Predict how increasing n moves Rayleigh maximum quantiles.
3. Identify rigid-body modes before applying boundary conditions.
4. Separate the initial transient from steady harmonic response.
5. Compare sampling error with uncertainty in the physical inputs.
6. Explain why a time-sample marginal cannot by itself determine record maxima.
7. Explain why a mixture of block maxima is not an annual maximum distribution.

## Sources

- NumPy linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- SciPy probability distributions: https://docs.scipy.org/doc/scipy/reference/stats.html
- SciPy Rayleigh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html
- SciPy Weibull: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
- SciPy Gumbel: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html
- SciPy GEV and shape-sign convention: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
- SciPy ODE integration: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

Element matrices, oscillator transfer functions, Gaussian spectral moments, and mixture/maximum formulas are stated
explicitly in the notebooks so students can follow the calculations without external code.
