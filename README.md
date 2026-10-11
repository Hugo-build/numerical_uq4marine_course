# Numerical Method in Uncertainty Quantification of Marine Applications

A script-based teaching course connecting Python, linear algebra, probability,
finite elements, dynamics, metocean statistics, wave generation, and extreme responses.
The lessons in `pyscripts/` use `# %%` code cells and `# %% [markdown]` explanations.
Functions and model equations are defined inside each lesson so the calculations
can be followed directly.

## Repository structure

```text
pyscripts/                       Python lessons (run independently)
  01_vectors_matrices.py
  02_probability_models.py
  03-1_static_simple_truss.py
  03-2_static_simple_truss_monte_carlo.py
  03-3_3d_truss.py
  03-4_3d_hollow_cylinder.py
  03-5_3d_solid_cylinder.py
  04_linear_dynamics_daf.py
  05_stats_metocean.py
  06_wave_generation_jonswap.py
  07_short_term_extremes.py
  08_long_term_distributions.py
  40barTruss/                     Node and element CSVs for the 3D truss
documents/                      Setup guides and metocean model references
notebooks/                      Currently empty
pyproject.toml                  Dependencies and Python requirement
uv.lock                         Locked dependency versions
requirements.txt                Dependencies for pip/other environment managers
.python-version                 Python 3.12 for the uv workflow
course_manifest.json            Earlier notebook curriculum metadata
validation_report.json          Earlier notebook execution results
```

The current lessons are Python scripts; no `.ipynb` notebooks or pre-generated
HTML previews are included. The manifest and validation report describe the
earlier notebook set, rather than the current script inventory or its validation.

## Open and run

Run commands from the repository root. Python 3.10+ is required; the uv setup
selects Python 3.12. For installation steps and a comparison with Miniconda, see
[Install Python with uv or Miniconda](documents/python_install_uv_miniconda.md).
For shell and version-control basics, see
[Shell, Git, and GitHub basics](documents/shell_git_github_basics.md).

### Set up with uv

Create `.venv` and install the locked dependencies, then run the first lesson:

```sh
uv sync --locked
uv run python pyscripts/01_vectors_matrices.py
```

Run another lesson by substituting its filename, for example:

```sh
uv run python pyscripts/03-1_static_simple_truss.py
uv run python pyscripts/05_stats_metocean.py
```

The scripts print numerical results and display Matplotlib figures. Depending
on the plotting backend, close a figure window to let execution continue.
For a headless run without plot windows, use:

```sh
MPLBACKEND=Agg uv run python pyscripts/01_vectors_matrices.py
```

### Set up with pip

On macOS or Linux, an alternative is:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pyscripts/01_vectors_matrices.py
```

`pyproject.toml` defines the dependencies and `uv.lock` records their resolved
versions. Keep `requirements.txt` aligned with `pyproject.toml` when changing
dependency ranges. The local `.venv` directory is ignored by Git.

### Run teaching cells in an editor

In VS Code, select `.venv/bin/python` as the interpreter, open a lesson in
`pyscripts/`, and use **Run Cell** above each `# %%` section with the Python and
Jupyter extensions installed. Spyder also supports these cell markers.
Run cells from top to bottom in a fresh session. Each script defines its own
model functions and does not require another lesson to run first.

For `03-3_3d_truss.py`, whole-file execution is the simplest option: it locates
the geometry relative to the script. If an interactive session does not define
`__file__`, use `pyscripts/` as its working directory so the `40barTruss/` fallback
can find the CSVs.

JupyterLab is included in the dependencies and can be launched with
`uv run jupyter lab`; the supplied lesson files remain `.py` scripts.
Random seeds are fixed in the sampling examples. Exact sampled values can differ
slightly across scientific-library versions.

## Curriculum

Follow the filenames in order. Section 03 groups the mechanics and static
uncertainty lessons; sections 05–08 cover environmental statistics, waves,
and response extremes. Some internal script titles retain earlier lesson numbers;
the filenames below define the current order.

| Script | Main content | Verification or comparison |
|---|---|---|
| [01 · Vectors and matrices](pyscripts/01_vectors_matrices.py) | Array shapes, vector algebra, matrix products, linear solves, eigenpairs, least squares | Residual and eigenpair checks |
| [02 · Probability models](pyscripts/02_probability_models.py) | Normal, Rayleigh, Rayleigh-n maxima, Gumbel, Weibull, lognormal, exponential; sampling and fitting | Exact maximum quantiles, bootstrap uncertainty, distribution identities |
| [03-1 · Static bars, truss, and beam](pyscripts/03-1_static_simple_truss.py) | Axial stiffness, 2D truss assembly, Euler–Bernoulli beam bending | Equilibrium and analytical cantilever solution |
| [03-2 · Static truss Monte Carlo](pyscripts/03-2_static_simple_truss_monte_carlo.py) | Uncertain load, modulus, and area; displacement distributions, exceedance, convergence | Exact lognormal response benchmark |
| [03-3 · 3D truss](pyscripts/03-3_3d_truss.py) | Geometry from CSVs, local/global matrices, bracing, static response, modes, Newmark-beta and RK4 | Stability, modal energy, integrator agreement, time-step refinement |
| [03-4 · Hollow cylinder](pyscripts/03-4_3d_hollow_cylinder.py) | Flat Q4 shell panels, membrane and bending action, shear, wall-face stress | Rigid motions, equilibrium, energy, uniform-tension patch test |
| [03-5 · Solid cylinder](pyscripts/03-5_3d_solid_cylinder.py) | T4 volume mesh, 3D elasticity, end-face traction, stress and cutaway plots | Volume, rigid motions, equilibrium, energy, uniform-tension patch test |
| [04 · Linear dynamics and DAF](pyscripts/04_linear_dynamics_daf.py) | SDOF response, dynamic amplification, phase, transients, free decay | Harmonic amplitude and phase, energy |
| [05 · Metocean statistics](pyscripts/05_stats_metocean.py) | Site 14 wind–Hs–Tp model, conditional density heatmaps, 50-year IFORM surface and wind slices | Captured probability, invalid model domain, transform round trips |
| [06 · JONSWAP wave generation](pyscripts/06_wave_generation_jonswap.py) | Station histories, long-crested 2D and directional 3D animations, HTML/CSV export | Spectral energy and space-time consistency |
| [07 · Short-term extremes](pyscripts/07_short_term_extremes.py) | Gaussian dynamic response, positive record maxima, Gumbel and peak approximations | PSD variance, upcrossings, bootstrap, one-record time refinement |
| [08 · Long-term distributions](pyscripts/08_long_term_distributions.py) | State-conditioned response, mixtures, block and annual maxima, return levels | Inverse sampling and annual CDF identity |

Prerequisite: variables, loops, functions, and basic differentiation. Matrices
and probability are introduced in the first two lessons. Mechanics examples use
SI units and linear elastic assumptions. Assertions are mathematical checks,
not a replacement for understanding the physical model.

### Three-dimensional mechanics examples

[03-3 · Space truss dynamics](pyscripts/03-3_3d_truss.py) loads the
geometry in `pyscripts/40barTruss`, checks stability, solves static response and
natural modes, and compares Newmark-beta with RK4 and a refined time step.
Standard local element stiffness and mass matrices are defined in a separate
function and rotated into global coordinates before assembly.
The original axial-only 40-member model has eight mechanisms with nodes 1–4
fixed. The script explicitly adds eight side braces for a stable 48-member
teaching extension and uses assumed steel properties in SI units. Run it with `uv run python pyscripts/03-3_3d_truss.py`.

Two further standalone scripts compare static response of hollow and filled
cylinders. Both use a 0.20 m outer radius, 2 m length, steel elasticity, a
clamped lower end, and distributed upper-end loads of 100 kN axial and
10 kN transverse:

- [03-4 · Hollow cylinder with flat shell elements](pyscripts/03-4_3d_hollow_cylinder.py)
  meshes the middle surface with plane Q4 panels and a 5 mm wall thickness.
  Each node has three translations and three rotations. The visible element
  equations combine plane-stress membrane action, Mindlin bending, MITC4
  transverse shear for rectangular panels, and a weak drilling constraint.
  Plots show the shell mesh, amplified displacement, and wall-face stress.
- [03-5 · Filled cylinder with solid elements](pyscripts/03-5_3d_solid_cylinder.py)
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
buckling, and plasticity. Both use the existing course dependencies.

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
uv run python pyscripts/03-4_3d_hollow_cylinder.py
uv run python pyscripts/03-5_3d_solid_cylinder.py
```

Shell formulation background: Bathe and Dvorkin,
[A four-node plate bending element based on Mindlin/Reissner plate theory and a mixed interpolation](https://doi.org/10.1002/nme.1620210213)
(1985). The script uses its rectangular assumed-shear approach within an
educational flat-panel shell, rather than implementing a general curved shell.

### Metocean statistics

[05_stats_metocean.py](pyscripts/05_stats_metocean.py) implements the Site 14
(Norway 5) joint wind–wave-height–wave-period model from Li, Gao, and Moan's
2013 paper. Wind speed is the one-hour mean at 10 m height. The figures are
analytical density heatmaps and IFORM contours, rather than measured observations.
The script masks invalid fitted domains and checks the probability represented
by the plotted ranges. Model notes, coefficients for all five sites, and the
source PDF are in [documents/five_site_models/](documents/five_site_models/).
The environmental contour does not by itself predict a structure's 50-year response.

### JONSWAP wave generation

[06_wave_generation_jonswap.py](pyscripts/06_wave_generation_jonswap.py) generates
three fixed-station elevation records, a long-crested 2D wave-profile animation,
and a directional 3D sea-surface animation. The script uses the course dependencies
and can be run in teaching cells or with:

```sh
uv run python pyscripts/06_wave_generation_jonswap.py
```

Running the script creates `previews/06_wave_generation_jonswap/`. After
execution, open its `index.html` in a browser for offline animation players with
play/pause and frame sliders, spectrum and time-history plots, and CSV station data. The fourth CSV history
is the directional realization shown in the 3D animation. No FFmpeg is required.
Change sea-state, random seed, direction weights, grids and export settings near
the top of the script, then rerun all cells. HTML export renders every frame and
can take a minute. For batch runs, set `show_figures = False`.

The one-sided spectrum uses Hz and is normalized over its retained band to
`Hm0 = 4 sqrt(m0) = Hs`. This is a linear deep-water model. Finite directional
records can have a measured `4 std(eta)` different from the spectral Hs; phases
from different directions can reinforce or cancel at a station. The clip is
shorter than the realization's repeat period, so replay jumps back to its start.

## Teaching conventions

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
The DAF and response-extreme lessons use a single oscillator; the 3D truss lesson
includes multiple degrees of freedom and natural modes. A useful extension is
modal reduction of a beam with a consistent mass matrix.

The short- and long-term response examples use an artificial Gaussian force PSD,
not measured environmental data or a wave-force model. They run independently
of the JONSWAP wave-generation script; no wave-to-force conversion connects them.
Normally distributed Fourier coefficients give a Gaussian ensemble; fixed-amplitude random-phase synthesis would not
be exactly Gaussian. The FFT synthesis is periodic with the record duration and approximates a continuous spectrum.
Vary duration, time step, spectral bandwidth, and number of records to assess the numerical and statistical effects.
The time-grid check in script 07 is one illustrative record, not a full convergence study.

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
explicitly in the scripts so students can follow the calculations without external code.
