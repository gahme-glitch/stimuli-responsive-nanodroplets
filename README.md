# Surface-Adsorbed Nanodroplets of Symmetric Diblock Copolymers Form Versatile and Stimuli-Responsive Nanostructures

[![DOI](https://zenodo.org/badge/DOI/XXXXXXX.svg)](https://doi.org/XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-consistent field theory (SCFT) code accompanying the manuscript by **Petrov, A.** *et al.* (2026).

`SCFT.c` is a real-space, pseudo-spectral SCFT solver for **multi-species, multi-chemistry blends of linear homopolymers and copolymers**, in **bulk** (periodic) or **thin-film** (confined, brush-coated) geometry.

---

## How to set up a run

Every run is defined in [parameters.h](parameters.h).

- `LX`, `LY`, `LZ` and `DR_TARGET` — cell size in R_g and target grid spacing. A dimension set to `1.0` collapses to a single grid point. The 2D runs of the paper use `LY = 1.0`.
- `ITER` — maximum number of iterations for the SCFT solver.
- `RELAXATION` — number of iterations where noise is added to the system.
- `VOL_DBK_1..4`, `VOL_HOMOPOLYMER_A..J` — volume fraction of each polymer. **These must sum to exactly 1.0.**
- `F_INPUT_DBK_n` — block fraction `f` of each diblock (0.5 = symmetric).
- `LEN_DBK_n`, `LEN_HOMO_x` — chain length relative to the reference chain `NS`.
- `CHI_N_XY` — the interaction matrix.
- `G1` — evolution step for the simulation.

### Bulk or thin film

`MASK` selects the geometry:

- **`MASK 0` — bulk.** Periodic boundary conditions in all directions.
- **`MASK 1` — thin film.** Dirichlet boundary conditions in `z` and periodic in `x`/`y`. Enables the brushes on the substrate (`z = 0`), and their volume fraction `VOL_S1+VOL_S2+VOL_S3` is subtracted from the volume fraction of free polymers, which are rescaled by `(1 − (VOL_S1+VOL_S2+VOL_S3))`.

## Output files

| File | Contents |
|---|---|
| `densities_dbk_1XXXXXXX.dat` | snapshot: `i j k`, the 13 density fields (`ϕ_A…ϕ_J`, `ϕ_S1…ϕ_S3`), the 13 w-fields, and `p` - pressure field |
| `free_energy.dat` | iteration, free energy, noise amplitude |
| `energies.dat` | -ln(single chain partition function) for every molecule species |
| `deviations.dat` | RMS deviation of `ϕ_tot` from 1, and the effective `χN` values |
| `run.txt` | all parameters of the run, appended at launch |
| `restart_fields.dat`, `restart_random.dat`, `counter.dat` | restart state |
| `DENSITY_CHECK_FAIL.error` | written only if the volume fractions do not sum to 1 |


`D_WRT` - sets the frequency at which density snapshots are written to disk.
`D_WRT_ENERGIES` - sets the frequency at which `free energy`, `energies`, and `deviations` are written to disk.
`counter.dat` - if set to `0`, the simulations starts from an initial random state. If contains a non-zero value, the simulation will restart from the last saved state (`restart_fields.dat` and `restart_random.dat`).

## Build and run

Requires a C compiler and [FFTW3](http://www.fftw.org/). 

```bash
icc -g -Ofast -ipo SCFT.c -o SCFT.out -lfftw3
```



# ICA Pseudocode: Iterative Phase-Diagram Construction Workflow

This section, in language-agnostic pseudocode, describes the iterative algorithm used to
build the phase diagrams. It is meant to let a reader reproduce the workflow in any programming language. 

## 1. Grid, planes, and points

The phase diagram is a 2D grid over the interaction parameters `(chi_AB, chi_AC)`. The
grid is duplicated across three parallel planes, one per starting contact angle:

- `P0`: near-90-degree contact angle
- `P1`: near-dewetted
- `P2`: near-wetted

A grid point is therefore indexed by `(plane, i, j)`, and **each grid point is one SCFT
simulation**: given `(chi_AB, chi_AC)`, other interaction parameters yielding a given contact angle (further: "plane" parameters), and a starting disordered droplet state, the solver relaxes to a
converged field and reports a free energy `F(plane, i, j)`.

Each point has up to six neighbors: four in-plane (`i±1, j` and `i, j±1`) and two
cross-plane, connecting the same `(i, j)` across the cyclic sequence
`P0 -> P1 -> P2 -> P0`.

Every point is also backed by a persistent record, stored as one JSON file per
`(plane, i, j)`, holding its current state: the converged free energy, the path to the
simulation that produced it, and its neighbor relationships. `STATE[plane, i, j]` below
refers to this record.

## 2. First iteration: initial state

```
FOR EACH plane in {P0, P1, P2}:
    FOR EACH (i, j) in grid:
        run an SCFT simulation at parameters (chi_AB[i], chi_AC[j], plane)
        STATE[plane, i, j] = (converged field, F(plane, i, j))
```

This first pass — one independent SCFT run per grid point — is the initial state of the
phase diagram. All later iterations only modify entries of `STATE`.

## 3. Update rule: comparing free energies

At each subsequent iteration, every point tests its neighbors' structures against its
own, and keeps whichever has the lowest free energy.

```
REPEAT:
    updated = False
    FOR EACH (plane, i, j):
        FOR EACH neighbor of (plane, i, j) not yet tried at this point:
            run a trial SCFT simulation at (chi_AB, chi_AC, plane) of (plane, i, j),
                seeded with neighbor's converged field
            F_trial = resulting free energy

            mark neighbor as tried at (plane, i, j)
            IF F_trial < STATE[plane, i, j].F:
                STATE[plane, i, j] = (trial's converged field, F_trial)
                updated = True
UNTIL updated == False
```

The loop terminates once a full pass over the grid produces no lower-energy replacement anywhere.

## 4. State history

Every time `STATE[plane, i, j]` is replaced by a winning trial, the previous entry is not
discarded but appended to that point's JSON record instead of being overwritten. Each
record therefore accumulates the full evolution of that grid point: the ordered sequence
of `(iteration, free energy, simulation path)` it passed through from the initial state
(Section 2) to its current, final value.

