# Project EquSpace — Tight-binding model of Bi donors in silicon

Atomistic tight-binding (TB) calculations for **substitutional bismuth (Bi) donors in silicon**, developed as early groundwork for the [EquSpace](https://equspace.eu/en) project. The long-term goal is to describe how **hydrostatic strain** modifies the Si:Bi donor spin qubit (binding energy, wavefunction, contact hyperfine interaction), following the TB donor model of Mansir *et al.*, PRL **120**, 167701 (2018).

The current code covers the **unstrained** case: bulk silicon, a pristine silicon reference box, and a Bi donor embedded in a periodic silicon supercell. Strain will be added on top of this validated baseline.

---

## Physics model

### Silicon host
- Orthogonal, nearest-neighbour **sp³d⁵s\*** tight binding, parameters from Niquet *et al.*, PRB **79**, 245201 (2009), Table I.
- 10 orbitals per atom: `s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2, s*` → 20 per atom with spin.
- On-site **spin–orbit coupling** `H_SO = λ L·σ` on the p manifold (λ = 0.01851 eV).
- Slater–Koster integrals are built in a local bond frame and rotated to the lab frame (no hand-written angular tables).
- Diamond lattice: 2-atom primitive cell for bulk bands, 8-atom conventional cubic cell for real-space boxes (a = 4·d₀/√3, d₀ = 2.35169 Å).

### Bi donor (Mansir *et al.*, Supplemental S1–S3)
- **S1 — long-range tail:** screened Coulomb potential `−e²ε(r)/(4πε₀ κ r)` with κ = 11.7, added to the on-site energies of all host atoms.
- **S2 — central cell:** at the Bi site the singular tail is replaced by orbital-dependent shifts `U_Bi` (s, s\*: 5.862 eV; p: 3.690 eV) and an enhanced spin–orbit parameter λ_Bi = 0.350 eV.
- **S3 — contact hyperfine proxy:** `|s(0) + 0.058 s*(0)|²` at the donor site (relative units, not absolute MHz).
- Host–host bonds are left unmodified (no lattice relaxation yet).

---

## Repository layout

```
silicon_tb_unstrained/
├── Si_bands/                      # Step 1: bulk silicon
│   ├── silicon_unstrained.py      #   core TB module (onsite, hopping, Bloch H, real-space H)
│   ├── plot_silicon_bands.ipynb   #   band structure + lattice/orbital schematic
│   ├── silicon_bands.png
│   └── si_unitcell_orbitals.png
├── pure_si/                       # Step 2: pristine reference box (+ updated donor script)
│   ├── silicon_pure_box.py        #   box band edges, folded vs sparse cross-check
│   ├── silicon_bi_donor.py        #   donor script with parameter block, open/periodic axes
│   ├── plot_silicon_bands.ipynb   #   bands + box levels + donor levels
│   └── silicon_bands.png / .svg, si_unitcell_orbitals.png
└── Bi_donor/                      # Step 3: Bi donor in a periodic box
    ├── silicon_unstrained.py      #   copy of the core TB module
    ├── silicon_bi_donor.py        #   donor Hamiltonian + shift-invert solver
    └── results/bi_donor.dat, bi_donor.npz
```

### Core module: `silicon_unstrained.py`

| Function | Purpose |
|---|---|
| `onsite(spin)` | 10×10 (or 20×20 with SOC) atomic on-site block |
| `hopping(d, spin)` | Slater–Koster hopping block for a bond vector `d` |
| `bloch_hamiltonian(k, spin)` | 40×40 (20×20) bulk Bloch Hamiltonian, k in Å⁻¹ |
| `make_real_space(ncells, periodic, spin)` | Sparse CSR Hamiltonian + atom positions for an `Nx×Ny×Nz` conventional-cell box |

---

## Workflow

```
 ┌───────────────────┐     ┌────────────────────┐     ┌────────────────────┐     ┌──────────────────┐
 │ 1. Bulk Si bands  │ ──▶ │ 2. Pure-Si box     │ ──▶ │ 3. Bi donor box    │ ──▶ │ 4. Strain (next) │
 │  validate TB      │     │  box band edges    │     │  bound states,     │     │  hydrostatic     │
 │  parameters       │     │  E_v^box, E_c^box  │     │  E_B, hyperfine    │     │  strain response │
 └───────────────────┘     └────────────────────┘     └────────────────────┘     └──────────────────┘
```

### Step 1 — Bulk silicon band structure (`Si_bands/`)
Diagonalise `bloch_hamiltonian(k)` along a high-symmetry path and check the gap, the Δ-valley conduction-band minimum (≈ 0.85 × 2π/a along Γ–X) and the spin–orbit split-off. Run `plot_silicon_bands.ipynb` to produce `silicon_bands.png` and the lattice/orbital schematic `si_unitcell_orbitals.png`.

### Step 2 — Pristine reference box (`pure_si/silicon_pure_box.py`)
Builds exactly the same supercell as the donor run, but without Bi, to obtain the **box band edges** that donor binding energies must be measured from:

`E_B = E_c^box − E_donor`

Two independent routes are used for a fully periodic box and must agree, which also validates `make_real_space`:
- **folded** — exact: diagonalise the bulk Bloch Hamiltonian at the k-points folded into the box, `k = 2π(n₁/Nx, n₂/Ny, n₃/Nz)/a`;
- **sparse** — shift-invert + LOBPCG on the real-space Hamiltonian.

```bash
cd silicon_tb_unstrained/pure_si
python silicon_pure_box.py --ncells 6                 # box edges for a 6×6×6 box
python silicon_pure_box.py --scan 2 4 6 8 --no-sparse # convergence of E_c^box with box size
```

### Step 3 — Bi donor (`Bi_donor/` or `pure_si/silicon_bi_donor.py`)
1. `bulk_edges()` — locate the bulk VBM and Δ-valley CBM.
2. `make_donor()` — build the box, substitute Bi near the centre, add the minimum-image screened Coulomb tail to all host sites, insert the Bi central-cell block.
3. `solve_near()` — sparse shift-invert (`eigsh`) for the states nearest a target energy (default `CBM − 0.07 eV`).
4. Analyse each state: binding energy, weight on the Bi site, RMS radius, weight in the outer 20 % of the box (finite-size check), contact-hyperfine proxy, in-gap flag.

```bash
cd silicon_tb_unstrained/Bi_donor
python silicon_bi_donor.py --ncells 2                    # small test box
python silicon_bi_donor.py --ncells 4 --nstates 12 --save-vectors
python silicon_bi_donor.py --ncells 4 --build-only       # only report matrix size / memory
```
The `pure_si/` version additionally accepts `--ncells Nx Ny Nz` and `--periodic true false ...` and has an editable parameter block at the top of the file.

**Outputs** (`results/bi_donor.*`):
- `.dat` — columns `E_eV, CBM_minus_E_meV, residual_eV, P_Bi, rms_radius_A, outer_weight, contact_proxy, in_bulk_gap`
- `.npz` — energies, positions, per-atom probabilities, tail potential, band edges and run parameters (plus eigenvectors with `--save-vectors`)

### Step 4 — Planned
- Hydrostatic strain: bond-length scaling of the hopping integrals and strain-dependent on-site terms (the current `hopping()` deliberately ignores bond length).
- Larger boxes / convergence of the binding energy and valley-orbit splitting toward the isolated-donor limit.
- Ewald-type treatment of the periodic Coulomb tail; absolute hyperfine constant; surface passivation for open boundaries.

---

## Scaling and memory

With spin, each atom carries 20 basis states and each conventional cell 8 atoms:

| `ncells` | Atoms | Matrix dimension |
|---|---|---|
| 2 | 64 | 1 280 |
| 4 | 512 | 10 240 |
| 6 | 1 728 | 34 560 |
| 10 | 8 000 | 160 000 |

The Hamiltonian is always kept sparse (never convert large boxes to dense). Shift-invert uses a sparse LU factorisation whose memory can be much larger than the CSR matrix itself; use `--build-only` to check sizes before solving.

---

## Current limitations

- **Unstrained only**; no lattice relaxation around Bi.
- Periodic boxes use a **minimum-image** Coulomb tail, not an Ewald sum — small boxes are prototypes, not isolated-donor results (check `outer_weight`).
- Open boundaries are **unpassivated** and produce dangling-bond states in the gap.
- `solve_near` returns states nearest `σ`; it is not guaranteed to capture every bound state.
- The hyperfine output is a relative proxy, not an absolute value in MHz.
- No Zeeman term or nuclear hyperfine Hamiltonian yet.

---

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib` (for notebooks) and `jupyter`.

```bash
pip install numpy scipy matplotlib jupyter
```

Each script imports `silicon_unstrained.py` from its own directory. The scripts in `pure_si/` also need a copy of `silicon_unstrained.py` in that folder (it is currently only present in `Si_bands/` and `Bi_donor/`).

---

## References

1. Y.-M. Niquet, D. Rideau, C. Tavernier, H. Jaouen, X. Blase, *Onsite matrix elements of the tight-binding Hamiltonian of a strained crystal: Application to silicon, germanium, and their alloys*, Phys. Rev. B **79**, 245201 (2009).
2. J. Mansir *et al.*, *Linear hyperfine tuning of donor spins in silicon using hydrostatic strain*, Phys. Rev. Lett. **120**, 167701 (2018), and Supplemental Material.
3. EquSpace project — https://equspace.eu/en

## Author

Anirban Das — Budapest University of Technology and Economics (BME)
