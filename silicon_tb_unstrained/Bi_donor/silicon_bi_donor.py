"""Unstrained substitutional Bi: Mansir et al. PRL 120, 167701, SM S1-S3.
Requires the accompanying silicon_unstrained.py. Energies eV; lengths Angstrom.
Periodic minimum-image impurity potential, NOT an Ewald Coulomb sum.
Small-box shift-invert prototype; no strain or hyperfine Hamiltonian.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.constants import physical_constants, elementary_charge, epsilon_0
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from silicon_unstrained import A, LAMBDA_SO, onsite, bloch_hamiltonian, make_real_space

# ----------------------------------------------------------------------------
# PARAMETER BLOCK: edit here (command-line flags override these defaults)
# ----------------------------------------------------------------------------
NCELLS = 4             # conventional cells per axis: int, or (Nx, Ny, Nz); box = NCELLS*A
PERIODIC = True        # periodic boundaries: bool, or (px, py, pz) per axis
SPIN = True            # include spin-orbit / spin degrees of freedom
NSTATES = 12           # number of eigenstates returned by the solver
SIGMA = None           # shift-invert target in eV; None -> CBM - 0.07
OUT = 'results/bi_donor'
# ----------------------------------------------------------------------------

BOHR = physical_constants['Bohr radius'][0] * 1e10 #the Bohr radius in ångströms
COULOMB = elementary_charge / (4 * np.pi * epsilon_0) * 1e10  # the coulomb prefactor eV Angstrom
KAPPA = 11.7 # screening by silicon or dimensionless relative dielectric constant k = eps_si/eps_0
U_BI = np.array([5.862] + [3.690]*3 + [0.0]*5 + [5.862]) # orbital-dependent corrections at the Bi atom
LAMBDA_BI = 0.350 # Bi onsite spin-orbit parameter


def donor_tail(r):
    """S1 at r>0. Rates in inverse Bohr are converted using r/BOHR."""
    r = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('Coulomb tail requires finite positive radii')
    x = r / BOHR
    screening = (1 + 1.175*KAPPA*np.exp(-0.757*x)
                 + (1-1.175)*KAPPA*np.exp(-0.312*x) - np.exp(-2.044*x))
    return -COULOMB * screening / (KAPPA*r)


def bi_onsite(spin=True, lambda_bi=LAMBDA_BI):
    """S2 shifts REPLACE the singular tail at the donor site.

    Preserve the host convention Hsoc=lambda L.sigma. The appendix uses
    L.S notation but quotes the same host parameter ~0.0185 eV. Here 0.350
    is interpreted in that host convention; donor fine structure must be
    checked against experiment before claiming quantitative reproduction.
    """
    if not np.isfinite(lambda_bi) or lambda_bi < 0:
        raise ValueError('lambda_bi must be finite and nonnegative')
    if not spin:
        return onsite(False) - np.diag(U_BI)
    atomic = np.kron(onsite(False), np.eye(2))
    soc = onsite(True) - atomic
    return atomic - np.diag(np.repeat(U_BI, 2)) + soc*(lambda_bi/LAMBDA_SO)


def box_geometry(ncells, periodic):
    """Return (shape, lengths, pbc): cells per axis, box lengths (A), periodic flags."""
    shape = np.broadcast_to(np.asarray(ncells, dtype=float), (3,))
    if np.any(shape < 1) or np.any(shape != shape.astype(int)):
        raise ValueError('ncells must contain positive integers')
    shape = shape.astype(int)
    pbc = np.broadcast_to(np.asarray(periodic, dtype=bool), (3,))
    return shape, shape*A, pbc


def min_image(displacement, lengths, pbc):
    """Shortest equivalent displacement, applied only along periodic axes."""
    return displacement - np.where(pbc, lengths*np.rint(displacement/lengths), 0.0)


def make_donor(ncells=NCELLS, spin=SPIN, donor_index=None, lambda_bi=LAMBDA_BI,
               periodic=PERIODIC):
    """Box of ncells conventional cells; radial tail, unmodified host bonds.

    periodic: bool or (px, py, pz). Periodic axes use the nearest-image distance;
    open axes use the plain distance (open surfaces are unpassivated).
    Returns H, positions, donor_index, distances, scalar_tail.
    scalar_tail[donor_index]=0; the full Bi onsite block is inserted separately.
    """
    shape, lengths, pbc = box_geometry(ncells, periodic)
    h, positions = make_real_space(shape, periodic=pbc, spin=spin)

    
    if donor_index is None:
        # this finds the existing atom closest to the box center.
        donor_index = int(np.argmin(np.sum((positions-lengths/2)**2, axis=1)))
    if not isinstance(donor_index, (int, np.integer)) or not 0 <= donor_index < len(positions):
        raise ValueError('invalid donor atom index')

    #calculate displacements from the donor
    displacement = positions - positions[donor_index]
    displacement = min_image(displacement, lengths, pbc) # shortest equivalent displacement along periodic axes only
    radius = np.linalg.norm(displacement, axis=1) # calculates one distance per atom
    potential = np.zeros(len(positions))  # Initially, every atom has zero additional scalar potential
    
    # creates a Boolean mask identifying all atoms except the donor
    # Five atoms, with donor_index = 2:
    # away = [True, True, False, True, True]
    away = np.arange(len(positions)) != donor_index

    # evaluates the screened Coulomb expression only at those atoms.
    potential[away] = donor_tail(radius[away])

    norb = 20 if spin else 10 # Add those scalar potentials to the Hamiltonian
    h.setdiag(h.diagonal() + np.repeat(potential, norb)) # This modifies the diagonal entries of the full Hamiltonian.
    correction = bi_onsite(spin, lambda_bi) - onsite(spin) # Calculate the correction needed at the donor site
    row, col = np.nonzero(correction) # Locate the correction inside the full matrix
    offset = donor_index*norb # finds the starting global basis index of the donor atom.

    # Insert the correction as a sparse matrix
    # This constructs a sparse matrix with the same dimensions as h, whose only nonzero entries lie within the donor’s onsite block.
    h = h + coo_matrix((correction[row, col], (offset+row, offset+col)), shape=h.shape).tocsr()
    h.eliminate_zeros()  # removes any explicitly stored zeros from the sparse matrix.
    return h, positions, donor_index, radius, potential






def bulk_edges(spin=True):
    """Find the Delta-valley minimum along Gamma-X for this unstrained host.
    Assumes known silicon valley location; not a global Brillouin-zone search.
    fraction tells what percentage away the CBM from the gamma point.
    """
    occupied = 8 if spin else 4
    def conduction(t):
        return np.linalg.eigvalsh(bloch_hamiltonian([t*2*np.pi/A, 0, 0], spin))[occupied]
    grid = np.linspace(0, 1, 101)
    j = int(np.argmin([conduction(t) for t in grid]))
    result = minimize_scalar(conduction, bounds=(grid[max(0,j-1)], grid[min(100,j+1)]),
                             method='bounded', options={'xatol':1e-12})
    candidates = [(result.fun, result.x), (conduction(0), 0.), (conduction(1), 1.)]
    cbm, fraction = min(candidates)
    vbm = np.linalg.eigvalsh(bloch_hamiltonian([0,0,0], spin))[occupied-1]
    return float(vbm), float(cbm), float(fraction)


def solve_near(h, sigma, nstates=12, tol=1e-10):
    """States nearest sigma, NOT guaranteed to include every donor bound state.
    Sparse LU in shift-invert can use much more memory than h itself.
    """
    # if not 1 <= nstates < h.shape[0]-1 or not np.isfinite(sigma):
    #     raise ValueError('invalid nstates or sigma')
    # rng = np.random.default_rng(18)
    # v0 = rng.normal(size=h.shape[0])
    # if np.iscomplexobj(h.data):
    #     v0 = v0 + 1j*rng.normal(size=h.shape[0])

    energy, vectors = eigsh(h, k=nstates, sigma=sigma, which='LM', tol=tol,)
                            # ncv=min(h.shape[0]-1, max(2*nstates+1, 40)), v0=v0)
    order = np.argsort(energy)
    energy, vectors = energy[order], vectors[:, order]
    vectors /= np.linalg.norm(vectors, axis=0)
    residual = np.linalg.norm(h@vectors-vectors*energy, axis=0)
    # if np.max(residual) > 1e-7:
        # raise RuntimeError(f'eigenpair residual too large: {residual.max():.3g} eV')
    return energy, vectors, residual


def main():
    # command line interface
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ncells', type=int, nargs='+', default=None,
                        help='cells per axis: one value or three (Nx Ny Nz); default NCELLS')
    parser.add_argument('--periodic', choices=['true', 'false'], nargs='+', default=None,
                        help='boundary condition: one value or three (x y z); default PERIODIC')
    parser.add_argument('--spinless', action='store_true')
    parser.add_argument('--nstates', type=int, default=NSTATES)
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help='absolute target energy in eV; default CBM-0.07')
    parser.add_argument('--lambda-bi', type=float, default=LAMBDA_BI)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--save-vectors', action='store_true')
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args()
    spin = SPIN and not args.spinless
    ncells = NCELLS if args.ncells is None else args.ncells
    periodic = PERIODIC if args.periodic is None else [p == 'true' for p in args.periodic]
    if len(np.atleast_1d(ncells)) not in (1, 3) or len(np.atleast_1d(periodic)) not in (1, 3):
        parser.error('--ncells and --periodic take 1 or 3 values')
    shape, lengths, pbc = box_geometry(ncells, periodic)


    # main physics part starts from here
    vbm, cbm, valley = bulk_edges(spin)
    print(f'VBM={vbm:.9f}, CBM={cbm:.9f} eV; valley k/(2pi/a)={valley:.7f}', flush=True)
    print(f'Box {shape} cells = {lengths} Angstrom; periodic (x,y,z) = {tuple(bool(p) for p in pbc)}', flush=True)
    h, positions, donor, radius, potential = make_donor(shape, spin, lambda_bi=args.lambda_bi,
                                                        periodic=pbc)
    storage = (h.data.nbytes+h.indices.nbytes+h.indptr.nbytes)/2**20
    print(f'{len(positions)} atoms; H={h.shape}; CSR={storage:.2f} MiB', flush=True)
    print(f'Bi site {donor}, position {positions[donor]} Angstrom', flush=True)
    print('Small boxes are NOT isolated-donor results; open faces are unpassivated.', flush=True)
    if args.build_only:
        return
    sigma = cbm-0.07 if args.sigma is None else args.sigma
    print(f'Shift-invert target {sigma:.9f} eV; LU memory exceeds CSR storage.', flush=True)
    energy, vectors, residual = solve_near(h, sigma, args.nstates)
    norb = 20 if spin else 10
    probability = np.sum(np.abs(vectors.reshape(len(positions), norb, -1))**2, axis=1)
    # Outer 20% of the donor-centred box, along any axis.
    delta = min_image(positions-positions[donor], lengths, pbc)
    outer = np.any(np.abs(delta) >= 0.4*lengths, axis=1)
    outer_weight = probability[outer].sum(axis=0)
    rms_radius = np.sqrt((radius**2)@probability)
    # S3 proxy, in units of |s(0)|^2; NOT absolute hyperfine MHz.
    donor_amplitude = vectors.reshape(len(positions), norb, -1)[donor]
    if spin:
        contact = np.sum(np.abs(donor_amplitude[0:2]+0.058*donor_amplitude[18:20])**2, axis=0)
    else:
        contact = np.abs(donor_amplitude[0]+0.058*donor_amplitude[9])**2
    in_gap = (energy > vbm) & (energy < cbm)
    table = np.column_stack([energy, 1000*(cbm-energy), residual, probability[donor],
                             rms_radius, outer_weight, contact, in_gap.astype(int)])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(out)+'.dat', table,
               header='E_eV CBM_minus_E_meV residual_eV P_Bi rms_radius_A outer_weight contact_proxy in_bulk_gap')
    data = dict(energy_eV=energy, positions_A=positions, donor_index=donor,
                probability=probability, radius_A=radius, tail_eV=potential,
                residual_eV=residual, vbm_eV=vbm, cbm_eV=cbm, valley_fraction=valley,
                ncells=shape, periodic=pbc, spin=spin, sigma_eV=sigma, lambda_bi=args.lambda_bi,
                central_cell_U_eV=U_BI,
                boundary_model='minimum_image_on_periodic_axes',
                contact_proxy=contact, outer_weight=outer_weight)
    if args.save_vectors:
        data['eigenvectors'] = vectors
    np.savez_compressed(str(out)+'.npz', **data)
    print('E (eV)     CBM-E (meV)    residual (eV)   outer weight')
    for e, b, r, w in zip(energy, 1000*(cbm-energy), residual, outer_weight):
        print(f'{e: .8f} {b:12.5f} {r:14.3e} {w:12.5f}')
    print('In-gap energies are candidates, not proof of isolated donor localization.')
    print(f'Saved {out}.dat and {out}.npz')


if __name__ == '__main__':
    main()