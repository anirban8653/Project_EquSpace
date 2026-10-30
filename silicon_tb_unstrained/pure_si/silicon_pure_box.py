"""Pure-Si reference box: same geometry as silicon_bi_donor.py, without Bi.

Gives the box band edges E_v^box and E_c^box that donor binding energies must
be measured from (E_B = E_c^box - E_donor), plus the degeneracy and k-point
origin of the lowest box conduction states.

Two independent routes for a fully periodic box:
  * folded  - exact: a periodic box of Nx*Ny*Nz conventional cells holds only
              the bulk Bloch states with k = 2pi (n1/Nx, n2/Ny, n3/Nz)/a.
              One 40x40 diagonalisation per k-point (4*Nx*Ny*Nz of them).
  * sparse  - shift-invert + LOBPCG on the real-space H from make_real_space,
              i.e. literally the same matrix as the donor run minus the Bi terms.
They must agree; the check validates make_real_space as well.
Open (non-periodic) axes: only the sparse route; surfaces are unpassivated,
so dangling-bond states will appear in the gap.
Energies eV; lengths Angstrom; k in units of 2pi/a.
"""
import argparse
import itertools
import warnings
from pathlib import Path
import numpy as np
from silicon_unstrained import A, BONDS, onsite, hopping, make_real_space, bloch_hamiltonian
from scipy.sparse import identity
from scipy.sparse.linalg import splu, LinearOperator, lobpcg



# ----------------------------------------------------------------------------
# PARAMETER BLOCK: edit here (command-line flags override these defaults)
# ----------------------------------------------------------------------------
NCELLS = 6             # conventional cells per axis: int, or (Nx, Ny, Nz)
PERIODIC = True        # periodic boundaries: bool, or (px, py, pz)
SPIN = True            # include spin-orbit / spin degrees of freedom
NLEVELS = 24           # states requested on each side of the gap (sparse route)
DEGEN_TOL = 1e-6       # eV; levels closer than this are grouped as degenerate
OUT = 'results/si_box'
# ----------------------------------------------------------------------------




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


def box_geometry(ncells, periodic):
    """Return (shape, lengths, pbc): cells per axis, box lengths (A), periodic flags."""
    shape = np.broadcast_to(np.asarray(ncells, dtype=float), (3,))
    if np.any(shape < 1) or np.any(shape != shape.astype(int)):
        raise ValueError('ncells must contain positive integers')
    shape = shape.astype(int)
    pbc = np.broadcast_to(np.asarray(periodic, dtype=bool), (3,))
    return shape, shape*A, pbc


def bloch_batch(frac, spin=True):
    """Eigenvalues of the primitive-cell Bloch H for many k (units 2pi/a).
    Same gauge and matrices as silicon_unstrained.bloch_hamiltonian, vectorised."""
    h0 = onsite(spin)
    t = np.array([hopping(d, spin) for d in BONDS])               # (4, n, n)
    phase = np.exp(1j*(2*np.pi*np.asarray(frac)) @ (BONDS.T/4))   # (nk, 4)
    hab = np.einsum('kd,dij->kij', phase, t)
    nk, n = len(frac), len(h0)
    h = np.zeros((nk, 2*n, 2*n), complex)
    h[:, :n, :n] = h0
    h[:, n:, n:] = h0
    h[:, :n, n:] = hab
    h[:, n:, :n] = hab.conj().transpose(0, 2, 1)
    return np.linalg.eigvalsh(h)


def solve_edge(h, sigma, nlevels, side, seed=0, tol=1e-8, maxiter=500):
    """The nlevels eigenstates just above (side='above') or below sigma.

    Block solver (LOBPCG) on the shift-inverted operator (H-sigma)^-1, so exact
    multiplets are returned complete. ARPACK eigsh, as in solve_near, is a
    single-vector Krylov method and can drop members of degenerate multiplets
    (seen here: a 12-fold box level came back as 9). sigma must lie in a gap
    so that all wanted states are on one side of it.
    """
    n = h.shape[0]
    t = (h - sigma*identity(n, format='csr')).tocsc()
    lu = splu(t, permc_spec='MMD_AT_PLUS_A', options=dict(SymmetricMode=True))
    def apply(x):
        x = np.asarray(x)
        return lu.solve(x.reshape(n, -1).astype(complex)).reshape(x.shape)
    op = LinearOperator(h.shape, matvec=apply, matmat=apply, dtype=complex)
    block = nlevels + max(8, nlevels//2)        # guard vectors for cut multiplets
    x0 = np.random.default_rng(seed).normal(size=(n, block)) + 0j
    with warnings.catch_warnings():  # convergence is checked by the residuals below
        warnings.simplefilter('ignore', UserWarning)
        _, vec = lobpcg(op, x0, largest=(side == 'above'), tol=tol, maxiter=maxiter)
    energy = np.real(np.einsum('ij,ij->j', vec.conj(), h @ vec))
    order = np.argsort(energy) if side == 'above' else np.argsort(energy)[::-1]
    energy, vec = energy[order][:nlevels], vec[:, order][:, :nlevels]
    residual = np.linalg.norm(h @ vec - vec*energy, axis=0)
    return energy, vec, residual


def box_kpoints(shape):
    """One k per primitive cell of the box: 4*Nx*Ny*Nz points (units 2pi/a).

    k = (n1/Nx, n2/Ny, n3/Nz). fcc reciprocal vectors (2,0,0) and (1,1,1)
    make n_i ~ n_i + 2N_i and (n1,n2,n3) ~ (n1+Nx, n2+Ny, n3+Nz), so
    n1 < 2Nx, n2 < 2Ny, n3 < Nz lists each distinct k exactly once.
    """
    nx, ny, nz = shape
    n = np.array(list(itertools.product(range(2*nx), range(2*ny), range(nz))))
    return n/np.asarray(shape, float)


_G = np.array([g for g in itertools.product(range(-2, 3), repeat=3)
               if len({abs(c) % 2 for c in g}) == 1])  # fcc reciprocal vectors, units 2pi/a


def reduce_k(f):
    """Shortest equivalent of k (units 2pi/a), i.e. its first-BZ image."""
    cand = f + _G
    return cand[np.argmin(np.einsum('ij,ij->i', cand, cand))]


def star_label(f):
    """Point-group star label, e.g. (0.750, 0, 0) for all six +-x,+-y,+-z images."""
    s = sorted(np.round(np.abs(reduce_k(f)), 4), reverse=True)
    return '(' + ', '.join(f'{c:.3f}'.rstrip('0').rstrip('.') if c else '0' for c in s) + ')'


def group_levels(energy, tol=DEGEN_TOL):
    """Sorted energies -> list of [mean energy, degeneracy, member indices]."""
    order = np.argsort(energy)
    groups = []
    for i in order:
        if groups and energy[i] - groups[-1][2][-1][1] < tol:
            groups[-1][2].append((i, energy[i]))
        else:
            groups.append([None, None, [(i, energy[i])]])
    return [[np.mean([e for _, e in g]), len(g), [i for i, _ in g]] for *_, g in groups]


def folded_spectrum(shape, spin=True, chunk=20000):
    """Exact periodic-box spectrum: (k-points, eigenvalues[nk, nbands])."""
    frac = box_kpoints(shape)
    energy = np.concatenate([bloch_batch(frac[i:i+chunk], spin)
                             for i in range(0, len(frac), chunk)])
    return frac, energy


def box_edges(shape, spin=True):
    """Fast E_v^box, E_c^box, and lowest conduction k-star for a periodic box."""
    occupied = 8 if spin else 4
    frac, energy = folded_spectrum(shape, spin)
    ev, ec = energy[:, occupied-1].max(), energy[:, occupied].min()
    kc = frac[np.argmin(energy[:, occupied])]
    ndeg = int(np.sum(np.abs(energy[:, occupied:] - ec) < DEGEN_TOL))
    return ev, ec, kc, ndeg


def scan(cells, spin=True):
    """Table of E_c^box - CBM_bulk versus box size (cubic, periodic)."""
    vbm, cbm, valley = bulk_edges(spin)
    print(f'bulk VBM={vbm:.6f}  CBM={cbm:.6f} eV; valley at k={valley:.4f} (2pi/a)')
    print(' N   L(nm)   atoms     Ec_box-CBM(meV)  deg  lowest k-star   N*k0 - nearest int')
    for n in cells:
        _, ec, kc, ndeg = box_edges((n, n, n), spin)
        mis = n*valley - round(n*valley)
        print(f'{n:3d} {n*A/10:6.2f} {8*n**3:8d} {1000*(ec-cbm):14.3f} {ndeg:6d}  '
              f'{star_label(kc):14s} {mis:+.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ncells', type=int, nargs='+', default=None,
                        help='cells per axis: one value or three (Nx Ny Nz); default NCELLS')
    parser.add_argument('--periodic', choices=['true', 'false'], nargs='+', default=None)
    parser.add_argument('--spinless', action='store_true')
    parser.add_argument('--nlevels', type=int, default=NLEVELS)
    parser.add_argument('--no-sparse', action='store_true',
                        help='skip the real-space shift-invert check (periodic boxes only)')
    parser.add_argument('--scan', type=int, nargs='+', metavar='N',
                        help='only tabulate E_c^box - CBM for cubic periodic boxes of these N')
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args()
    spin = SPIN and not args.spinless
    if args.scan:
        scan(args.scan, spin)
        return
    ncells = NCELLS if args.ncells is None else args.ncells
    periodic = PERIODIC if args.periodic is None else [p == 'true' for p in args.periodic]
    shape, lengths, pbc = box_geometry(ncells, periodic)
    occupied = 8 if spin else 4

    vbm, cbm, valley = bulk_edges(spin)
    print(f'bulk VBM={vbm:.9f}, CBM={cbm:.9f} eV; valley k/(2pi/a)={valley:.7f}')
    print(f'Box {shape} cells = {lengths} Angstrom; periodic = {tuple(bool(p) for p in pbc)}; '
          f'{8*np.prod(shape)} atoms')

    results = {}
    if np.all(pbc):
        frac, bands = folded_spectrum(shape, spin)
        flat = bands.ravel()
        kidx = np.repeat(np.arange(len(frac)), bands.shape[1])
        cond = flat >= bands[:, occupied].min() - 1e-12
        val = flat <= bands[:, occupied-1].max() + 1e-12
        cE, ck = flat[cond], kidx[cond]
        vE, vk = flat[val], kidx[val]
        ci = np.argsort(cE)[:args.nlevels]
        vi = np.argsort(vE)[::-1][:args.nlevels]
        results['folded'] = (np.sort(vE[vi]), np.sort(cE[ci]))
        ec, ev = cE[ci[0]], vE[vi[0]]
        print(f'\n[folded, exact]  E_v^box={ev:.9f}  E_c^box={ec:.9f} eV')
        print(f'  E_v^box - VBM_bulk = {1000*(ev-vbm):8.3f} meV')
        print(f'  E_c^box - CBM_bulk = {1000*(ec-cbm):8.3f} meV   <- reference offset for the donor')
        print('  lowest conduction levels of the box:')
        print('    E (eV)        E-CBM_bulk (meV)  deg   k-star (2pi/a)')
        for e, deg, members in group_levels(cE[ci])[:8]:
            stars = sorted({star_label(frac[ck[ci[m]]]) for m in members})
            print(f'    {e:.8f} {1000*(e-cbm):14.3f} {deg:6d}   {" ".join(stars)}')
        if len(group_levels(cE[ci])) < 2:
            print('  (increase --nlevels to see more groups)')
    else:
        print('\nOpen axes present: exact folding not available; unpassivated surfaces '
              'give dangling-bond states, so gap states here are NOT bulk-like.')

    if not args.no_sparse or not np.all(pbc):
        h, positions = make_real_space(shape, periodic=pbc, spin=spin)
        print(f'\n[sparse] H={h.shape}; shift-invert + LOBPCG at both band edges...', flush=True)
        ce, _, cres = solve_edge(h, cbm - 0.01, args.nlevels, 'above')
        ve, _, vres = solve_edge(h, vbm + 0.01, args.nlevels, 'below')
        results['sparse'] = (np.sort(ve), np.sort(ce))
        print(f'  highest {len(ve)} valence:  {ve.max():.9f} ... {ve.min():.9f} eV'
              f'  (max residual {vres.max():.1e})')
        print(f'  lowest  {len(ce)} conduction: {ce.min():.9f} ... {ce.max():.9f} eV'
              f'  (max residual {cres.max():.1e})')
        for e, deg, _ in group_levels(ce)[:8]:
            print(f'    {e:.8f} {1000*(e-cbm):14.3f} meV  deg {deg}')
        if 'folded' in results:
            dc = np.abs(np.sort(ce) - results['folded'][1]).max()
            dv = np.abs(np.sort(ve) - results['folded'][0]).max()
            print(f'  max |sparse - folded|: conduction {dc:.2e} eV, valence {dv:.2e} eV')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(vbm_bulk_eV=vbm, cbm_bulk_eV=cbm, valley_fraction=valley,
                ncells=shape, periodic=pbc, spin=spin)
    for key, (v, c) in results.items():
        save[f'{key}_valence_eV'] = v
        save[f'{key}_conduction_eV'] = c
    ref = results.get('folded', results.get('sparse'))
    save['Ev_box_eV'], save['Ec_box_eV'] = ref[0].max(), ref[1].min()
    np.savez(str(out) + '.npz', **save)
    print(f'\nSaved {out}.npz  (Ec_box_eV is the reference for donor binding energies)')


if __name__ == '__main__':
    main()
