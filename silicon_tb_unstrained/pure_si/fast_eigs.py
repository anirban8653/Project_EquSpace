"""Fast shift-invert eigensolver for sparse Hermitian H using MKL Pardiso.

Preferred: pydiso, native complex-Hermitian LDL^H (fastest, least memory).
Fallbacks: pypardiso on the exact real 2N embedding, then SciPy SuperLU.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator, splu

def shift_invert_op(H, sigma, refine=2):
    n = H.shape[0]
    M = (H - sigma*sp.identity(n, format='csr', dtype=H.dtype)).tocsr()
    cplx = np.iscomplexobj(M.data)
    try:
        from pydiso.mkl_solver import MKLPardisoSolver
        mtype = 'complex_hermitian_indefinite' if cplx else 'real_symmetric_indefinite'
        s = MKLPardisoSolver(sp.triu(M, format='csr'), matrix_type=mtype,
                             max_iterative_refinement_steps=refine)
        return LinearOperator((n, n), matvec=lambda b: s.solve(np.asarray(b).ravel()), dtype=H.dtype)
    except ImportError:
        pass
    try:
        import pypardiso
        R = sp.bmat([[M.real, -M.imag], [M.imag, M.real]], format='csr') if cplx else M
        solver = pypardiso.PyPardisoSolver(); solver.factorize(R)
        if cplx:
            def mv(b):
                b = np.asarray(b).ravel(); y = solver.solve(R, np.concatenate([b.real, b.imag]))
                return y[:n] + 1j*y[n:]
        else:
            mv = lambda b: solver.solve(R, np.asarray(b).ravel())
        return LinearOperator((n, n), matvec=mv, dtype=H.dtype)
    except ImportError:
        lu = splu(M.tocsc(), permc_spec='MMD_AT_PLUS_A', options=dict(SymmetricMode=True))
        return LinearOperator((n, n), matvec=lu.solve, dtype=H.dtype)

def eigs_near(H, sigma, k=12, tol=1e-10, **kw):
    OPinv = shift_invert_op(H, sigma)
    vals, vecs = eigsh(H, k=k, sigma=sigma, which='LM', OPinv=OPinv, tol=tol, **kw)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]
