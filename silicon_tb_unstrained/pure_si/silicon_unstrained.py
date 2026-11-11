"""Unstrained Si: orthogonal nearest-neighbour sp3d5s* tight binding.
Niquet et al., PRB 79, 245201 (2009), Table I. Energies eV, lengths Angstrom.
Basis per atom: (s,px,py,pz,dxy,dyz,dzx,dx2-y2,d3z2-r2,s*).
With SOC: orbital-major (s up,s down,px up,px down,...).
No strain, impurity, Zeeman term, or surface passivation is included.
"""
import numpy as np
from scipy.sparse import coo_matrix

#----------------------------------
# hopping parameters
#----------------------------------

ORBITALS = ('s','px','py','pz','dxy','dyz','dzx','dx2-y2','d3z2-r2','s*')
D0 = 2.35169  #NN bond length, not the side length of the cubic unit cell.
A = 4 * D0 / np.sqrt(3)  #the conventional cubic lattice constant
ONSITE = np.array([-2.55247]+[4.48593]*3+[14.01053]*5+[23.44607]) #onsite eneries of the 10 orbitals
LAMBDA_SO = 0.01851 #spin orbit parameter
V = dict(ss=-1.86600, ssstar=-1.39107, sp=2.91067, sd=-2.23992,
         sstarsstar=-4.51331, sstarp=3.06822, sstard=-0.77711,
         pps=4.08481, ppp=-1.49207, pds=-1.66657, pdp=2.39936,
         dds=-1.82945, ddp=3.08177, ddd=-1.56676) #hopping integrals between neighbouring atoms
# Four A -> B bonds, in units a/4.
BONDS = np.array([[1,1,1],
                  [1,-1,-1],
                  [-1,1,-1],
                  [-1,-1,1]]) #the four nearest-neighbour displacements

#the eight atom positions inside a conventional cell
#The last four positions are obtained by shifting the first four by a/4(1,1,1)
BASIS_Q = np.array([[0,0,0],[0,2,2],[2,0,2],[2,2,0],
                    [1,1,1],[1,3,3],[3,1,3],[3,3,1]])




def _d_tensors():
    """Orthonormal traceless tensors representing the five real d harmonics."""
    q = np.zeros((5,3,3))
    for n,(i,j) in enumerate([(0,1),(1,2),(2,0)]):
        q[n,i,j] = q[n,j,i] = 1/np.sqrt(2)
    q[3] = np.diag([1,-1,0])/np.sqrt(2)
    q[4] = np.diag([-1,-1,2])/np.sqrt(6)
    return q


def hopping(displacement, spin=True):
    """Return <i|H|j> for displacement R_j-R_i; unstrained bond only.

    Slater-Koster integrals are assembled in a bond frame (local z along
    displacement), then real p and d orbitals are rotated to the lab frame.
    This avoids a long, error-prone explicit table of angular polynomials.
    Length is deliberately ignored: this function does NOT model strain.
    """
    z = np.asarray(displacement, dtype=float)
    if z.shape != (3,) or not np.all(np.isfinite(z)) or np.linalg.norm(z)==0:
        raise ValueError('displacement must be a finite nonzero 3-vector')
    z = z/np.linalg.norm(z)
    ref = np.eye(3)[np.argmin(np.abs(z))]
    x = np.cross(ref,z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    rot = np.column_stack([x,y,z])
    u = np.eye(10)
    u[1:4,1:4] = rot
    q = _d_tensors()
    rotated = np.array([rot @ qi @ rot.T for qi in q])
    u[4:9,4:9] = np.einsum('aij,bij->ab',q,rotated)
    t = np.zeros((10,10))
    def pair(i,j,value,parity=1):
        t[i,j]=value; t[j,i]=parity*value
    t[0,0]=V['ss']; t[9,9]=V['sstarsstar']
    pair(0,9,V['ssstar'])
    pair(0,3,V['sp'],-1) 
    pair(9,3,V['sstarp'],-1)
    pair(0,8,V['sd']) 
    pair(9,8,V['sstard'])
    t[1,1]=t[2,2]=V['ppp'] 
    t[3,3]=V['pps']
    pair(3,8,V['pds'],-1)
    pair(1,6,V['pdp'],-1) 
    pair(2,5,V['pdp'],-1)
    t[4,4]=t[7,7]=V['ddd'] 
    t[5,5]=t[6,6]=V['ddp'] 
    t[8,8]=V['dds']
    result = u @ t @ u.T
    result[np.abs(result)<1e-13] = 0
    return np.kron(result,np.eye(2)) if spin else result


def onsite(spin=True):
    """Atomic onsite matrix. Table-I convention: H_SOC=lambda_so L.sigma.

    L and Pauli matrices are dimensionless; equivalently 2*lambda_so L.S.
    Isolated atomic p manifold has j=3/2 to j=1/2 splitting 3*lambda_so.
    Do not interpret the table's lambda_so directly as the coefficient of L.S.
    """
    h = np.diag(ONSITE)
    if not spin:
        return h
    h = np.kron(h,np.eye(2)).astype(complex)
    lx = np.array([[0,0,0],[0,0,-1j],[0,1j,0]])
    ly = np.array([[0,0,1j],[0,0,0],[-1j,0,0]])
    lz = np.array([[0,-1j,0],[1j,0,0],[0,0,0]])
    sx = np.array([[0,1],[1,0]]); sy = np.array([[0,-1j],[1j,0]])
    sz = np.diag([1,-1])
    h[2:8,2:8] += LAMBDA_SO*sum(np.kron(l,s) for l,s in zip((lx,ly,lz),(sx,sy,sz)))
    return h


def bloch_hamiltonian(k, spin=True):
    """Primitive two-atom cell, k in inverse Angstrom; shape 40 or 20.
    Gauge includes the atomic positions: H_AB=sum_delta T(delta) exp(i k.delta).
    """
    k = np.asarray(k,dtype=float)
    if k.shape != (3,):
        raise ValueError('k must have three Cartesian components')
    h0 = onsite(spin)
    hab = sum(hopping(d,spin)*np.exp(1j*np.dot(k,d*A/4)) for d in BONDS)
    return np.block([[h0,hab],[hab.conj().T,h0]])


def make_real_space(ncells=10, periodic=True, spin=True):
    """Build CSR Hamiltonian and atom positions for a conventional-cell box.

    ncells: positive integer or (Nx,Ny,Nz). periodic: bool or three bools.
    Returns H, positions (Angstrom). Atom index i owns H[i*norb:(i+1)*norb].
    Default periodic boundaries describe pristine bulk. Open boundaries are
    UNPASSIVATED and can produce dangling-bond gap states; they are not yet
    suitable for quantitative isolated-donor calculations.
    Does not diagonalize the matrix. Never convert a large result to dense.
    """
    shape = np.broadcast_to(np.asarray(ncells), (3,))
    if np.any(shape<1) or np.any(shape!=shape.astype(int)):
        raise ValueError('ncells must contain positive integers')
    shape=shape.astype(int) 
    pbc=np.broadcast_to(np.asarray(periodic,bool),(3,))
    cells=np.indices(tuple(shape)).reshape(3,-1).T
    q=(4*cells[:,None,:]+BASIS_Q[None,:,:]).reshape(-1,3)
    lookup={tuple(r):i for i,r in enumerate(q)}
    rows=[]; cols=[]; values=[]
    h0=onsite(spin); norb=len(h0)
    def add_blocks(sites_i,sites_j,block):
        bi,bj=np.nonzero(np.abs(block)>1e-13)
        rows.append((np.asarray(sites_i)[:,None]*norb+bi).ravel())
        cols.append((np.asarray(sites_j)[:,None]*norb+bj).ravel())
        values.append(np.tile(block[bi,bj],len(sites_i)))
    add_blocks(np.arange(len(q)),np.arange(len(q)),h0)
    a_sites=np.flatnonzero(np.all(q%2==0,axis=1))
    for d in BONDS:
        src=[]; dst=[]
        for i in a_sites:
            target=q[i]+d
            if np.any(((target<0)|(target>=4*shape)) & ~pbc):
                continue
            target=np.where(pbc,target%(4*shape),target)
            j=lookup.get(tuple(target))
            if j is not None:
                src.append(i); dst.append(j)
        if src:
            t=hopping(d,spin)
            add_blocks(src,dst,t); add_blocks(dst,src,t.conj().T)
    dim=len(q)*norb
    h=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(dim,dim)).tocsr()
    h.sum_duplicates()
    h.eliminate_zeros()
    return h,q*A/4

# H_k = bloch_hamiltonian([0,0,0], spin=True)
# print(H_k.shape)

# H, positions = make_real_space(
#     ncells=4,
#     periodic=True,
#     spin=True
# )

# print(H.shape)
# print(positions.shape)

if __name__ == "__main__":
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import time

    NCELLS = 5
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    H, positions = make_real_space(ncells=NCELLS, periodic=True, spin=True)
    t_build = time.perf_counter() - t0

    print("Hamiltonian shape:", H.shape)
    print("Positions shape:", positions.shape)

    sparse_bytes = H.data.nbytes + H.indices.nbytes + H.indptr.nbytes
    print("Nonzeros:", H.nnz)
    print(f"Sparse (CSR) Hamiltonian memory: {sparse_bytes/1024**2:.2f} MB ({sparse_bytes/1024**3:.4f} GB)")
    print(f"Equivalent dense memory would be: {H.shape[0]*H.shape[1]*H.dtype.itemsize/1024**3:.2f} GB")

    sigma = 1.173  # target eigenvalue
    # NB: do not name this A -- that is the lattice constant used by make_real_space
    H_shift = (H - sigma * sp.identity(H.shape[0], dtype=H.dtype, format='csc')).tocsc()
    t0 = time.perf_counter()
    lu = spla.splu(H_shift, permc_spec='MMD_AT_PLUS_A')   # far less fill than COLAMD for symmetric patterns
    t_lu = time.perf_counter() - t0
    print(f"LU fill: nnz(L)+nnz(U) = {lu.L.nnz + lu.U.nnz}")
    OPinv = spla.LinearOperator(H_shift.shape, matvec=lu.solve, dtype=H_shift.dtype)
    t0 = time.perf_counter()
    vals, vecs = spla.eigsh(H, k=40, sigma=sigma, which='LM', OPinv=OPinv, tol=1e-10)
    t_eigs = time.perf_counter() - t0

    print("Eigenvalues near sigma:", vals[:10])  # print first 10 eigenvalues near sigma

    print(f"\nTiming for ncells={NCELLS} (dim={H.shape[0]}):")
    print(f"  build H     : {t_build:9.2f} s")
    print(f"  LU factor   : {t_lu:9.2f} s")
    print(f"  eigsh       : {t_eigs:9.2f} s")
    print(f"  total       : {time.perf_counter() - t_start:9.2f} s")
