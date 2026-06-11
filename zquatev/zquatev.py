import os
import ctypes
import numpy
import numpy as np

libzquatev = ctypes.CDLL(os.path.abspath(os.path.join(__file__, '..', 'libzquatev.so')))

def eigh(mat, iop=0):
    '''HC=CE'''
    n2 = mat.shape[0]
    n = n2 // 2
    fmat = np.array(mat, dtype=np.complex128, order='F', copy=True)
    eigs = np.zeros(n2, dtype=np.float64)
    err = libzquatev.eigh(
        fmat.ctypes, eigs.ctypes, ctypes.c_int(n2), ctypes.c_int(n2))
    if err != 0:
        raise RuntimeError('zquatev failed')

    if iop == 1:
        e = np.empty_like(eigs)
        v = np.empty_like(fmat)
        e[0::2] = eigs[:n]
        e[1::2] = eigs[:n]
        v[:,0::2] = fmat[:,:n]
        v[:,1::2] = fmat[:,n:]
    else:
        eigs[n:] = eigs[:n]
        e, v = eigs, fmat
    return e, v

def geigh(tfock, tova, debug=False):
    '''FC=SCE'''
    # canonicalization in a kramers symmetry adapted basis
    e,v = eigh(tova)
    if debug: print('eig(S)=',e)
    vcanon = v/numpy.sqrt(e)
    # transform fock matrix to orthonormal basis
    tfockmo = vcanon.conj().T.dot(tfock.dot(vcanon))
    e,v = eigh(tfockmo, iop=1)
    if debug: print('eig(Fmo)=',e)
    # back to AO basis
    v = vcanon.dot(v)
    return e,v

def check_kramers_structure(mat, thresh=1.e-8):
    print('check_kramers_structure for matrix = [A,B;C,D]')
    n = mat.shape[0]//2
    a = mat[:n,:n]
    b = mat[:n,n:]
    c = mat[n:,:n]
    d = mat[n:,n:]
    #print(a)
    #print(b)
    #print(c)
    #print(d)
    diff1 = numpy.linalg.norm(a-a.T.conj())
    diff2 = numpy.linalg.norm(b+b.T)
    diff3 = numpy.linalg.norm(c+b.conj())
    diff4 = numpy.linalg.norm(d-a.conj())
    print('A-Ah=',diff1)
    print('B+Bt=',diff2)
    print('C+B*=',diff3)
    print('D-A*=',diff4)
    assert(diff1 < thresh and
           diff2 < thresh and
           diff3 < thresh and
           diff4 < thresh)
    return 0

def solve_KR_FCSCE(mol, fock, ova, debug=False):
    # index for kramers basis from
    # https://github.com/sunqm/pyscf/blob/master/pyscf/gto/mole.py
    trmaps = mol.time_reversal_map()
    idxA = numpy.where(trmaps > 0)[0]
    idxB = trmaps[idxA] - 1
    if fock.shape[0] == trmaps.size:
        idx2 = numpy.hstack((idxA,idxB))
    else:
        n = trmaps.size
        idx2 = numpy.hstack((idxA,idxA+n,idxB,idxB+n))

    # {|chi>,T|chi>}
    tova = ova[numpy.ix_(idx2,idx2)]
    tfock = fock[numpy.ix_(idx2,idx2)]
    if debug:
        labels = mol.spinor_labels()
        print(labels)
        for i in range(n):
            print(i,labels[i],trmaps[i])
            print('idxA=',idxA)
            print('idxB=',idxB)
            print('idx2=',idx2)
            check_kramers_structure(tova)
            check_kramers_structure(tfock)
    e,v = geigh(tfock,tova,debug)
    kmo_coeff = numpy.zeros_like(v)
    kmo_coeff[idx2,:] = v
    return e, kmo_coeff

