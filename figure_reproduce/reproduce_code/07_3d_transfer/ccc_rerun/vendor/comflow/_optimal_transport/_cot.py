import numpy as np
import pandas as pd
from scipy import sparse
from sys import getsizeof
import random
from collections import defaultdict
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._unot import unot 

def cot_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """ Solve the collective optimal transport problem with distance limits.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    (ns_s,ns_d,n_pos_s,n_pos_d) numpy.ndarray
        The transport plans among the multiple species.
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    C = np.inf * np.ones([len(a),len(b)])
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_M = np.array(M)
                tmp_M[np.where(tmp_M > cutoff[i,j])] = np.inf
                C[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d] = A[i,j] * tmp_M
    C = C/np.max(C[np.where(~np.isinf(C))])
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    P = np.zeros_like(C)
    for i in range(len(nzind_a)):
        for j in range(len(nzind_b)):
            P[nzind_a[i],nzind_b[j]] = tmp_P[i,j]
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            P_expand[i,j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,j*n_pos_d:(j+1)*n_pos_d]
    return P_expand * max_amount

def cot_row_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[i,D_j])] = np.inf
            C[:,j*n_pos_d:(j+1)*n_pos_d] = A[i,D_j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for j in range(len(D_ind)):
            P_expand[i,D_ind[j],:,:] = P[:,j*n_pos_d:(j+1)*n_pos_d] * max_amount
    return P_expand

def cot_col_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for j in range(ns_d):
        b = D[:,j]
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C = np.inf * np.ones([len(a), len(b)], float)
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_M = np.array(M)
            tmp_M[np.where(tmp_M > cutoff[S_i,j])] = np.inf
            C[i*n_pos_s:(i+1)*n_pos_s,:] = A[S_i,j] * tmp_M
        C = C/np.max(C[np.where(~np.isinf(C))])
        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
        P = np.zeros_like(C)
        for ii in range(len(nzind_a)):
            for jj in range(len(nzind_b)):
                P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
        for i in range(len(S_ind)):
            P_expand[S_ind[i],j,:,:] = P[i*n_pos_s:(i+1)*n_pos_s,:] * max_amount
    return P_expand

def cot_blk_dense(S, D, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8):
    """Solve for each pair of species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    P_expand = np.zeros([ns_s, ns_d, n_pos_s, n_pos_d], float)
    for i in range(ns_s):
        for j in range(ns_d):
            if np.isinf(A[i,j]): continue
            a = S[:,i]; b = D[:,j]
            max_amount = max(a.sum(), b.sum())
            a = a / max_amount; b = b / max_amount
            C = np.array(M)
            C[np.where(C > cutoff[i,j])] = np.inf
            C = C/np.max(C[np.where(~np.isinf(C))])
            nzind_a = np.where(a > 0)[0]
            nzind_b = np.where(b > 0)[0]
            tmp_P = unot(a[nzind_a], b[nzind_b], C[nzind_a,:][:,nzind_b], eps_p, rho, \
                eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=False, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
            P = np.zeros_like(C)
            for ii in range(len(nzind_a)):
                for jj in range(len(nzind_b)):
                    P[nzind_a[ii],nzind_b[jj]] = tmp_P[ii,jj]
            P_expand[i,j,:,:] = P[:,:]
    return P_expand

def cot_combine_sparse(S, D, Q, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, lam=1e1, weights=(0.25,0.25,0.25,0.25), nitermax=1e4, stopthr=1e-8, verbose=False):
    if isinstance(eps_p, tuple):
        eps_p_cot, eps_p_row, eps_p_col, eps_p_blk = eps_p
    else:
        eps_p_cot = eps_p_row = eps_p_col = eps_p_blk = eps_p
    if isinstance(rho, tuple):
        rho_cot, rho_row, rho_col, rho_blk = rho
    else:
        rho_cot = rho_row = rho_col = rho_blk = rho
    if isinstance(lam, tuple):
        lam_cot, lam_row, lam_col, lam_blk = lam
    else:
         lam_cot = lam_row = lam_col = lam_blk = lam
    if eps_mu is None:
        eps_mu_cot = eps_p_cot; eps_mu_row = eps_p_row
        eps_mu_col = eps_p_col; eps_mu_blk = eps_p_blk
    elif isinstance(eps_mu, tuple):
        eps_mu_cot, eps_mu_row, eps_mu_col, eps_mu_blk = eps_mu
    else:
        eps_mu_cot = eps_mu_row = eps_mu_col = eps_mu_blk = eps_mu
    if eps_nu is None:
        eps_nu_cot = eps_p_cot; eps_nu_row = eps_p_row
        eps_nu_col = eps_p_col; eps_nu_blk = eps_p_blk
    elif isinstance(eps_nu, tuple):
        eps_nu_cot, eps_nu_row, eps_nu_col, eps_nu_blk = eps_nu
    else:
        eps_nu_cot = eps_nu_row = eps_nu_col = eps_nu_blk = eps_nu

    P_cot,CostC = cot_sparse(S, D, Q, A, M, cutoff, \
        eps_p=eps_p_cot, eps_mu=eps_mu_cot, eps_nu=eps_nu_cot, rho=rho_cot, lam=lam_cot, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_row = cot_row_sparse(S, D, Q, A, M, cutoff, \
        eps_p=eps_p_row, eps_mu=eps_mu_row, eps_nu=eps_nu_row, rho=rho_row, lam=lam_row, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_col = cot_col_sparse(S, D, Q, A, M, cutoff, \
        eps_p=eps_p_col, eps_mu=eps_mu_col, eps_nu=eps_nu_col, rho=rho_col, lam=lam_col, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    P_blk = cot_blk_sparse(S, D, Q, A, M, cutoff, \
        eps_p=eps_p_blk, eps_mu=eps_mu_blk, eps_nu=eps_nu_blk, rho=rho_blk, lam=lam_blk, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)

    P = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isinf(A[i,j]):
                P[(i,j)] = float(weights[0]) * P_cot[(i,j)] + float(weights[1]) * P_row[(i,j)] \
                    + float(weights[2]) * P_col[(i,j)] + float(weights[3]) * P_blk[(i,j)]
    return(P),CostC

def cot_sparse(S, D, Q, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, lam=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """ Solve the collective optimal transport problem with distance limits in sparse format.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    Q : (n_pos_d,ns_d) numpy.ndarray
        Intra-signal destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e1
        The coefficient for penalizing unmatched mass.
    lam : float, defaults to le1
        The coefficient for KL divergence between inter- and intra-signal distributions.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    A dictionary of scipy.sparse.coo_matrix
        The transport plan in coo sparse format for source species i and destinaton species j can be retrieved with the key (i,j).
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape
    max_amount = max( S.sum(), D.sum() )
    S = S / max_amount
    D = D / max_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    q = Q.flatten('F')
    
    C_data, C_row, C_col = [], [], []

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    M_max_sp = M.copy()
    
    cost_scales = []
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                cost_scales.append(np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])])*A[i,j])
                C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,j] )
                C_row.append( tmp_row+i*n_pos_s )
                C_col.append( tmp_col+j*n_pos_d )

    cost_scale = np.max(cost_scales)
    C_data = np.concatenate(C_data, axis=0)
    C_row = np.concatenate(C_row, axis=0)
    C_col = np.concatenate(C_col, axis=0)
    C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a),len(b)))

    # Solve the problem on nonzero mass
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    nzind_q = np.where(q > 0)[0]
    overlap = np.intersect1d(nzind_b, nzind_q)
    # C_csr = C.tocsr()

    # valid_rows = np.where(np.diff(C_csr.indptr) > 0)[0]
    # valid_cols = np.unique(C_csr.indices)

    # nzind_a = np.intersect1d(nzind_a, valid_rows)
    # overlap = np.intersect1d(overlap, valid_cols)

    C_nz = coo_submatrix_pull(C, nzind_a, overlap)

    if verbose:
        print('Number of non-infinity entries in transport cost:', len(C.data))

    # del C_data, C_row, C_col, C
    del C_data, C_row, C_col

    tmp_P = unot(a[nzind_a], b[overlap], q[overlap], C_nz, eps_p, rho, lam, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    
    del C_nz

    P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], overlap[tmp_P.col])), shape=(len(a),len(b)))
    P = P.tocsr()

    # Output a dictionary of transport plans
    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_P = P[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d]
                P_expand[(i,j)] = tmp_P.tocoo() * max_amount

    return P_expand,C

def cot_row_sparse(S, D, Q, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, lam=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    M_max_sp = M.copy()
    
    P_expand = {}
    for i in range(ns_s):
        a = S[:,i]
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F')
        q = Q[:,D_ind].flatten('F')
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]; nzind_q = np.where(q > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0 or len(nzind_q)==0:
            for j in range(len(D_ind)):
                P_expand[(i,D_ind[j])] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:,i] > 0)[0]
            tmp_nzind_d = np.where(D[:,D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,D_j] )
            C_row.append( tmp_row )
            C_col.append( tmp_col+j*n_pos_d )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,D_j])])*A[i,D_j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        nzind_q = np.where(q > 0)[0]
        overlap = np.intersect1d(nzind_b, nzind_q)
        # C_csr = C.tocsr()

        # valid_rows = np.where(np.diff(C_csr.indptr) > 0)[0]
        # valid_cols = np.unique(C_csr.indices)

        # nzind_a = np.intersect1d(nzind_a, valid_rows)
        # overlap = np.intersect1d(overlap, valid_cols)
        C_nz = coo_submatrix_pull(C, nzind_a, overlap)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[overlap], q[overlap], C_nz, eps_p, rho, lam,\
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)          

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], overlap[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:,j*n_pos_d:(j+1)*n_pos_d]
            P_expand[(i,D_ind[j])] = tmp_P.tocoo() * max_amount

        del P

    return P_expand

def cot_col_sparse(S, D, Q, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, lam=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    M_max_sp = M.copy()
    
    P_expand = {}
    for j in range(ns_d):
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        a = S[:,S_ind].flatten('F')
        b = D[:,j]
        q = Q[:,j]
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]; nzind_q = np.where(q > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0 or len(nzind_q)==0 :
            for i in range(len(S_ind)):
                P_expand[(S_ind[i],j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s,n_pos_d), dtype=float)
            continue
        max_amount = max(a.sum(), b.sum())
        a = a / max_amount; b = b / max_amount
        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]
            tmp_nzind_s = np.where(S[:,S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:,j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i,j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[S_i,j] )
            C_row.append( tmp_row+i*n_pos_s )
            C_col.append( tmp_col )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i,j])])*A[S_i,j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        nzind_q = np.where(q > 0)[0]
        overlap = np.intersect1d(nzind_b, nzind_q)
        # C_csr = C.tocsr()

        # valid_rows = np.where(np.diff(C_csr.indptr) > 0)[0]
        # valid_cols = np.unique(C_csr.indices)

        # nzind_a = np.intersect1d(nzind_a, valid_rows)
        # overlap = np.intersect1d(overlap, valid_cols)
        C_nz = coo_submatrix_pull(C, nzind_a, overlap)          

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[overlap], q[overlap], C_nz, eps_p, rho, lam,\
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], overlap[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i*n_pos_s:(i+1)*n_pos_s,:]
            P_expand[(S_ind[i],j)] = tmp_P.tocoo() * max_amount

        del P

    return P_expand

def cot_blk_sparse(S, D, Q, A, M, cutoff, eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, lam=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu), abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    M_max_sp = M.copy()

    P_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                a = S[:,i]; b = D[:,j]; q = Q[:,j]
                nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]; nzind_q = np.where(q > 0)[0]
                if len(nzind_a)==0 or len(nzind_b)==0:
                    P_expand[(i,j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                    continue
                max_amount = max(a.sum(), b.sum())
                a = a / max_amount; b = b / max_amount
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                C_data = tmp_M_max_sp.data[tmp_ind] * A[i,j]
                cost_scale = np.max( M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])] )*A[i,j]
                C = sparse.coo_matrix((C_data/cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                nzind_q = np.where(q > 0)[0]
                overlap = np.intersect1d(nzind_b, nzind_q)
                # C_csr = C.tocsr()

                # valid_rows = np.where(np.diff(C_csr.indptr) > 0)[0]
                # valid_cols = np.unique(C_csr.indices)

                # nzind_a = np.intersect1d(nzind_a, valid_rows)
                # overlap = np.intersect1d(overlap, valid_cols)
                C_nz = coo_submatrix_pull(C, nzind_a, overlap)  

                del C_data, C

                tmp_P = unot(a[nzind_a], b[overlap], q[overlap], C_nz, eps_p, rho, lam,\
                    eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
                     
                del C_nz

                P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], overlap[tmp_P.col])), shape=(len(a),len(b)))

                P_expand[(i,j)] = P * max_amount
    
    return P_expand


def co_exp_batch_adata(adata, ggi_res, co_exp_ratio=0.05, batch_size=5000):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = pd.Index(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    ggi_res_temp = ggi_res[['src', 'dest']].drop_duplicates().copy()
    ggi_res_temp['src_idx'] = ggi_res_temp['src'].map(gene_to_idx)
    ggi_res_temp['dest_idx'] = ggi_res_temp['dest'].map(gene_to_idx)

    co_ratios = np.empty(len(ggi_res_temp), dtype=np.float32)

    # Process in batches
    for start in range(0, len(ggi_res_temp), batch_size):
        end = min(start + batch_size, len(ggi_res_temp))
        batch = ggi_res_temp.iloc[start:end]

        src_matrix = X[:, batch['src_idx'].values]
        dest_matrix = X[:, batch['dest_idx'].values]

        x12 = src_matrix * dest_matrix  # (n_cells × batch_size)
        co_ratios[start:end] = np.sum(x12 > 0, axis=0) / x12.shape[0]

    ggi_res_temp['co_ratio'] = co_ratios

    # Merge back
    ggi_res = ggi_res.merge(
        ggi_res_temp[['src', 'dest', 'co_ratio']],
        on=['src', 'dest'],
        how='left'
    )

    if co_exp_ratio > 0:
        ggi_res = ggi_res[ggi_res['co_ratio'] > co_exp_ratio]

    return ggi_res[['src', 'dest', 'src_tf', 'dest_tf', 'hop', 'co_ratio']]


def generate_tf_res(tf_gene_all, receptor_name, ggi_res):
    import pandas as pd

    tf_names = list(tf_gene_all.keys())
    tf_n_hop = list(tf_gene_all.values())
    
    # Initialize DataFrame
    receptor_tf_temp = pd.DataFrame({
        "receptor": [receptor_name] * len(tf_names),
        "tf": tf_names,
        "n_hop": tf_n_hop,
        "target_genes": [[] for _ in tf_names]  # initialize with empty lists
    })

    # Fill in `target_genes`
    for i in range(len(tf_names)):
        tf = tf_names[i]
        hop = tf_n_hop[i]
        # Filter rows where src == tf and hop == hop + 1
        ggi_res_tf = ggi_res[(ggi_res["src"] == tf) & (ggi_res["hop"] == hop + 1)]
        receptor_tf_temp.at[i, "target_genes"] = ggi_res_tf["dest"].unique().tolist()

    return receptor_tf_temp

def generate_tf_gene_all(ggi_res: pd.DataFrame, max_hop: int) -> dict:
    """
    Track TF-to-gene paths across hops and collect destination genes with TF activity.

    Args:
        ggi_res: DataFrame with columns ['src', 'dest', 'hop', 'dest_tf']
        max_hop: Maximum number of hops to search

    Returns:
        Dictionary of {gene_name: hop}
    """
    tf_gene_all = {}
    ggi_hop = ggi_res[ggi_res["hop"] == 1].copy()

    for k in range(1, max_hop + 1):
        # Get the rows where dest_tf == "YES"
        ggi_hop_yes = ggi_hop[ggi_hop["dest_tf"] == "YES"]

        if not ggi_hop_yes.empty:
            ggi_hop_tf = ggi_res[ggi_res["hop"] == k + 1]
            if not ggi_hop_tf.empty:
                # Keep only rows where ggi_hop_yes$dest in ggi_hop_tf$src
                ggi_hop_yes = ggi_hop_yes[ggi_hop_yes["dest"].isin(ggi_hop_tf["src"])]
                if not ggi_hop_yes.empty:
                    for _, row in ggi_hop_yes.iterrows():
                        tf_gene_all[row["dest"]] = row["hop"]

        # Prepare next ggi_hop from "NO" group
        ggi_hop_no = ggi_hop[ggi_hop["dest_tf"] == "NO"]
        ggi_hop = ggi_res[ggi_res["hop"] == k + 1]
        ggi_hop = ggi_hop[ggi_hop["src"].isin(ggi_hop_no["dest"])]

    return tf_gene_all

def random_walk_vectorized(receptor_tf, ggi_res, max_n_hop=10):
    """
    Vectorized random walk to compute TF scores.

    Args:
        receptor_tf: DataFrame with columns 'receptor' and 'tf'
        ggi_res: DataFrame with 'src' and 'dest' columns representing edges
        max_n_hop: maximum number of hops

    Returns:
        dict: normalized probability of reaching each TF from the receptor
    """
    # All nodes
    nodes = pd.Index(np.unique(ggi_res[['src', 'dest']].values.ravel()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # Adjacency matrix
    row = ggi_res['src'].map(node_to_idx).values
    col = ggi_res['dest'].map(node_to_idx).values
    data = np.ones(len(row))
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes), dtype=float)

    # Normalize rows to get transition probabilities
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero
    adj = adj.multiply(1 / row_sums[:, None])

    # Initialize probability vector (start at receptor)
    prob = np.zeros(n_nodes)
    receptor_idx = node_to_idx[receptor_tf['receptor'].iloc[0]]
    prob[receptor_idx] = 1.0

    # Nodes that are TFs
    tf_indices = [node_to_idx[tf] for tf in receptor_tf['tf']]

    # Accumulate probabilities at TFs
    tf_prob = np.zeros(len(tf_indices))

    # Perform walks via matrix multiplication
    current_prob = prob.copy()
    for hop in range(max_n_hop):
        current_prob = current_prob @ adj  # next hop probabilities
        tf_prob += current_prob[tf_indices]
    
    hop_tf = receptor_tf['n_hop'].values.astype(float)   # one hop per TF
    # adjust probability -inverse-hop weighting 
    tf_prob = tf_prob / hop_tf

    # Normalize by max_n_hop
    tf_prob = tf_prob / tf_prob.sum()

    return dict(zip(receptor_tf['tf'], tf_prob))


def calculate_TF_scores(adata, receptor_tf_temp):
    """
    Calculate TF scores for each cell in adata based on receptor_tf_temp info. (receptor to TF probability, TF gene expression, sum of target genes' gene expression)

    Args:
        adata: AnnData object with gene expression in adata.X and var_names as gene names
        receptor_tf_temp: pd.DataFrame with columns ['tf', 'target_genes', 'score', 'n_hop']

    Returns:
        np.ndarray: TF scores, shape (n_cells, 1)
    """

    gene_names = adata.var_names.to_list()
    n_cells = adata.n_obs

    # Initialize score vector for each cell
    tf_scores = np.zeros(n_cells)

    # Pre-build gene index mapping for quick lookup
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    for idx, row in receptor_tf_temp.iterrows():
        target_genes = row["target_genes"]
        tf_expr = adata[:, row["tf"]].X
        if hasattr(tf_expr, "toarray"):
            tf_expr = tf_expr.toarray()
        tf_expr = tf_expr.flatten()    # shape = (n_cells,)
        score_weight = row["score"] * tf_expr # (n_cells,)

        # Find indices of target genes in adata.var_names
        valid_genes = [g for g in target_genes if g in gene_to_idx]

        if not valid_genes:
            continue  # skip if no targets in dataset

        gene_indices = [gene_to_idx[g] for g in valid_genes]

        # Extract expression for these genes (cells x genes)
        # Assuming adata.X is sparse or dense matrix: slicing columns for these genes
        expr_subset = adata.X[:, gene_indices]

        # Sum expression across target genes for each cell
        # If sparse, use .sum(axis=1).A1 to get flat numpy array
        if hasattr(expr_subset, "toarray"):  # sparse matrix
            gene_sum = expr_subset.sum(axis=1).A1
        else:
            gene_sum = expr_subset.sum(axis=1)

        # Weighted contribution of this TF to each cell
        tf_scores += gene_sum * score_weight # (n_cells,)

    # Reshape to n_cells x 1 vector
    return tf_scores.reshape(-1, 1)

def generate_ggi_res(ggi_tf, receptor_name, adata, max_hop, co_exp_ratio):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = adata.var_names

    ggi_tf1 = ggi_tf[(ggi_tf['src'] == receptor_name) & ggi_tf['dest'].isin(gene_names)].drop_duplicates()
    if ggi_tf1.empty:
        return pd.DataFrame()

    ggi_res = []
    current = ggi_tf1.copy()
    current['hop'] = 1

    for hop in range(1, max_hop + 1):
        ggi_res.append(current)
        next_srcs = current['dest'].unique()
        current = ggi_tf[ggi_tf['src'].isin(next_srcs) & ggi_tf['dest'].isin(gene_names)].drop_duplicates()
        if current.empty:
            break
        current = current.copy()
        current['hop'] = hop + 1

    ggi_res = pd.concat(ggi_res, ignore_index=True).drop_duplicates()

    # Use vectorized batch version
    ggi_res = co_exp_batch_adata(adata, ggi_res, co_exp_ratio)

    return ggi_res


def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows],
        gc[newcols]])),(lr, lc))