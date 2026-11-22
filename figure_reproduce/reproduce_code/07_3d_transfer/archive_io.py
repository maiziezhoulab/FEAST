"""Read contiguous native row spans from an AnnData CSR HDF5 group."""
from scipy.sparse import csr_matrix


def csr_rows(group, lo, hi):
    pointers = group["indptr"][lo:hi + 1]
    start, stop = int(pointers[0]), int(pointers[-1])
    return csr_matrix((group["data"][start:stop], group["indices"][start:stop], pointers - start),
                      shape=(hi - lo, int(group.attrs["shape"][1])))
