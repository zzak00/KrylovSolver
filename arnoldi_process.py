def arnoldi_iteration(A,r0,M):
    n = A.shape[0]
    h = np.zeros((M + 1, M))
    V= np.zeros((n,M+1))
    V[:,0] = r0 / np.linalg.norm(r0)

    for j in range(M):
        w = A.dot(V[:,j])
        for i in range(j + 1):
            h[i, j] = np.dot(w,V[:,i])
            w = w - h[i, j] * V[:,i]

        h[j + 1, j] = np.linalg.norm(w)
        if h[j + 1, j] == 0:
            return V,h
        else:
            V[:,j+1] = w / h[j + 1, j]
    return V,h

 """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """
