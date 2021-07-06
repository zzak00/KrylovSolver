# GMRES A BASE DE LA FONCTION ARNOLDI DEJA DEFINI
def GMRES_R(A, b, x0, nmax_iter,tol):
    m=min(nmax_iter, A.shape[0])
    n = A.shape[0]
    rm=b-A.dot(x0)
    for j in range(min(nmax_iter, A.shape[0])):
        beta=np.linalg.norm(rm)
        V,h=arnoldi_iteration(A,rm,j+1)
        s=np.zeros(j+2)
        s[0]=beta
        y=np.linalg.lstsq(h,s,rcond=None)[0]
        xm=x0+V[:,:-1].dot(y)
        rm=b-A.dot(xm)
        if np.linalg.norm(rm)<=tol:
            return xm
        else:
            x0=xm
    return xm
