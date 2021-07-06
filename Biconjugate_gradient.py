def BCG(A,b,x0,max_iter,tol):
    n=A.shape
    r0=b-A.dot(x0)
    p0=r0.copy()
    rt0=np.zeros(n)
    rt0[0]=1/r0[0]
    pt0=rt0.copy()
    counter=0
    while counter!=max_iter and np.linalg.norm(r0)>tol:
        alpha=r0.dot(rt0)/A.dot(p0).dot(pt0)
        x0=x0+alpha*p0
        r1=r0-alpha*A.dot(p0)
        rt1=rt0-alpha*A.T.dot(pt0)
        beta=r1.dot(rt1)/r0.dot(rt0)
        p1=r1+beta*p0
        pt1=rt1+beta*pt0
        r0=r1
        rt0=rt1
        p0=p1
        pt0=pt1
        counter+=1
    return x0,counter
