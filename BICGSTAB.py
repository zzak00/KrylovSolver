def BICGSTAB(A,b,x0,rt0,max_iter,tol):
    n=A.shape[0]
    r0=b-A.dot(x0)
    p0=b-A.dot(x0)
    counter=0
    while counter!=max_iter and np.linalg.norm(r0)>tol:
        alpha=r0.dot(rt0)/A.dot(p0).dot(rt0)
        s=r0-alpha*A.dot(p0)
        w=A.dot(s).dot(s)/np.dot(A.dot(s),A.dot(s))
        x0=x0+alpha*p0+w*s
        r1=s-w*(A.dot(s))
        beta=(r1.dot(rt0)/r0.dot(rt0))*alpha/w
        p1=r1+beta*(p0-w*A.dot(p0))
        r0=r1
        p0=p1
        counter+=1
    return x0,counter
