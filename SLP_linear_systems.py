def SLLS(A, b, x0, nmax_iter,tol):
    n=A.shape
    rm=b-A.dot(x0)
    counter=0
    for m in range(min(nmax_iter, A.shape[0])):
        beta=np.linalg.norm(rm)
        v=rm / beta
        Tm,Vm,_,_=SLP(A,v,m+1)
        e1=np.zeros(m+1)
        e1[0]=1
        y=np.dot(beta*np.linalg.inv(Tm),e1)
        xm=x0+np.dot(Vm,y)
        rm=b-A.dot(xm)
        counter+=1
        if np.linalg.norm(rm)<=tol:
            return xm,counter
        else:
            x0=xm
    return xm,counter
