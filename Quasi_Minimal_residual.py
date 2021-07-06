def QMR(A,b,x0,max_iter):
    r0=b-A.dot(x0)
    beta=np.linalg.norm(r0)
    w=r0/beta
    v=r0/beta
    for i in range(1,max_iter+1):
        Tm,V,_=NLBP(A,w,v,i+1)
        s=np.zeros(i+2)
        s[0]=beta
        y=np.linalg.lstsq(Tm,s,rcond=None)[0]
        xm=x0+V.dot(y)
    return xm
        
    
