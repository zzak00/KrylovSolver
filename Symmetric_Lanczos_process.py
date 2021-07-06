def SLP(A,v,m):
    n = A.shape[0]
    V= np.zeros((n,m+2))
    V[:,1]=v
    beta=np.zeros(m+2)
    alpha=np.zeros(m+2)
    for j in range(1,m+1):
        w = A.dot(V[:,j])-beta[j]*V[:,j-1]
        alpha[j]=np.dot(w,V[:,j])
        w=w-alpha[j]*V[:,j]
        beta[j+1]=np.linalg.norm(w)
        if beta[j+1]==0:
            return V,alpha,beta
        else:
            V[:,j+1]=w/beta[j+1]
    Tm=np.zeros([m,m])
    np.fill_diagonal(Tm,alpha[1:-1])
    np.fill_diagonal(Tm[1:,:-1],beta[2:-1])
    np.fill_diagonal(Tm[:-1,1:],beta[2:-1])
    return Tm,V[:,1:-1],alpha,beta
    
