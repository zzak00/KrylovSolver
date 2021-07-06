def NLBP(A,w,v,m):
    n=A.shape[0]
    #W=np.column_stack((np.zeros(n),w))
    #V=np.column_stack((np.zeros(n),v))
    V= np.zeros((n,m+2))
    V[:,1]=v
    W= np.zeros((n,m+2))
    W[:,1]=w
    beta=np.zeros(m+2)
    gama=np.zeros(m+2)
    alpha=np.zeros(m+2)
    add_last=np.zeros(m)
    for j in range(1,m+1):
        alpha[j]=np.dot(A.dot(V[:,j]),W[:,j])
        v=A.dot(V[:,j])-alpha[j]*V[:,j]-beta[j]*V[:,j-1]
        w=A.T.dot(W[:,j])-alpha[j]*W[:,j]-gama[j]*W[:,j-1]
        gama[j+1]=np.sqrt(np.abs(v.dot(w)))
        if gama[j+1]!=0:
            beta[j+1]=v.dot(w)/gama[j+1]
            W[:,j+1]=w/beta[j+1]
            V[:,j+1]=v/gama[j+1]
    Tm=np.zeros([m,m])
    add_last[-1]=gama[-1]
# Je vais l'ameliorer -> deux boucle imbriquee == un seul parcours
    Tm=np.row_stack((Tm,add_last))
    np.fill_diagonal(Tm,alpha[1:-1])
    np.fill_diagonal(Tm[1:,:-1],gama[2:-1])
    np.fill_diagonal(Tm[:-1,1:],beta[2:-1])
    return Tm,V[:,1:-1],W[:,1:-1]
