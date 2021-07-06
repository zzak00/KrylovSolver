def GKB(A,u,max_iter):
    alpha=np.linalg.norm(A.T.dot(u))
    v=A.T.dot(u)/alpha
    U=np.zeros([n,max_iter])
    U[:,0]=u
    V=np.zeros([n,max_iter])
    V[:,0]=v
    for j in range(max_iter-1):
        U[:,j+1]=A.dot(V[:,j])-alpha*U[:,j]
        beta=np.linalg.norm(U[:,j+1])
        U[:,j+1]=U[:,j+1]/beta
        V[:,j+1]=A.T.dot(U[:,j+1])-beta*V[:,j]
        alpha=np.linalg.norm(V[:,j+1])
        V[:,j+1]=V[:,j+1]/alpha
    return U,V
        
        
