import numpy as np

def topsis(X,w,direction):
    '''输入决策矩阵X、权重向量w、方向向量direction,
    return (C, order, D_plus, D_minus)  — 贴近度/排序/正负距离'''
    # 向量归一化
    Xn=X / np.sqrt((X**2).sum(axis=0))
    
    Xb = np.where(direction[:, None] < 0, 1.0 / np.maximum(Xn, 1e-12), Xn)
    V = Xb * w
    
    A_plus  = V.max(axis=0)   # 每列最优值，"完美方案"
    A_minus = V.min(axis=0)   # 每列最差值，"最差方案"
    
    D_plus  = np.sqrt(((V - A_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))
    
    C = D_minus / (D_plus + D_minus)
    order = np.argsort(-C)     # C 越大越好
    
    return C, order, D_plus, D_minus