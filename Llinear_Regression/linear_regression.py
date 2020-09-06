import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression():

    def __init__(self):
        self.theta = None
        self.n_iters = None

    def fit(self, data, alpha, lamda, epsilon):
        # 梯度下降求解
        data = np.array(data)
        m = data.shape[0]
        n = data.shape[-1]-1
        X = data[:,:-1].reshape(m,n)
        Y = data[:,-1].reshape(m,1)
        Ones = np.ones_like(Y)
        X = np.hstack((Ones,X))
        Theta = np.zeros((n+1,1))
        E = np.eye(n+1)
        E[0,0] = 0
        initial_cost = 0.5/m*np.square(np.linalg.norm(X@Theta-Y))+ 0.5*lamda*np.square(np.linalg.norm(E@Theta))
        print('初始代价：%s' % str(initial_cost))
        last_cost = initial_cost + epsilon + 1
        Theta_new = Theta
        cost = initial_cost
        c = 0
        while last_cost - cost > epsilon:
            last_cost = cost
            Theta_old = Theta_new
            Theta_new = Theta_old - alpha * (1/m*X.T@(X@Theta_old-Y)+lamda*E@Theta_old)
            cost = 0.5/m*np.square(np.linalg.norm(X@Theta_new-Y))+ 0.5*lamda*np.square(np.linalg.norm(E@Theta_new))
            c += 1
        else:
            if cost > last_cost:
                print('学习率不恰当！代价函数有变大趋势！')
                return None
            print('经过 %s 次迭代，代价收敛为：%s' % (str(c),str(cost)))
            self.theta = Theta_new
            self.n_iters = c
    
    def predict(self,x):
        x = np.array(x)
        m,n = x.shape
        Ones = np.ones((m,1))
        x = np.hstack((Ones,x))
        y_pre = x@self.theta
        return y_pre

    def solve(self, data, lamda):
        # 正规方程求解
        data = np.array(data)
        m = data.shape[0]
        n = data.shape[1]-1
        x = data[:,:-1].reshape(m,n)
        y = data[:,-1].reshape(m,1)
        x = np.hstack((np.ones_like(y),x))
        I = np.eye(n+1)
        I[0,0] = 0
        result = np.linalg.inv(x.T@x+lamda*I)@x.T@y
        return result       


if __name__ == '__main__':
    x = np.random.randint(-10,10,size=(5000,2))
    y = x[:,0]+2*x[:,1]+5+np.random.rand(x.shape[0])
    y = y.reshape(x.shape[0],1)
    data = np.hstack((x,y))
    model = Linear_Regression()
    model.fit(data,0.01,0.1,0.0001)
    print(model.theta)
    result = model.solve(data,0.1)
    print(result)
    print(model.predict(x))
    