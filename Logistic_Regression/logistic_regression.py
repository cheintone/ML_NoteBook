import pandas as pd 
import numpy as np


class Logistic_Regression():
    '''
    带L2正则化项的逻辑回归的简单实现
    '''
    def __init__(self):
        self.theta = None # 参数 首个元素为截距项
        self.n_iter = None # 实际迭代次数
    
    def get_Hx(self,x,theta):
        '''
        计算H(x)
        x.shape = (m,n+1) 首列为常数项1
        theta.shape = (n+1,1)
        '''
        result = 1 / (1+np.exp(-x@theta))
        return result
    
    def get_Cost(self,x,y,theta,lamda,I):
        '''
        计算代价函数
        '''
        m = x.shape[0]
        Hx = self.get_Hx(x,theta).reshape(m,1)
        cost = -np.sum(np.multiply(y,np.log(Hx))+np.multiply(1-y,np.log(1-Hx))) + lamda/2*np.sum((I@theta)**2)
        return cost

    def fit(self,x,y,alpha,lamda,epsilon):
        '''
        x：二维特征矩阵或DataFrame shape=(m,n)
        y：标记矩阵或列表等
        alpha：学习率
        lamda：L2正则化系数系数
        epsilon：精度
        '''
        x = np.array(x)
        y = np.array(y)
        m,n = x.shape
        Ones = np.ones((m,1))
        x = np.hstack((Ones,x))
        y = y.reshape(m,1)
        initial_theta = np.zeros((n+1,1))
        I = np.eye(n+1)
        I[0,0] = 0
        initial_cost = self.get_Cost(x,y,initial_theta,lamda,I)
        print('初始代价为：%s' % str(initial_cost))
        last_cost = initial_cost + epsilon + 1
        cost = initial_cost
        theta_new = initial_theta
        c = 0
        while last_cost - cost > epsilon:
            last_cost = cost
            theta_old = theta_new
            theta_new = theta_old - alpha*(x.T@(self.get_Hx(x,theta_old)-y) + lamda*I@theta_old)
            cost = self.get_Cost(x,y,theta_new,lamda,I)
            c += 1
        else:
            if cost > last_cost:
                print('学习率不恰当！代价函数有变大趋势！')
                return None
            print('经过 %s 次迭代，代价收敛为：%s' % (str(c),str(cost)))
            self.theta = theta_new
            self.n_iter = c
    
    def predict(self,x):
        '''
        预测
        x：二维特征矩阵或DataFrame shape=(m,n)
        '''
        x = np.array(x)
        m,n = x.shape
        Ones = np.ones((m,1))
        x = np.hstack((Ones,x))
        Hx = self.get_Hx(x,self.theta)
        y_pre = np.array([1 if i>0.5 else 0 for i in Hx])
        return y_pre

if __name__ == '__main__':
    data = pd.read_csv('data.csv',header=None)
    m,n = data.shape
    x = data.values[:,:-1].reshape(m,n-1)
    y = data.values[:,-1]
    x = (x-x.mean(axis=0))/x.std(axis=0)
    model = Logistic_Regression()
    model.fit(x,y,alpha=0.1,lamda=0,epsilon=0.001)
    print(model.theta)
    print(y)
    print(model.predict(x))
    print(np.sum(model.predict(x)==y)/m)




