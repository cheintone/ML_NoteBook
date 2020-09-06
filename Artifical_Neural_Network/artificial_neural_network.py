import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ANN():
    '''
    人工神经网络的简单实现
    '''

    def __init__(self):
        self.layers_list = None # 神经元个数列表
        self.w_list = None # 权重矩阵列表
        self.b_list = None # 阈值矩阵列表

    def sigmoid(self,x):
        '''
        Sigmoid激活函数
        '''
        result = 1 / (1+np.exp(-x))
        return result

    def fit(self,x,y,layers_list,alpha,epsilon):
        '''
        x 特征矩阵 shape = (m,n) m为样本数 n为特征数
        y 标记矩阵 shape = (m,1) m为样本数 标记为 0或1
        layers_list 神经元个数列表 [n,……,1] 首个元素为特征数n 最后一个元素为1 中间元素为各个隐藏层神经元个数
        alpha 学习率
        epsilon 计算精度
        '''
        m,n = x.shape # 样本数 特征数
        L = len(layers_list) # 神经网络层数
        k = layers_list[-1] # 输出层神经元个数
        # 初始化权重矩阵 -1到1之间的均匀分布 shape = (本层神经元个数，上一层神经元个数)
        last_w_list = [np.random.uniform(low=-1,high=1,size=(layers_list[i+1],layers_list[i])) for i in range(L-1)]
        # 初始化阈值为0 shape = (本层神经元个数，1)
        last_b_list = [np.zeros((layers_list[i+1],1)) for i in range(L-1)]
        print('已初始化权重矩阵和阈值矩阵')
        # 计算代价
        input_list = [x.T]
        output_list = [x.T] # 输入层的输入和输出都是x本身
        print('开始计算代价')
        for i in range(L-1): # 对每一个隐藏层和输出层
            w = last_w_list[i] # 上一层至当前层的权重矩阵
            b = last_b_list[i] # 当前层的阈值矩阵
            c = b.shape[0] # 当前层的神经元个数
            z = w @ output_list[-1] + b
            z = z.reshape(c,m) # 当前层的输入矩阵
            input_list.append(z)
            a = self.sigmoid(z)
            a = a.reshape(z.shape) # 当前层的输出矩阵
            output_list.append(a)
        y_head = output_list[-1] # 输出层的输出 shape = (k,m) 区间 = (0,1)
        initial_accuracy = ((y_head.T>0.5)+0==y).sum() / m
        accuracy_list = [initial_accuracy]
        next_cost = -np.mean(np.multiply(y,np.log(y_head.T)) + np.multiply(1-y,np.log(1-y_head.T)))
        print('初始代价：%s' % str(next_cost))
        cost_list = [next_cost]
        # 定义一个无穷大的初始代价
        last_cost = np.inf
        count = 0 # 迭代次数
        while last_cost - next_cost > epsilon: # 当代价的减小量高于设定的精度epsilon，继续迭代
            count += 1
            print('正在进行第 %s 次迭代……' % str(count))
            last_cost = next_cost # 将当前代价设定为上一轮代价
            # 反向传播 梯度更新
            # 计算中间变量：损失函数对输出层的输入矩阵的梯度
            z = input_list[-1]
            temp = (y_head - y.T) / m
            gradient_w_list = [temp@output_list[-2].T] # 权重矩阵梯度列表
            gradient_b_list = [temp.mean(axis=1).reshape(k,1)] # 阈值矩阵梯度列表
            for i in range(L-3,-1,-1): # 对于所有隐藏层,逆向计算
                c = layers_list[i+1] # 当前层神经元个数
                next_w = last_w_list[i+1] # 当前层至下一层的权重矩阵
                z = input_list[i+1] # 当前层的输入矩阵
                temp = np.multiply(next_w.T@temp,np.multiply(self.sigmoid(z),1-self.sigmoid(z)))
                last_a = output_list[i] # 上一层的输出矩阵
                gradient_w = temp@last_a.T # 当前层权重矩阵梯度
                gradient_b = temp.mean(axis=1).reshape(c,1) # 当前层阈值矩阵梯度
                # 在梯度列表最前面添加当前层梯度
                gradient_w_list.insert(0,gradient_w)
                gradient_b_list.insert(0,gradient_b)
            # 梯度更新
            last_w_list = [i-alpha*j for i,j in zip(last_w_list,gradient_w_list)]
            last_b_list = [i-alpha*j for i,j in zip(last_b_list,gradient_b_list)]
            # 接下来进行前向传播，得到新代价，并记录每层的输入和输出
            # 计算代价
            input_list = [x.T]
            output_list = [x.T] # 输入层的输入和输出都是x本身
            for i in range(L-1): # 对每一个隐藏层和输出层
                w = last_w_list[i] # 上一层至当前层的权重矩阵
                b = last_b_list[i] # 当前层的阈值矩阵
                c = b.shape[0] # 当前层的神经元个数
                z = w @ output_list[-1] + b
                z = z.reshape(c,m) # 当前层的输入矩阵
                input_list.append(z)
                a = self.sigmoid(z)
                a = a.reshape(z.shape) # 当前层的输出矩阵
                output_list.append(a)
            y_head = output_list[-1] # 输出层的输出 shape = (k,m) 区间 = (0,1)
            accuracy = ((y_head.T>0.5)+0==y).sum() / m
            accuracy_list.append(accuracy)
            next_cost = -np.mean(np.multiply(y,np.log(y_head.T)) + np.multiply(1-y,np.log(1-y_head.T)))
            print('代价计算为：%s' % next_cost)  
            cost_list.append(next_cost)
        # 循环结束
        if next_cost > last_cost:
            print('学习率不恰当！代价函数有变大趋势！')
            return None
        print('经过 %s 次迭代，代价函数收敛为：%s' % (str(count),str(next_cost)))
        fig,ax1=plt.subplots()
        ax1.plot(range(count+1),cost_list,label='cost',c='r')
        ax1.set_xlabel('n_iters')
        ax1.set_ylabel('cost',c='r')
        ax1.set_title('Cost & Accuracy')
        ax2=ax1.twinx()
        ax2.plot(range(count+1),accuracy_list,label='accuracy',c='g')
        ax2.set_ylabel('accuracy',c='g')
        plt.show()
        self.layers_list = layers_list
        self.w_list = last_w_list
        self.b_list = last_b_list

    def predict(self,x):
        '''
        预测
        '''
        L = len(self.w_list)
        a = x.T
        for i in range(L):
            w = self.w_list[i] 
            b = self.b_list[i]
            z = w@a + b
            a = self.sigmoid(z)
        output = a.T > 0.5
        return output+0

        
          


if __name__ == '__main__':
    from sklearn import datasets
    x,y = datasets.make_classification(50000,4,n_classes=2)
    y = y.reshape(50000,1)
    model = ANN()
    model.fit(x,y,[4,5,1],0.05,0.00001)
    output = model.predict(x)
    accuracy = (output==y).sum() / x.shape[0]
    print('final accuracy：%s' % accuracy)
        