import pandas as pd
import numpy as np


class Node():
    '''节点类'''
    def __init__(self,parent=None):
        '''初始化'''
        self.parent = parent # 父节点
        self.split_feature = None # 划分属性
        self.branchs = {} # 子节点字典
        self.label = None # 作为叶节点时存储的标记
    
    def add_branch(self,split_value,child_node):
        '''增加子节点'''
        self.branchs[split_value] = child_node
    
    def setlabel(self,label):
        '''设置标记'''
        self.label = label

    def isleaf(self):
        '''判断是否为叶节点'''
        if self.label is not None:
            return True
        else:
            return False

class ID3Tree():
    '''ID3决策树
        利用最大信息增益选择特征
        针对无缺失值的离散特征
    '''
    def getentroy(self,x):
        '''计算信息熵'''
        e = 0
        for i in x.unique():
            p = (x==i).sum() / x.count()
            e += -p * np.log2(p)
        return e
    
    def info_gain(self,x,y):
        '''计算信息增益'''
        gy = self.getentroy(y) # y的信息熵
        ce = 0 # y在x已知下的条件熵
        for i in x.unique():
            p = (x==i).sum() / x.count()
            sy = y[x==i]
            gsy = self.getentroy(sy)
            ce += p*gsy
        ig = gy - ce
        return ig

    def fit(self,Data:pd.DataFrame,epsilon:'信息增益阈值'):
        '''开始生成决策树'''
        self.Tree = Node() # 决策树初始化
        def create_tree(data:pd.DataFrame,node:Node):
            X = data.iloc[:,:-1] # 特征集
            m,n = X.shape
            y = data.iloc[:,-1]
            if len(y.unique()) == 1:
                # 停机条件1：标签集相同 设为叶节点
                node.setlabel(y.unique()[0])
                return
            if n == 0:
                # 停机条件2：特征集为空 设标签众数为叶节点
                node.setlabel(y.mode()[0])
                return
            # 计算各属性信息增益
            info_gain_list = []
            for i in range(n):
                x = X.iloc[:,i]
                info_gain = self.info_gain(x,y)
                info_gain_list.append(info_gain)
            max_info_gain = max(info_gain_list)
            if max_info_gain < epsilon:
                # 停机条件3：最大信息增益小于阈值 设标签众数为叶节点
                node.setlabel(y.mode()[0])
                return
            # bsfn:best_split_feature_number
            bsfn = info_gain_list.index(max_info_gain)
            # bsf:best_split_feature
            bsf = X.columns[bsfn]
            node.split_feature = bsf
            for i in X.iloc[:,bsfn].unique():
                # 对该特征下每一个唯一取值
                subnode = Node(parent=node) # 生成子节点
                node.add_branch(i,subnode)  # 父节点连接子节点
                subdata = data[X.iloc[:,bsfn]==i].drop(bsf,axis=1) # 构造新数据
                create_tree(subdata,subnode)
        create_tree(Data,self.Tree)
    
    def predcit(self,X:pd.DataFrame):
        '''预测'''
        def get_label(x:pd.Series,node:Node):
            if node.isleaf():
                return node.label
            else:
                sf = node.split_feature
                return get_label(x,node.branchs[x[sf]])
        m = X.shape[0]
        y = []
        for i in range(m):
            x = X.iloc[i,:]
            y.append(get_label(x,self.Tree))
        return np.array(y)
        

            
        
if __name__ == '__main__':
    data = pd.read_csv('glass.txt',sep=r'\s{4}',header=None,names=['x1','x2','x3','x4','y'],engine='python')
    #data = pd.read_csv('wm.csv')
    model = ID3Tree()
    model.fit(data,0.1)
    yh = model.predcit(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    print(yh)
    print((y==yh).sum()/len(y))