import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SVM():
    '''
    SVM的简单实现，使用线性核
    '''
    def __init__(self):
        self.w = None
        self.b = None
        self.alpha = None
    
    def get_w(self,alpha,x,y):
        n = x.shape[-1]
        temp = np.multiply(alpha,y) # shape = (m,1)
        result = np.multiply(temp,x).sum(axis=0).reshape(n,1)
        return result
    
    def get_g(self,w,x,b):
        m = x.shape[0]
        g = x@w + b
        g = g.reshape(m,1)
        return g
    
    def k(self,x1,x2):
        result = x1.T @ x2
        return result
    
    def get_loss(self,w,alpha):
        temp1 = w.T @ w     # 注意：temp1.shape = (1,1)
        temp1 = temp1[0,0] / 2
        temp2 = alpha.sum()
        result = temp1 - temp2
        return result

    def fit(self,x:'shape=(m,n)',y:'shape=(m,1)',C:">0",epsilon:'小正数 训练精度',max_iters:'最大迭代次数'):
        '''
        x.shape = (m,n)
        y.shape = (m,1)
        C 实数  对软间隔的惩罚系数  C越大、对误分类的容忍程度越小、超平面和分隔平面之间越“紧凑”、支持向量越少
        epsilon 实数 训练精度
        max_iters 最大迭代次数，避免因C值过大或epsilon过小而陷入死循环
        '''
        m,n = x.shape
        # 初始化拉格朗日算子 alpha
        alpha_new = np.zeros_like(y)
        alpha_new = alpha_new.astype(np.float64)
        # 因为这里alpha初始化为0向量，所以w也为0向量，损失函数也是0，省去一步运算
        w_new = np.zeros((n,1)).astype(np.float64)
        loss_new = 0
        b_new = 0
        # -----开始SMO算法-----
        # 根据KKT条件得出以下关于最优解的停机条件：
        #   当 alpha = 0 时，    y(w.T@x+b) > 1
        #   当 0 < alpha < C 时，y(w.T@x+b) = 1
        #   当 alpha > C 时，    y(w.T@x+b) < 1
        # 当然，上述条件只需要在精度 epsilon 范围内满足就可以
        iters = 0  
        while 1:
            alpha = alpha_new
            w = w_new
            b = b_new
            loss = loss_new
            support_samples_violate_KKT = {} # 存储违反KKT条件的支持向量样本 key=样本索引 value=违反KKT的程度
            common_samples_violate_KKT = {}
            g = self.get_g(w,x,b) # w.T@x+b
            values_compare_with_1 = np.multiply(y,g) # y(w.T@x+b)
            # --寻找违反KKT条件的样本并按支持向量和普通样本分开存储--
            for i in range(m):
                a = alpha[i,0]
                v = values_compare_with_1[i,0]
                if a == 0:
                    if v < 1-epsilon:
                        common_samples_violate_KKT[i] = 1-epsilon-v
                elif a > 0 and a < C:
                    if v < 1-epsilon or v > 1+epsilon:
                        support_samples_violate_KKT[i] = max(v-1-epsilon,1-epsilon-v)
                elif a == C:
                    if v > 1+epsilon:
                        support_samples_violate_KKT[i] = v-1-epsilon
            # 字典按值降序排列
            support_samples_violate_KKT = {i[0]:i[1] for i in sorted(support_samples_violate_KKT.items(),key=lambda x:x[1],reverse=True)}
            common_samples_violate_KKT = {i[0]:i[1] for i in sorted(common_samples_violate_KKT.items(),key=lambda x:x[1],reverse=True)} 
            # 字典合并：支持向量在前，普通样本在后；违反KKT严重的在前，不严重的在后
            all_samples_violate_KKT = support_samples_violate_KKT.copy()
            all_samples_violate_KKT.update(common_samples_violate_KKT)
            error = g-y # 误差
            print('The amount of samples violate KKT: %s' % str(len(all_samples_violate_KKT)))
            if not all_samples_violate_KKT:
                # 达到停机条件
                self.w = w
                self.b = b
                self.alpha = alpha
                print('所有样本均在精度 %s 下满足KKT条件' % str(epsilon))
                print('迭代次数：%s' % str(iters))
                return
            if iters == max_iters:
                self.w = w
                self.b = b
                self.alpha = alpha
                print('程序达到最大迭代次数而退出')
                return
            for s_1 in all_samples_violate_KKT.keys():
                # 用 s_1 表示选取的第一个样本
                print('Try_Sample_One: %s' % str(s_1))
                alpha_1 = alpha[s_1,0]
                e_1 = error[s_1,0]
                x1 = x[s_1]
                y1 = y[s_1,0]
                abs_all_sub_e1 = np.abs(error-e_1)
                # 将所有样本按 |e_1 - e_2| 降序排序                   
                abs_all_sub_e1 = [i[0] for i in sorted(enumerate(abs_all_sub_e1[:,0]),key=lambda x:x[1],reverse=True)]
                abs_all_sub_e1.remove(s_1)
                min_loss = loss
                found_sample_2 = -1
                for s_2 in abs_all_sub_e1:
                    if (x[s_2]==x[s_1]).all():
                        continue
                    print('For_Sample_One: %s -- Try_Sample_Two: %s' % (str(s_1),str(s_2)))
                    e_2 = error[s_2,0]
                    alpha_2 = alpha[s_2,0]
                    x2 = x[s_2]
                    y2 = y[s_2,0]
                    alpha_1_new_unc = alpha_1 - y1*(e_1-e_2)/(self.k(x1,x1)+self.k(x2,x2)-2*self.k(x1,x2))
                    low = min(alpha_1+alpha_2*y1*y2, alpha_1+alpha_2*y1*y2-C*y1*y2)
                    high = max(alpha_1+alpha_2*y1*y2, alpha_1+alpha_2*y1*y2-C*y1*y2)
                    L = max(0,low)
                    H = min(C,high)
                    # 裁剪alpha1
                    alpha_1_new = np.clip(alpha_1_new_unc,L,H)
                    alpha_2_new = (alpha_1-alpha_1_new)*y1*y2 + alpha_2
                    w_temp = w + y1*x1.reshape(n,1)*(alpha_1_new-alpha_1) + y2*x2.reshape(n,1)*(alpha_2_new-alpha_2)
                    alpha_temp = alpha.copy()
                    alpha_temp[s_1,0] = alpha_1_new
                    alpha_temp[s_2,0] = alpha_2_new
                    loss_temp = self.get_loss(w_temp,alpha_temp)
                    if loss_temp < min_loss:
                        found_sample_2 = s_2
                        min_loss = loss_temp
                        loss_new = loss_temp
                        w_new = w_temp
                        alpha_new = alpha_temp
                        b1 = y1 - x1.reshape(1,n)@w_new
                        b2 = y2 - x2.reshape(1,n)@w_new
                        b1 = b1[0,0]
                        b2 = b2[0,0]
                        if alpha_1_new > 0 and alpha_1_new < C:
                            b_new = b1
                        elif alpha_2_new > 0 and alpha_2_new < C:
                            b_new = b2
                        else:    
                            b_new = (b1+b2) / 2
                if found_sample_2 != -1:
                    # 说明找到了恰当的alpha2，跳出循环，进行下一次迭代
                    print('Found_Sample_One: %s -- Found_Sample_Two: %s' % (str(s_1),str(found_sample_2)))
                    print('alpha_%s: %s --> %s' % (str(s_1),str(alpha_1),str(alpha_new[s_1,0])))
                    print('alpha_%s: %s --> %s' % (str(found_sample_2),str(alpha[s_2,0]),str(alpha_new[s_2,0])))
                    break
                else:
                    # 说明没有找到恰当的alpha2，更换alpha1后重试
                    continue
            if found_sample_2 == -1:
                # 说明遍历所有样本都没有使目标函数下降
                # 接下来会进入死循环，应退出
                self.w = w
                self.b = b
                self.alpha = alpha
                print('程序在陷入死循环前退出，考虑降低C值或提高epsilon值')
                print('迭代次数：%s' % str(iters))
                return
            iters += 1

    def predict(self,x):
        x = np.array(x)
        m = x.shape[0]
        y = x@self.w + self.b
        y = np.array([1 if i>=0 else -1 for i in y]).reshape(m,1)
        return y



if __name__ == '__main__':
    data = pd.read_csv("iris.data",header=None,names=['x1','x2','x3','x4','y'])
    data = data[data.y != 'Iris-setosa']
    m = data.shape[0]
    y = data.values[:,-1]
    y = np.array([1 if i=='Iris-versicolor' else -1 for i in y]).reshape(m,1)
    # 选两个特征测试
    f1 = 'x1' 
    f2 = 'x3'
    x = data.loc[:,[f1,f2]].values
    model = SVM()
    model.fit(x, y, C=1, epsilon=0.001, max_iters=1000)  
    g = model.predict(x)
    print('accuracy:%.2f%%' % ((g==y).sum()*100/m))
    w = model.w
    b = model.b
    y0 = -(w[0,0]/w[1,0])*data[f1]-b/w[1,0]
    y1 = -(w[0,0]/w[1,0])*data[f1]-(b-1)/w[1,0]
    y_1 = -(w[0,0]/w[1,0])*data[f1]-(b+1)/w[1,0]
    plt.scatter(data[data['y']=='Iris-versicolor'][f1],data[data['y']=='Iris-versicolor'][f2],c='red')
    plt.scatter(data[data['y']=='Iris-virginica'][f1],data[data['y']=='Iris-virginica'][f2],c='green')
    plt.plot(data[f1],y0,c='blue')
    plt.plot(data[f1],y1,c='red')
    plt.plot(data[f1],y_1,c='green')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()














            
            
            







