# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:57:29 2019

@author: SY
"""

import numpy as np
import pandas as pd
import math

class Node():
    '''节点电压的结构体'''
    def __init__(self, comp, cat, _dict):
        assert cat in ['PQ', 'PU', 'BAL']
        self.cat = cat
        self.real = comp.real
        self.imag = comp.imag
        self.P = _dict['P'] if 'P' in _dict else 0
        self.Q = _dict['Q'] if 'Q' in _dict else 0
        self.U = _dict['U'] if 'U' in _dict else 0
        self.dP = 0
        self.dQ = 0
        self.dU = 0

class NewtonRaphsonAlgorithm():
    '''计算潮流的类'''
    def __init__(self, Y):
        self.Y = Y
        print('Size of the matrix is: {}'.format(Y.shape))
    
    def run(self, U, Z=None, sigma=1e-5, verbose=True, mantissa=5):
        n = len(self.Y)
        self.U = U
        tp_sum = np.zeros([2*n-2, 2]) # 暂存导纳矩阵的部分运算结果
        def _get_dW():
            for i in range(n-1):
                tp_sum[i] = [np.sum([self.Y[i][j].real*self.U[j].real - self.Y[i][j].imag*self.U[j].imag for j in range(n)]),
                             np.sum([self.Y[i][j].real*self.U[j].imag + self.Y[i][j].imag*self.U[j].real for j in range(n)])]
            for i in range(n-1):
                pi = self.U[i].real*tp_sum[i][0] + self.U[i].imag*tp_sum[i][1]
                qi = self.U[i].imag*tp_sum[i][0] - self.U[i].real*tp_sum[i][1]
                self.U[i].dP = self.U[i].P - pi
                if self.U[i].cat == 'PQ':
                    self.U[i].dQ = self.U[i].Q - qi
                elif self.U[i].cat == 'PU':
                    self.U[i].dU = self.U[i].U**2 - (self.U[i].real**2 + self.U[i].imag**2)
                else:
                    raise ValueError('BAL')
            dW = np.zeros([2*n-2])
            for i in range(n-1):
                dW[2*i] = self.U[i].dP
                dW[2*i+1] = self.U[i].dQ if self.U[i].cat == 'PQ' else self.U[i].dU
            return dW
            
        def _get_J():
            '''雅克比矩阵'''
            J = np.zeros([2*n-2,2*n-2])
            temp = np.zeros([2, 2])
            for i in range(n-1):
                for j in range(n-1):
                    if i != j:
                        temp[0][0] = - (self.Y[i][j].real*self.U[i].real + self.Y[i][j].imag*self.U[i].imag)
                        temp[0][1] = self.Y[i][j].imag*self.U[i].real - self.Y[i][j].real*self.U[i].imag
                        if self.U[i].cat == 'PQ':
                            temp[1][0] = temp[0][1]
                            temp[1][1] = - temp[0][0]
                        elif self.U[i].cat == 'PU':
                            temp[1][0], temp[1][1] = 0, 0
                        else:
                            raise ValueError('BAL')
                    else:
                        temp[0][0] = - tp_sum[i][0] - (self.Y[i][j].real*self.U[i].real + self.Y[i][j].imag*self.U[i].imag)
                        temp[0][1] = - tp_sum[i][1] + self.Y[i][j].imag*self.U[i].real - self.Y[i][j].real*self.U[i].imag
                        if self.U[i].cat == 'PQ':
                            temp[1][0] = tp_sum[i][1] + self.Y[i][j].imag*self.U[i].real - self.Y[i][j].real*self.U[i].imag
                            temp[1][1] = - tp_sum[i][0] + (self.Y[i][j].real*self.U[i].real + self.Y[i][j].imag*self.U[i].imag)
                        elif self.U[i].cat == 'PU':
                            temp[1][0] = -2*self.U[i].real
                            temp[1][1] = -2*self.U[i].imag
                        else:
                            raise ValueError('BAL')
                    for ii in range(2):
                        for jj in range(2):
                            J[2*i+ii][2*j+jj] = temp[ii][jj]
            return J
        
        def if_convergence(arr):
            for i in arr:
                if abs(i) > sigma:
                    return False
            return True
        
        # 计算
        self.U_k = {}
        self.dW_k = {}
        U_res = None
        for i in range(n-1):
            U_res = np.concatenate((U_res, np.array([self.U[i].real])), axis=0) if U_res is not None else np.array([self.U[i].real])
            U_res = np.concatenate((U_res, np.array([self.U[i].imag])), axis=0)
        dW = _get_dW()
        self.U_k[0] = U_res.tolist()
        self.dW_k[0] = dW.tolist()
        k = 0
        while(True):
            k += 1
            if if_convergence(dW): # 检查是否满足退出循环的条件
                break
            J = _get_J() # 雅克比矩阵
            J_I = np.mat(J).I # 转置
            res = - np.dot(J_I, dW) # 点乘
            # 更新参数
            U_res = U_res + np.array(res.tolist()[0])
            for i in range(n-1):
                self.U[i].real = U_res[2*i]
                self.U[i].imag = U_res[2*i+1]
            dW = _get_dW()
            self.U_k[k] = U_res.tolist()
            self.dW_k[k] = dW.tolist()
        # 输出部分，不用管
        if verbose:
            temp = complex(self.U[n-1].real, self.U[n-1].imag).conjugate() * np.sum([complex(self.Y[n-1][j].real,self.Y[n-1][j].imag).conjugate()*complex(self.U[j].real,self.U[j].imag).conjugate() for j in range(n)])
            self.U[n-1].P, self.U[n-1].Q = temp.real, temp.imag
            def exchange(x):
                if 'e' in str(x):
                    if '-' in str(x):
                        d = 3
                    else:
                        d = 2
                    return float(str(x).split('e')[0][:mantissa+d]+'e'+str(x).split('e')[1]) if len(str(x).split('e')[0])>mantissa+d else x
                else:
                    return round(x, mantissa)
            print('迭代过程中电压的变化：')
            for key in self.U_k.keys():
                for i in range(len(self.U_k[key])):
                    self.U_k[key][i] = exchange(self.U_k[key][i])
                print(key, self.U_k[key])
            print('迭代过程中节点不平衡量的变化：')
            for key in self.dW_k.keys():
                for i in range(len(self.dW_k[key])):
                    self.dW_k[key][i] = exchange(self.dW_k[key][i])
                print(key, self.dW_k[key])
            print('最终结果：')
            for i in range(n):
                if self.U[i].cat != 'BAL':
                    theta = math.atan(self.U[i].imag/self.U[i].real)
                    print('Node={}, U = {:.4f} + j{:.4f}, |U| = {:.4f}, theta = {:.4f}°, rad = {:.4f}'.format(i+2, self.U[i].real, self.U[i].imag,(self.U[i].real**2+self.U[i].imag**2)**0.5, theta/math.pi*180, theta))
                else:
                    print('平衡节点：P = {:.1f}MW, Q = {:.1f}MVar'.format(self.U[i].P*100, self.U[i].Q*100))
            # 有功损耗
            self.temp_df = pd.DataFrame(columns=['节点1', '节点2', '电流幅值'])
            def getPowerLoss(lst):
                real_node_1, real_node_2 = convert(lst[0]), convert(lst[1])
                u = complex(self.U[real_node_1].real, self.U[real_node_1].imag) - complex(self.U[real_node_2].real, self.U[real_node_2].imag)
                i = u / Z[real_node_1][real_node_2]
                return (u*(i.conjugate())).real, i
            SUM = 0
            for ix, item in enumerate([[1,2],[1,3],[1,5],[2,4], [4,5], [3,'G'], [4,'G'], [5,'G']]):
                temp, i = getPowerLoss(item)
                print('节点 {} 与节点 {} 的有功损耗 = {:.6f}, 线路电流 = {:.4f}∠{:.4f}'.format(item[0], item[1], temp, (i.real**2+i.imag**2)**0.5, math.atan(i.imag/i.real)/math.pi*180))
                SUM += temp
                self.temp_df.loc[ix] = item+[(i.real**2+i.imag**2)**0.5]
            print('所有节点的有功损耗 = {:.6f}'.format(SUM))

if __name__ == '__main__':    
    '''输入节点参数'''
    Z = np.array([[complex(0, 0)]*6]*6); B = np.array([[complex(0, 0)]*6]*6)
    Z[5][0] = Z[0][5] = complex(0.0011, 0.0064); B[5][0] = B[0][5] = complex(0, 35.27)
    Z[5][1] = Z[1][5] = complex(0.0015, 0.0086); B[5][1] = B[1][5] = complex(0, 47.62)
    Z[5][3] = Z[3][5] = complex(0.0033, 0.0190); B[5][3] = B[3][5] = complex(0, 26.46)
    Z[0][2] = Z[2][0] = complex(0.0020, 0.0110); B[0][2] = B[2][0] = complex(0, 15.88)
    Z[2][3] = Z[3][2] = complex(0.0023, 0.0130); B[2][3] = B[3][2] = complex(0, 18.517)
    Z[1][4] = Z[4][1] = complex(0.0014, 0.0080); B[1][4] = B[4][1] = complex(0, 44.10)
    Z[2][4] = Z[4][2] = complex(0.0022, 0.0130); B[2][4] = B[4][2] = complex(0, 17.64)
    Z[3][4] = Z[4][3] = complex(0.0019, 0.0110); B[3][4] = B[4][3] = complex(0, 14.99)
    
    '''N-1运行方式下的参数转换'''
    def N_1(Z, B, node_1, node_2, is_DoubleCircuit=True):
        if is_DoubleCircuit:
            B[node_1][node_2] = B[node_2][node_1] = B[node_1][node_2] / 2
            Z[node_1][node_2] = Z[node_2][node_1] = Z[node_1][node_2] * 2
        else:
            Z[node_1][node_2] = Z[node_2][node_1] = complex(0, 0)
            B[node_1][node_2] = B[node_2][node_1] = complex(0, 0)
        return Z, B
    
    '''节点名与实际下标的转换'''
    def convert(node):
        if node == 1:
            return 5
        if node in ['G', 0]:
            return 4
        return node - 2
    
    # =============================================================================
    # Z, B = N_1(Z, B, convert('G'), convert(3))
    # =============================================================================
    '''计算节点导纳矩阵'''
    def getY(Z, B):
        n = len(Z)
        Y = np.array([[complex(0, 0)]*n]*n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    Y[i][j] = np.sum([1/Z[i][k] if Z[i][k] != complex(0, 0) else complex(0, 0) for k in range(n)]) + np.sum([B[i][k] for k in range(n)])/1000
                else:
                    Y[i][j] = -(1/Z[i][j] + B[i][j]/1000 if Z[i][j] != complex(0, 0) else B[i][j]/1000)
        return Y
    Y = getY(Z,B)
    
    import time
    U = [Node(complex(1, 0), 'PQ', {'P':-4, 'Q':-2.2}),
         Node(complex(1, 0), 'PQ', {'P':-3.5, 'Q':-1.8}),
         Node(complex(1, 0), 'PQ', {'P':-3.2, 'Q':-1.4}),
         Node(complex(1, 0), 'PQ', {'P':-4.2, 'Q':-1.5}),
         Node(complex(1.05, 0), 'PU', {'P':7.65, 'U':1.05}),
         Node(complex(1.045, 0), 'BAL', {'U':1.045}),]
    U = np.array(U)
    t = time.time()
    model = NewtonRaphsonAlgorithm(Y)
    model.run(U, Z)
    print(f'耗时：{time.time()-t:.4f}s')

