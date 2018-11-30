# -*- coding: utf-8 -*-
'''
Created on 2018年4月16日
@author: Jason.F
@summary: SVM -序列最小优化(Sequential Minimal Optimization，SMO)，将大优化问题分解为多个小优化问题来求解。
'''
import time
import os
import numpy as np
import random
 
class SMO:
    
    def __init__(self,C,toler,maxIter):
        self.C=C #常数
        self.toler=toler#容错率
        self.maxIter=maxIter#退出迭代的最大循环次数
 
    def loadDataSet(self,fileName):
        dataMat =[]
        labelMat = []
        fr = open (fileName)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]),float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat,labelMat
    
    def selectJrand(self,i,m):#随机选择
        j=i
        while (j==i):
            j = int(random.uniform(0,m))
        return j
    
    def clipAlpha(self,aj,H,L):#最小不能超过L，最大不能超过H
        if aj>H:
            aj = H
        if L>aj:
            aj=L
        return aj

    def smoSimple(self,dataMatIn,classLabels):
        dataMatrix = np.mat(dataMatIn)
        labelMat = np.mat(classLabels).transpose()
        b=0
        m,n=np.shape(dataMatrix)
        alphas = np.mat(np.zeros((m,1)))
        iter=0
        while (iter<self.maxIter):
            alphaPairsChanged=0
            for i in range(m):
                fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T))+b
                Ei = fXi- float(labelMat[i])#误差
                if ( (labelMat[i]*Ei < -self.toler) and (alphas[i]<self.C)) or ((labelMat[i]*Ei >self.toler) and (alphas[i]>0)):
                    j = self.selectJrand(i,m)
                    fXj = float(np.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T))+b
                    Ej = fXj- float(labelMat[j])#误差
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    if (labelMat[i] != labelMat[j]):
                        L = max(0,alphas[j]-alphas[i])
                        H = min(self.C,self.C+alphas[j]-alphas[i])
                    else:
                        L = max(0,alphas[j]+alphas[i]-self.C)
                        H = min(self.C,alphas[j]+alphas[i])
                    if L==H:
                        print('L==H')
                        continue
                    eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                    if eta>=0:
                        print('eta>=0')
                        continue
                    alphas[j] -= labelMat[j] *(Ei - Ej)/eta
                    alphas[j] = self.clipAlpha(alphas[j], H, L)
                    if (abs(alphas[j]-alphaJold)<0.00001):
                        print('j not moving enough.')
                        continue
                    alphas[i] += labelMat[j] * labelMat[i] *(alphaJold-alphas[j])
                    b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:]*dataMatrix[j,:].T
                    b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T
                    if (0<alphas[i]) and (self.C > alphas[i]):
                        b =b1
                    elif (0<alphas[j]) and (self.C >alphas[j]):
                        b=b2
                    else :
                        b =(b1+b2)/2.0
                    alphaPairsChanged +=1
                    print('iter: %d i: %d, pairs changed %d' % (iter,i,alphaPairsChanged))
            if (alphaPairsChanged ==0 ) :
                iter +=1
            else:
                iter =0
            print('iteration number:%d'%iter)
        return b,alphas
    
    
if __name__ == "__main__":    
    start = time.clock() 
    
    homedir = os.getcwd()#获取当前文件的路径
    smo=SMO(C=0.6,toler=0.001,maxIter=50)#传递方法
    dataArr,labelArr = smo.loadDataSet(homedir+'/testSet.txt')
    b,alphas =smo.smoSimple(dataArr, labelArr)
    print (b)
    print (alphas)
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))
