#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 07:57:40 2020

@author: usama
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ =='__main__':
    plt.close('all')
    savePlot=False and True
    loadFile='./Dataset/MgO.xlsx'
    savePath='./MgO_Result/DataViz/'
    df = pd.read_excel (loadFile)
    data=df.loc[:, df.columns != 'QE']
    label=df['QE']
    plt.close('all')
    length=11
    height=0.625*length

    d=np.matrix(data)
    p1=np.matmul(d.T,d)
    p1=p1.I
    p2=np.matmul(d,p1)
    H=np.matmul(p2,d.T)
    
    y=np.matrix(label)
    
    ypred=np.matmul(H,y.T)
    
    err=ypred.T - y
    err=np.array(err)[0]
    
    Hdiag=H.diagonal()
    Hdiag=np.array(Hdiag)[0]
    
    p=sum(Hdiag)
    n=len(Hdiag)
    H_star= 3 * (p+1)/n
    
#    H_star=0.3
    l=np.where(Hdiag>H_star)[0]
    l=np.append(l,np.where(abs(err)>3)[0])

    plt.figure(figsize=(length, height))
    
    if list(l) :
        outX=Hdiag[l]; outY=err[l]
        err=np.delete(err,l)
        Hdiag=np.delete(Hdiag,l)
        
        plt.scatter(outX,outY,label='Outliers',marker='x',c='r')

      
    plt.scatter(Hdiag,err,label='Data points')
    plt.hlines(3,xmin=0,xmax=1.2*H_star,colors='r',linestyles='--',label='Outlier Upper Limit')
    
    plt.hlines(-3,xmin=0,xmax=1.2*H_star,colors='r',linestyles='-.',label='Outlier Lower Limit')
    plt.vlines(H_star,ymin=-3.2,ymax=3.2,colors='g',linestyles='-.',label='Leverage Limit ({0:.3f})'.format(H_star))
    plt.ylim([-5,5])
    plt.xlim([0,1.25*H_star])
    plt.grid(True)
    plt.xlabel('Predicted Response',fontsize=16)
    plt.ylabel('Residuals',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    if savePlot:
        plt.savefig(savePath+'HatLev.png',quality=95)
        plt.savefig(savePath+'HatLev.jpg',quality=95)
        plt.savefig(savePath+'HatLev.eps',quality=95)
    
