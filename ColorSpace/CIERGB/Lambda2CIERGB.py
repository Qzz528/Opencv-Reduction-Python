# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:51:19 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt

# data = pd.read_excel(r"F:\rgb.xlsx")
data = pd.DataFrame()
#三刺激值，实验数据，指定的λ的r,g,b构成
data['lambda'] = [380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780]
data['r'] = [3e-05, 5e-05, 0.0001, 0.00017, 0.0003, 0.00047, 0.00084, 0.00139, 0.00211, 0.00266, 0.00218, 0.00036, -0.00261, -0.00673, -0.01213, -0.01874, -0.02608, -0.03324, -0.03933, -0.04471, -0.04939, -0.05364, -0.05814, -0.06414, -0.07173, -0.0812, -0.08901, -0.09356, -0.09264, -0.08473, -0.07101, -0.05136, -0.03152, -0.00613, 0.02279, 0.05514, 0.0906, 0.1284, 0.16768, 0.20715, 0.24526, 0.27989, 0.30928, 0.33184, 0.34429, 0.34756, 0.33971, 0.32265, 0.29708, 0.26348, 0.22677, 0.19233, 0.15968, 0.12905, 0.10167, 0.07857, 0.05932, 0.04366, 0.03149, 0.02294, 0.01687, 0.01187, 0.00819, 0.00572, 0.0041, 0.00291, 0.0021, 0.00148, 0.00105, 0.00074, 0.00052, 0.00036, 0.00025, 0.00017, 0.00012, 8e-05, 6e-05, 4e-05, 3e-05, 1e-05, 0.0]
data['g'] = [-1e-05, -2e-05, -4e-05, -7e-05, -0.00014, -0.00022, -0.00041, -0.0007, -0.0011, -0.00143, -0.00119, -0.00021, 0.00149, 0.00379, 0.00678, 0.01046, 0.01485, 0.01977, 0.02538, 0.03183, 0.03914, 0.04713, 0.05689, 0.06948, 0.08536, 0.10593, 0.1286, 0.15262, 0.17468, 0.19113, 0.20317, 0.21083, 0.21466, 0.21487, 0.21178, 0.20588, 0.19702, 0.18522, 0.17087, 0.15429, 0.1361, 0.11686, 0.09754, 0.07909, 0.06246, 0.04776, 0.03557, 0.02583, 0.01828, 0.01253, 0.00833, 0.00537, 0.00334, 0.00199, 0.00116, 0.00066, 0.00037, 0.00021, 0.00011, 6e-05, 3e-05, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
data['b'] = [0.00117, 0.00189, 0.00359, 0.00647, 0.01214, 0.01969, 0.03707, 0.06637, 0.11541, 0.18575, 0.24769, 0.29012, 0.31228, 0.3186, 0.3167, 0.31166, 0.29821, 0.27295, 0.22991, 0.18592, 0.14494, 0.10968, 0.08257, 0.06246, 0.04776, 0.03688, 0.02698, 0.01842, 0.01221, 0.0083, 0.00549, 0.0032, 0.00146, 0.00023, -0.00058, -0.00105, -0.0013, -0.00138, -0.00135, -0.00123, -0.00108, -0.00093, -0.00079, -0.00063, -0.00049, -0.00038, -0.0003, -0.00022, -0.00015, -0.00011, -8e-05, -5e-05, -3e-05, -2e-05, -1e-05, -1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#根据实验数据插值，填充其他λ对应的rgb
f_r = scipy.interpolate.interp1d(data['lambda'],data['r'],kind='cubic')
f_g = scipy.interpolate.interp1d(data['lambda'],data['g'],kind='cubic')
f_b = scipy.interpolate.interp1d(data['lambda'],data['b'],kind='cubic')


#波长lamb的单色光，转换成人眼等效的rgb三色光组合 
#有的人眼等效颜色并没有对应的单色光，没有逆变换
def Lambda2CIERGB(lamb):
    #lamb float => return ndarray shape of (3,)
    #or lamb ndarray shape of (n,) => return ndarray shape of (n,3)
    return np.stack((f_r(lamb),f_g(lamb),f_b(lamb))).T


#对CIERGB值按三者和进行归一化，即将三维空间点CIERGB投影至二维平面x+y+z=1 
#只保留CIERGB三者的比例，丢失绝对强度信息，没有逆变换 
def CIE3Dto2D(src):
    #src ndarray shape of (3,) or (n,3) 
    #=> retrun ndarray same shape 
    if src.ndim==2:
        return src/src.sum(axis=-1)[:,None]
    else:
        return src/src.sum(axis=-1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    import matplotlib.gridspec as gsc
    from mpl_toolkits.mplot3d import Axes3D
    
    
    'Lambda => CIERGB'
    lamb = np.linspace(380,780,400)
    CIERGB = Lambda2CIERGB(lamb)
    plt.plot(lamb,CIERGB[:,0],'r-',label='r(λ)')
    plt.plot(data['lambda'],data['r'],'r.')
    plt.plot(lamb,CIERGB[:,1],'g-',label='g(λ)')
    plt.plot(data['lambda'],data['g'],'g.')
    plt.plot(lamb,CIERGB[:,2],'b-',label='b(λ)')
    plt.plot(data['lambda'],data['b'],'b.')
    plt.legend()
    plt.title('CIERGB tristimulus values')
    plt.show()


    
    #自创一个光谱图，便于后续展示，【注意】波长与颜色的对应并不严谨
    clist=['purple','purple','blue','cyan','lime','yellow','orange','red','red','red','red']
    newcmp = LinearSegmentedColormap.from_list('chaos',clist)
    lamb = np.linspace(380,780,400)


    'CIERGB space'
    gs = gsc.GridSpec(2,2,height_ratios=[1,0.2],width_ratios=[1,1])
    axs = [None for _ in range(3)]
    fig = plt.figure(figsize=(10,5))
    #光谱图
    mat = lamb.reshape(-1,1).T
    axs[0] = plt.subplot(gs[2]) 
    axs[0].imshow(mat,cmap=newcmp,extent=[380,780,-10,10])
    axs[0].get_yaxis().set_visible(False)
    axs[0].xaxis.tick_bottom()
    axs[0].set_title('spectrum')
    
    #三刺激值
    axs[1] = plt.subplot(gs[0])
    axs[1].plot(data['lambda'],data['r'],color = 'r' , label = 'r(λ)')
    axs[1].plot(data['lambda'],data['g'],color = 'g' , label = 'g(λ)')
    axs[1].plot(data['lambda'],data['b'],color = 'b' , label = 'b(λ)')
    axs[1].set_xlim(380,780)
    axs[1].legend()
    axs[1].set_title('CIERGB tristimulus values')
    

    #CIERGB空间的光谱
    axs[2] = plt.subplot(gs[1],projection='3d')
    axs[2].scatter3D(data['r'],data['g'],data['b'], c=data['lambda'], cmap=newcmp)  #绘制散点图
    axs[2].set_xlabel("R")
    axs[2].set_ylabel("G")
    axs[2].set_zlabel("B")
    axs[2].plot3D([0,0.5],[0,0],[0,0],'r')
    axs[2].plot3D([0,0],[0,0.5],[0,0],'lime')
    axs[2].plot3D([0,0],[0,0],[0,0.5],'b')
    axs[2].view_init(30, 70)
    axs[2].set_title('spectrum in CIERGB space')
    
    
    plt.show()
    
    
    
    'CIERGB => CIErgb'
    CIERGB = np.stack((data['r'],data['g'],data['b'])).T
    ciergb = CIE3Dto2D(CIERGB)
    
    gs = gsc.GridSpec(1,2,width_ratios=[1,1])
    axs = [None for _ in range(2)]
    
    fig = plt.figure(figsize=(9,5))
    #绘制CIERGB空间的光谱
    axs[0] = plt.subplot(gs[0],projection='3d')
    axs[0].scatter3D(data['r'],data['g'],data['b'], c=data['lambda'], cmap=newcmp, s=1)  #绘制散点图
    axs[0].set_xlabel("R")
    axs[0].set_ylabel("G")
    axs[0].set_zlabel("B")
    #绘制x+y+z=1且z>0的平面
    x = np.arange(-1.5,1.5,0.1)
    y = np.arange(-0.2,3,0.1)
    xx,yy = np.meshgrid(x,y)
    z = 1-xx-yy
    xx[z<-0.1] =np.nan
    yy[z<-0.1] =np.nan 
    z[z<-0.1] =np.nan
    axs[0].plot_surface(xx,yy,z,alpha=0.1)
    #将光谱上的点投影到x+y+z=1(z>0)平面
    for i in range(len(ciergb)):
        #延长线
        axs[0].plot3D([0,ciergb[i,0]],[0,ciergb[i,1]],[0,ciergb[i,2]],'gray', alpha=0.5, linewidth=0.5)
    #投影点
    axs[0].scatter3D(ciergb[:,0],ciergb[:,1],ciergb[:,2], c=data['lambda'], cmap=newcmp, alpha=0.5, s=10)
    #rgb光各1/3的白点(位于x+y+z=1平面)
    axs[0].scatter3D(1/3,1/3,1/3,color = 'gray',label="White", s = 10)
    axs[0].view_init(30, 70)
    axs[0].set_title('spectrum in CIERGB space')
    plt.legend()
    
    #绘制CIErg平面的光谱
    axs[1] = plt.subplot(gs[1])
    axs[1].scatter(ciergb[:,0],ciergb[:,1],c=data['lambda'], cmap=newcmp)
    axs[1].scatter(1/3,1/3,color = 'gray',label="White")
    axs[1].set_aspect(1)
    axs[1].set_xlabel("r")
    axs[1].set_ylabel("g")
    axs[1].set_title('spectrum in CIErgb 2D')
    plt.legend()
    
    plt.show()
    