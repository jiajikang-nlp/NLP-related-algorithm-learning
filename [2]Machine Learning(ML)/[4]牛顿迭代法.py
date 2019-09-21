"""
 author:jjk
 datetime:2019/5/2
 coding:utf-8
 project name:Pycharm_workstation
 Program function: https://mp.csdn.net/mdeditor/99714602
 
"""
def newton(x0,niter,threshold,f,fgrad):
    iter = 0 # 记录每次迭代过程标记符
    while niter>0:
        x1 = x0-f(x0)/fgrad(x0) # 获取一次近似值
        if f(x1)<threshold:
            break
        x0 = x1# 循环求解二次近似值
        iter+=1
        print(str(iter)+" x= "+str(x1)) # 打印每次迭代结果
        niter-=1
    return x1

# x0 = 50
# niter = 100
# threshold=1e-10
# f(x) = x*x-11*x+10, 这个方程的解： 1，10
# f1(x) = 2*x-11 表示一阶导数
s = newton(50,100,1e-10,lambda x:x*x-11*x+10,lambda x:2*x-11)


