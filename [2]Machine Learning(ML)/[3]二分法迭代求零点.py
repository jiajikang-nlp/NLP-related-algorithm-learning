"""
 author:jjk
 datetime:2019/5/2
 coding:utf-8
 project name:Pycharm_workstation
 Program function:
 
"""
def diSection(a,b,threshold,f):
    iter = 0
    while a < b:
        mid = a + abs(b-a)/2.0 # 求a b的中点
        if abs(f(mid)) < threshold:
            return mid
        if f(mid)*f(b) <0:
            a = mid
        if f(a)*f(mid) <0:
            b= mid
        iter+=1
        print(str(iter)+"a="+str(a)+",b="+str(b))

a = 5
b = 50
threshold = 1e-10
# f(x) = x*x-11*x+10,此方程的解：1,10，因此在（5,50）的解是10
# 调用函数，用于二分法求解
s = diSection(5,50,1e-10,lambda x:x*x-11*x+10)
print("result="+str(s))