
# 计算组合数
from scipy.special import comb,perm
import numpy as np
import matplotlib.pyplot as plt

# 二项分布概率计算公式
def getp(m,n,pa):
    if m<n:
        return 0.0
    return comb(m,n)*(pa**n)*((1-pa)**(m-n))
# 获取画图数据
klist = np.arange(21)
plist = [getp(m=20,n=k,pa=0.75) for k in klist]
plt.plot(klist,plist) # klist:x轴，plist：y轴

plt.xlabel('number of good apples')
plt.ylabel('k-distribution proba')
plt.title('distribution proba')

plt.xticks(np.arange(0,22,1))
plt.grid()
plt.show()


