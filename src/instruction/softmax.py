'''
在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。
它能将一个含任意实数的K维的向量“压缩”到另一个K维实向量中，使得每一个元素的范围都在(0,1)之间，并且所有元素的和为1。
softmax函数的特点和作用：https://www.zhihu.com/question/23765351
'''

import math


z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]  # exp返回x的指数，e的x次方
print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
sum_z_exp = sum(z_exp)
print(sum_z_exp)  # Result: 114.98
softmax = [round(i / sum_z_exp, 3) for i in z_exp]  # round()方法返回浮点数x的四舍五入值，3表示保留三位小数
print(softmax)  # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
