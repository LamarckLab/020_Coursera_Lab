import numpy as np  # 导入库numpy用于数值计算
import matplotlib.pyplot as plt  # matplotlib.pyplot用于绘图


def costFunctionJ(x, y, theta):  # 代价函数Cost Function
    m = np.size(x, axis=0)  # 获取样本数量(统计数据集的行数), 存入m
    predictions = x * theta  # 计算预测值, x向量与θ向量相乘
    sqrErrors = np.multiply((predictions - y), (predictions - y))  # 计算残差平方
    j = 1 / (2 * m) * np.sum(sqrErrors)  # 计算代价函数的值
    return j


def gradientDescent(x, y, theta, alpha, num_iters):  # 梯度下降算法(alpha是学习率, num_iters是迭代次数))
    m = len(y)  # 计算样本数量
    n = len(theta)  # 计算θ的维度(参数个数)
    temp = np.mat(np.zeros([n, num_iters]))  # 创建一个全零数组, 用于存储后续每次迭代所更新的θ向量
    j_history = np.mat(np.zeros([num_iters, 1]))  # 创建了一个全零列向量, 用于存储后续每次迭代的代价值

    for i in range(num_iters):  # 循环进行梯度下降
        h = x * theta  # 计算每个x对应的预测值、
        # 更新θ值(梯度下降算法)
        temp[0, i] = theta[0, 0] - (alpha / m) * np.dot(x[:, 0].T, (h - y)).sum()
        temp[1, i] = theta[1, 0] - (alpha / m) * np.dot(x[:, 1].T, (h - y)).sum()
        theta = temp[:, i]  # 将更新后的值赋给θ变量
        j_history[i] = costFunctionJ(x, y, theta)  # 计算本次迭代的代价函数值
    return theta, j_history, temp


# x是一个12*2的矩阵, 第一列是偏置项, 全为0, 第二列是特征的值
x = np.mat([1, 3, 1, 4, 1, 6, 1, 5, 1, 1, 1, 4, 1, 3, 1, 4, 1, 3.5, 1, 4.5, 1, 2, 1, 5]).reshape(12, 2)
# 初始化参数θ的值
theta = np.mat([0, 1]).reshape(2, 1)
# y是一个12*1的列向量, 代表真实值
y = np.mat([1, 2, 3, 2.5, 1, 2, 2.2, 3, 1.5, 3, 1, 3]).reshape(12, 1)

# 求代价函数值
j = costFunctionJ(x, y, theta)

# 绘制训练前的图像
plt.figure(figsize=(15, 5))  # 创建画布并设置大小
plt.subplot(1, 2, 1)  # 画布被分成一行两列, 该子图是第一列
plt.scatter(np.array(x[:, 1])[:, 0], np.array(y[:, 0])[:, 0], c='r', label='real data')  # 绘制真实数据的散点
plt.plot(np.array(x[:, 1])[:, 0], x * theta, label='test data')  # 绘制预测值的直线(初始参数)
plt.legend(loc='best')
plt.title('before')

# 执行梯度下降算法的参数
theta, j_history, temp = gradientDescent(x, y, theta, 0.01, 100000)
print('最终j_history值：\n', j_history[-1])
print('最终theta值：\n', theta)
print('每次迭代的代价值：\n', j_history)
print('theta值更新历史：\n', temp)

# 执行梯度下降后的图像
plt.subplot(1, 2, 2)
plt.scatter(np.array(x[:, 1])[:, 0], np.array(y[:, 0])[:, 0], c='r', label='real data')  # 画梯度下降后的图像
plt.plot(np.array(x[:, 1])[:, 0], x * theta, label='predict data')
plt.legend(loc='best')
plt.title('after')
plt.show()
