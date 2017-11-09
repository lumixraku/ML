#!/usr/bin/env python
# coding=utf8


import matplotlib.pyplot as plt
import numpy as np

def main():
    print 'main'
    x = np.linspace(0, 2 * np.pi, 50)
    plt.plot(x, np.sin(x)) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
    plt.show() # 显示图形

if __name__ == "__main__":
    main()