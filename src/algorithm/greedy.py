# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 20:41:47 2025

@author: Administrator
"""
'''
1. 贪心选择性质
可以通过局部最优选择来构造全局最优解。
即：整体最优解可以通过一系列局部最优选择得到。
2. 最优子结构
问题的最优解包含其子问题的最优解。
'''
import logging
import pdb
from core.config import get_config
import threading
local1 = threading.local()  # 这是一个对象实例
cfg = get_config()
logger = logging.getLogger(__name__)
'''
def greedy(problem):
    solution = []  # 存储解
    while not is_complete(solution, problem):  # 未达到完整解
        # 1. 选择当前最优的候选
        candidate = select_best_candidate(problem)
        
        # 2. 检查是否可行
        if is_feasible(candidate, solution):
            solution.append(candidate)
    
    return solution
'''
#零钱兑换问题（特定面值时）
def coin_change(coins, amount):
    coins.sort(reverse=True)  # 从大到小排序
    count = 0
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
    return count if amount == 0 else -1

#活动选择问题
def activity_selection(start, finish):
    # 按结束时间排序
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    selected = [activities[0]]
    
    for i in range(1, len(activities)):
        # 如果当前活动开始时间 >= 上一个选中活动的结束时间
        if activities[i][0] >= selected[-1][1]:
            selected.append(activities[i])
    
    return selected
def select_activities(start, finish):
    # 按结束时间排序
    activities = sorted(zip(start, finish), key=lambda x: x[1])
    
    selected = []
    last_finish = 0  # 上一个选择活动的结束时间
    print(f'activities is {activities}')
    for s, f in activities:
        print(f' s={s} f={f} last_finish={last_finish}')
        if s >= last_finish:  # 活动开始时间不早于上一个活动的结束时间
            selected.append((s, f))
            print(f'select s ={s} f={f}')
            last_finish = f
    
    return selected

# 使用示例
start = [1, 3, 0, 5, 8, 5]
finish = [1, 4, 2, 5, 9, 3]

result = select_activities(start, finish)
print(result)  # 输出：[(1, 2), (3, 4), (5, 7), (8, 9)]

import heapq
#霍夫曼编码
def huffman_coding(freq):
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return heap[0][1:]