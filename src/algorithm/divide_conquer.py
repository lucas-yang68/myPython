# -*- coding: utf-8 -*-
import logging
import pdb
from core.config import get_config
import threading
local1 = threading.local()  # 这是一个对象实例
cfg = get_config()
logger = logging.getLogger(__name__)


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # 分解：将数组分成两半
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # 递归：分别排序
    left = merge_sort(left)
    right = merge_sort(right)
    
    # 合并：合并两个有序数组
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

a = [3,45,0,4,5,7,7,9,0,0]