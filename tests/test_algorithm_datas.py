# -*- coding: utf-8 -*-
import random
import time
import numpy as np
from typing import List, Union, Any
import heapq
import sys
from dataclasses import dataclass
from collections import Counter
import json

class TestArrayGenerator:
    """
    用于生成各种测试数组的工具类
    """
    
    def __init__(self, seed=42):
        """初始化随机种子"""
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_random_int_array(self, 
                                 size: int = 10000, 
                                 min_val: int = -10000, 
                                 max_val: int = 10000) -> List[int]:
        """
        生成随机整数数组
        
        参数:
            size: 数组大小
            min_val: 最小值
            max_val: 最大值
            
        返回:
            随机整数列表
        """
        return [random.randint(min_val, max_val) for _ in range(size)]
    
    def generate_sorted_array(self, 
                             size: int = 10000, 
                             ascending: bool = True) -> List[int]:
        """
        生成已排序数组
        
        参数:
            size: 数组大小
            ascending: 是否升序
            
        返回:
            已排序数组
        """
        arr = list(range(size))
        if not ascending:
            arr.reverse()
        return arr
    
    def generate_reverse_sorted_array(self, size: int = 10000) -> List[int]:
        """生成逆序数组"""
        return self.generate_sorted_array(size, ascending=False)
    
    def generate_nearly_sorted_array(self, 
                                    size: int = 10000, 
                                    swap_times: int = 100) -> List[int]:
        """
        生成几乎有序的数组
        
        参数:
            size: 数组大小
            swap_times: 随机交换次数
            
        返回:
            几乎有序数组
        """
        arr = list(range(size))
        for _ in range(swap_times):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    def generate_array_with_duplicates(self, 
                                      size: int = 10000, 
                                      unique_values: int = 100) -> List[int]:
        """
        生成包含大量重复元素的数组
        
        参数:
            size: 数组大小
            unique_values: 唯一值的数量
            
        返回:
            包含重复元素的数组
        """
        # 先生成少量唯一值
        unique_list = [random.randint(-1000, 1000) for _ in range(unique_values)]
        
        # 从这些值中随机选择创建大数组
        return [random.choice(unique_list) for _ in range(size)]
    
    def generate_power_of_two_array(self, 
                                   power: int = 14) -> List[int]:
        """
        生成2的幂次大小的数组
        
        参数:
            power: 2的幂次
            
        返回:
            大小为2^power的随机数组
        """
        size = 2 ** power
        return self.generate_random_int_array(size)
    
    def generate_mountain_array(self, size: int = 10000) -> List[int]:
        """
        生成山脉数组（先升后降）
        
        参数:
            size: 数组大小
            
        返回:
            山脉数组
        """
        peak = random.randint(size // 4, 3 * size // 4)
        arr = []
        
        # 上升部分
        for i in range(peak):
            arr.append(i * 2)
        
        # 下降部分
        for i in range(size - peak):
            arr.append((peak - i) * 2)
        
        return arr
    
    def generate_rotated_sorted_array(self, 
                                     size: int = 10000, 
                                     rotation: int = None) -> List[int]:
        """
        生成旋转有序数组
        
        参数:
            size: 数组大小
            rotation: 旋转位置，None表示随机
            
        返回:
            旋转有序数组
        """
        arr = list(range(size))
        if rotation is None:
            rotation = random.randint(0, size - 1)
        
        return arr[rotation:] + arr[:rotation]
    
    def generate_sparse_array(self, 
                             size: int = 10000, 
                             density: float = 0.1) -> List[int]:
        """
        生成稀疏数组（大部分为0）
        
        参数:
            size: 数组大小
            density: 非零元素密度
            
        返回:
            稀疏数组
        """
        arr = [0] * size
        non_zero_count = int(size * density)
        
        for _ in range(non_zero_count):
            idx = random.randint(0, size - 1)
            arr[idx] = random.randint(1, 1000)
        
        return arr
    
    def generate_consecutive_array_with_gap(self, 
                                           size: int = 10000, 
                                           gap_size: int = 100) -> List[int]:
        """
        生成连续但中间有空缺的数组
        
        参数:
            size: 数组大小
            gap_size: 空缺大小
            
        返回:
            有空缺的数组
        """
        arr = []
        for i in range(size):
            if i % gap_size == 0:
                arr.append(i)
            else:
                arr.append(-1)
        return arr
    
    def generate_float_array(self, 
                            size: int = 10000, 
                            min_val: float = -1000.0, 
                            max_val: float = 1000.0,
                            precision: int = 2) -> List[float]:
        """
        生成浮点数数组
        
        参数:
            size: 数组大小
            min_val: 最小值
            max_val: 最大值
            precision: 小数位数
            
        返回:
            浮点数列表
        """
        return [round(random.uniform(min_val, max_val), precision) 
                for _ in range(size)]
    
    def generate_string_array(self, 
                             size: int = 10000, 
                             min_len: int = 3, 
                             max_len: int = 10) -> List[str]:
        """
        生成随机字符串数组
        
        参数:
            size: 数组大小
            min_len: 最小长度
            max_len: 最大长度
            
        返回:
            字符串列表
        """
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return [''.join(random.choices(chars, k=random.randint(min_len, max_len)))
                for _ in range(size)]
    
    def generate_nested_array(self, 
                             size: int = 1000, 
                             max_depth: int = 5) -> List[Any]:
        """
        生成嵌套数组（用于测试递归算法）
        
        参数:
            size: 总元素数量（约数）
            max_depth: 最大嵌套深度
            
        返回:
            嵌套列表
        """
        def generate_nested(current_depth):
            if current_depth >= max_depth or random.random() < 0.3:
                return random.randint(1, 100)
            else:
                sub_size = random.randint(1, max(1, size // (2 ** current_depth)))
                return [generate_nested(current_depth + 1) 
                       for _ in range(sub_size)]
        
        return generate_nested(0)
    
    def generate_multidimensional_array(self, 
                                       shape: tuple = (100, 100)) -> np.ndarray:
        """
        生成多维数组
        
        参数:
            shape: 数组形状
            
        返回:
            numpy多维数组
        """
        return np.random.rand(*shape) * 100
    
    def generate_special_patterns(self) -> dict:
        """
        生成特殊模式的测试数组集合
        
        返回:
            包含各种特殊模式数组的字典
        """
        patterns = {
            # 1. 全部相同
            'all_same': [42] * 10000,
            
            # 2. 只有两个值交替
            'alternating': [1, 2] * 5000,
            
            # 3. 等差数列
            'arithmetic': list(range(0, 10000, 2)),
            
            # 4. 等比数列
            'geometric': [2**i for i in range(14)],  # 2^14 = 16384 > 10000
            
            # 5. 回文数组
            'palindrome': list(range(5000)) + list(range(5000))[::-1],
            
            # 6. 峰谷交替
            'peak_valley': [i if i % 2 == 0 else 10000 - i for i in range(10000)],
            
            # 7. 正弦波数据
            'sin_wave': [int(5000 + 5000 * np.sin(i / 100)) for i in range(10000)],
            
            # 8. 只有三个唯一值
            'three_values': [random.choice([-1, 0, 1]) for _ in range(10000)],
        }
        return patterns


class AlgorithmTester:
    """
    算法测试工具类
    """
    
    def __init__(self):
        self.generator = TestArrayGenerator()
        self.results = []
    
    def time_algorithm(self, 
                      algorithm_func, 
                      arr: List, 
                      description: str = "",
                      warmup: bool = True) -> dict:
        """
        测试算法运行时间
        
        参数:
            algorithm_func: 算法函数
            arr: 输入数组
            description: 算法描述
            warmup: 是否预热
            
        返回:
            包含测试结果的字典
        """
        # 预热（避免第一次运行因编译等开销影响结果）
        if warmup and len(arr) > 1000:
            test_arr = arr[:1000].copy() if hasattr(arr, 'copy') else arr[:1000]
            for _ in range(3):
                algorithm_func(test_arr.copy() if hasattr(arr, 'copy') else test_arr[:])
        
        # 正式测试
        test_arr = arr.copy() if hasattr(arr, 'copy') else arr[:]
        
        start_time = time.perf_counter()
        result = algorithm_func(test_arr)
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 验证结果（对排序算法）
        if description and "sort" in description.lower():
            is_sorted = all(test_arr[i] <= test_arr[i+1] for i in range(len(test_arr)-1))
            sorted_check = "✓" if is_sorted else "✗"
        else:
            sorted_check = "N/A"
        
        test_result = {
            "algorithm": description,
            "size": len(arr),
            "time_ms": round(elapsed, 3),
            "sorted_check": sorted_check,
            "result_sample": str(result)[:100] + "..." if isinstance(result, list) else str(result)
        }
        
        self.results.append(test_result)
        return test_result
    
    def compare_algorithms(self, 
                          algorithms: List[tuple], 
                          array_types: List[tuple],
                          sizes: List[int] = [1000, 10000, 50000]) -> None:
        """
        比较多个算法在不同数组上的性能
        
        参数:
            algorithms: [(函数, "描述"), ...]
            array_types: [("类型名", 生成函数), ...]
            sizes: 要测试的大小列表
        """
        print("=" * 80)
        print("算法性能比较测试")
        print("=" * 80)
        
        for size in sizes:
            print(f"\n数组大小: {size}")
            print("-" * 50)
            
            for arr_type_name, arr_generator in array_types:
                # 生成测试数组
                if arr_type_name == "sorted":
                    arr = self.generator.generate_sorted_array(size)
                elif arr_type_name == "reverse_sorted":
                    arr = self.generator.generate_reverse_sorted_array(size)
                elif arr_type_name == "random":
                    arr = self.generator.generate_random_int_array(size)
                elif arr_type_name == "nearly_sorted":
                    arr = self.generator.generate_nearly_sorted_array(size, size//100)
                else:
                    arr = arr_generator(size) if callable(arr_generator) else arr_generator
                
                print(f"\n数组类型: {arr_type_name}")
                print("-" * 30)
                
                for algorithm_func, algo_name in algorithms:
                    result = self.time_algorithm(algorithm_func, arr, algo_name)
                    print(f"  {algo_name:20s}: {result['time_ms']:8.2f} ms {result['sorted_check']}")
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report = "=" * 80 + "\n"
        report += "算法测试报告\n"
        report += "=" * 80 + "\n\n"
        
        # 按算法分组
        algo_groups = {}
        for result in self.results:
            algo = result["algorithm"]
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(result)
        
        # 生成报告
        for algo, tests in algo_groups.items():
            report += f"算法: {algo}\n"
            report += "-" * 50 + "\n"
            
            for test in tests:
                report += (f"  大小: {test['size']:8d} | "
                          f"时间: {test['time_ms']:8.2f} ms | "
                          f"排序检查: {test['sorted_check']}\n")
            report += "\n"
        
        return report


# ============================================================================
# 常用算法实现（用于测试）
# ============================================================================

def bubble_sort(arr):
    """冒泡排序"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    """归并排序"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
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

def heap_sort(arr):
    """堆排序"""
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

def linear_search(arr, target):
    """线性搜索"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def binary_search(arr, target):
    """二分搜索（要求数组已排序）"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def find_max(arr):
    """查找最大值"""
    return max(arr) if arr else None

def find_min(arr):
    """查找最小值"""
    return min(arr) if arr else None

def calculate_sum(arr):
    """计算数组和"""
    return sum(arr)

def calculate_average(arr):
    """计算平均值"""
    return sum(arr) / len(arr) if arr else 0


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数：演示如何使用测试工具"""
    
    # 1. 创建生成器
    generator = TestArrayGenerator()
    
    print("生成各种测试数组示例:")
    print("=" * 60)
    
    # 2. 生成不同类型的数组
    test_arrays = {
        "随机数组": generator.generate_random_int_array(20),
        "已排序数组": generator.generate_sorted_array(20),
        "逆序数组": generator.generate_reverse_sorted_array(20),
        "几乎有序数组": generator.generate_nearly_sorted_array(20, 5),
        "重复元素数组": generator.generate_array_with_duplicates(20, 5),
        "山脉数组": generator.generate_mountain_array(20),
        "旋转有序数组": generator.generate_rotated_sorted_array(20, 7),
        "稀疏数组": generator.generate_sparse_array(20, 0.2),
    }
    
    # 显示示例
    for name, arr in test_arrays.items():
        print(f"{name:15s}: {str(arr)[:50]}...")
    
    print("\n" + "=" * 60)
    print("性能测试示例:")
    print("=" * 60)
    
    # 3. 性能测试
    tester = AlgorithmTester()
    
    # 定义要测试的算法
    algorithms = [
        (bubble_sort, "冒泡排序"),
        (quick_sort, "快速排序"),
        (merge_sort, "归并排序"),
        (heap_sort, "堆排序"),
    ]
    
    # 定义要测试的数组类型
    array_types = [
        ("random", generator.generate_random_int_array),
        ("sorted", generator.generate_sorted_array),
        ("reverse_sorted", generator.generate_reverse_sorted_array),
        ("nearly_sorted", lambda size: generator.generate_nearly_sorted_array(size, size//100)),
    ]
    
    # 运行测试
    tester.compare_algorithms(algorithms, array_types, [1000, 2000])
    
    # 4. 生成报告
    print("\n" + "=" * 60)
    print("详细测试报告:")
    print("=" * 60)
    print(tester.generate_report())
    
    # 5. 生成大数组用于手动测试
    print("\n" + "=" * 60)
    print("生成大数组保存到文件:")
    print("=" * 60)
    
    # 生成各种大数组
    big_arrays = {
        "large_random_100k": generator.generate_random_int_array(100000),
        "large_sorted_100k": generator.generate_sorted_array(100000),
        "large_nearly_sorted_100k": generator.generate_nearly_sorted_array(100000, 1000),
        "large_with_duplicates_100k": generator.generate_array_with_duplicates(100000, 100),
    }
    
    # 保存到文件
    for name, arr in big_arrays.items():
        filename = f"{name}.txt"
        with open(filename, 'w') as f:
            # 写入数组信息
            f.write(f"# Array: {name}\n")
            f.write(f"# Size: {len(arr)}\n")
            f.write(f"# Min: {min(arr)}\n")
            f.write(f"# Max: {max(arr)}\n")
            f.write(f"# Average: {sum(arr)/len(arr):.2f}\n")
            f.write("# Data:\n")
            
            # 写入数据（每行10个）
            for i in range(0, len(arr), 10):
                f.write(' '.join(str(x) for x in arr[i:i+10]) + '\n')
        
        print(f"已保存: {filename} ({len(arr)} 个元素)")
    
    # 6. 特殊模式数组
    print("\n" + "=" * 60)
    print("特殊模式数组:")
    print("=" * 60)
    
    special_patterns = generator.generate_special_patterns()
    for pattern_name, arr in special_patterns.items():
        print(f"{pattern_name:15s}: 大小={len(arr)}, 示例={str(arr[:5]) if len(arr) > 5 else arr}")


if __name__ == "__main__":
    main()
