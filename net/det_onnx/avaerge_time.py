import time
from functools import wraps

def average_execution_time(func):
    total_time = 0.0
    num_calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal total_time, num_calls
        start_time = time.perf_counter()  # 记录开始时间
        result = func(*args,**kwargs)    # 执行原函数
        end_time = time.perf_counter()    # 记录结束时间
        elapsed = end_time - start_time   # 计算单次耗时
        total_time += elapsed             # 累加总耗时
        num_calls += 1                    # 增加调用次数
        return result
    def reset():
        nonlocal total_time, num_calls
        total_time = 0.0
        num_calls = 0
    def get_average():
        """返回当前平均执行时间（单位：秒）"""
        return total_time / num_calls if num_calls > 0 else 0.0

    wrapper.get_average = get_average     # 附加方法用于获取平均值
    return wrapper


@average_execution_time
def example_function(n):
    time.sleep(n)  # 模拟耗时操作

# 多次调用函数
example_function(0.1)
example_function(0.2)
example_function(0.3)

# 获取平均执行时间
avg_time = example_function.get_average()
print(f"Average execution time: {avg_time:.4f} seconds")