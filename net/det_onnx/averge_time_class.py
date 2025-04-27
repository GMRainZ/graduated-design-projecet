import time
from functools import wraps
import types

def average_execution_time(func):
    @wraps(func)
    def wrapper(self, *args,**kwargs):
        # 初始化实例的状态（如果不存在）
        if not hasattr(self, '_total_time'):
            self._total_time = 0.0
            self._num_calls = 0

        # 执行原函数并计时
        start_time = time.perf_counter()
        result = func(self, *args,**kwargs)
        elapsed = time.perf_counter() - start_time

        # 更新统计状态
        self._total_time += elapsed
        self._num_calls += 1
        return result

    # 定义获取平均时间的函数
    def get_average(self):
        return self._total_time / self._num_calls if self._num_calls > 0 else 0.0

    # 将 get_average 转换为描述符以支持实例绑定
    class GetAverageDescriptor:
        def __get__(self, instance, owner):
            if instance is None:
                return get_average
            return types.MethodType(get_average, instance)

    # 将描述符附加到 wrapper 的 get_average 属性
    wrapper.get_average = GetAverageDescriptor()
    return wrapper


class MyClass:
    @average_execution_time
    def my_method(self, delay):
        time.sleep(delay)  # 模拟耗时操作

# 创建实例并调用方法
obj = MyClass()
obj.my_method(0.1)  # 第1次调用
obj.my_method(0.2)  # 第2次调用

# 获取平均执行时间
avg_time = obj.my_method.get_average()
print(f"Average execution time: {avg_time:.4f} seconds")