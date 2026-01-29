import time

class Singleton:
    instances = None
    def __init__(self,name):
        self.name=name

    def __new__(cls, *args, **kwargs):
        # 返回空对象
        if cls.instances:
            return cls.instances
        time.sleep(0.1)
        # 创建空对象
        cls.instances=object.__new__(cls)
        return cls.instances

obj1=Singleton('obj1')
print(obj1.name)
# 第二次创建这个对象的时候，他们是同一个对象，但是 init 修改了 name，违背了单例的语义其实
obj2=Singleton('obj2')
print(obj1)
print(obj2)
print(obj2.name)