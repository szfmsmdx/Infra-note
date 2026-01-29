from multiprocessing import Process

def fun1(name):
    print('测试%s多进程' %name)

process_list = []
for i in range(5):  #开启5个子进程执行fun1函数
    p = Process(target=fun1,args=(i,)) # 这里传参需要是可迭代的
    p.start()
    process_list.append(p)

for i in process_list:
    p.join()

print('结束测试')

# 通过继承来实现多进程
class MyProcess(Process):
    def __init__(self, name):
        super(MyProcess, self).__init__()
        self.name = str(name)    # 父类 Process 会调用他的 name setter，强制要求 str

    def run(self):
        print(f"测试{self.name}多线程")

process_list = []

for i in range(5):
    p = MyProcess(i)
    p.start()
    process_list.append(p)

for p in process_list:
    p.join()

print("结束测试")