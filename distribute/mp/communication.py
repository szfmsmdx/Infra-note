from multiprocessing import Process,Queue,Pipe

# # Queue
# def fun1(q,i):
#     print('子进程%s 开始put数据' %i)
#     q.put('我是%s 通过Queue通信' %i)

# if __name__ == '__main__':
#     q = Queue()

#     process_list = []
#     for i in range(3):
#         p = Process(target=fun1,args=(q,i,))  #注意args里面要把q对象传给我们要执行的方法，这样子进程才能和主进程用Queue来通信
#         p.start()
#         process_list.append(p)

#     for i in process_list:
#         p.join()

#     print('主进程获取Queue数据')
#     print(q.get())
#     print(q.get())
#     print(q.get())
#     print('结束测试')

def fun1(conn):
    print('子进程发送消息：')
    conn.send('你好主进程')
    print('子进程接受消息：')
    print(conn.recv())
    conn.close()

if __name__ == '__main__':
    conn1, conn2 = Pipe() #关键点，pipe实例化生成一个双向管
    p = Process(target=fun1, args=(conn2,)) #conn2传给子进程
    p.start()
    print('主进程接受消息：')
    print(conn1.recv())
    print('主进程发送消息：')
    conn1.send("你好子进程")
    p.join()
    print('结束测试')