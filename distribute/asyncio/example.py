def consumer():
    r = ''
    while True:
        n = yield r 
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)   # n 来接受 r
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

# 函数体中包含 yield 就已经是一个 generate 对象了，创建的时候不会执行任何一行代码
c = consumer()
produce(c)

def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print(n)
