import asyncio

# loop api
# 直接操作 event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
print(loop.time())

loop.call_soon(lambda: print("Hello, world! at call_soon"))
loop.call_later(1, lambda: print("Hello, world! at call_later"))
loop.call_at(loop.time() + 1, lambda: print("Hello, world! at call_at"))
loop.call_later(2, loop.stop)

loop.run_forever()  # run_forever是死循环，等上面的 call_later 执行 loop.stop 才停止

# future
# Future 对象是用来表示将来要完成的 **结果** ，维护状态，如果编程finished，就不再等待
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

future = loop.create_future()
# set_result 会把 future标记成done，并记录参数作为结果
handle = loop.call_later(1, lambda: future.set_result("Hello, world!")) 
result = loop.run_until_complete(future)    # 其实等价于 await future，返回值就是 set_result 设置的返回值
print(result)
loop.close()

