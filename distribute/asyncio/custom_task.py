import asyncio

class CustomTask(asyncio.Future):
    def __init__(self, coro, loop = None):
        super().__init__(loop=loop)
        if not asyncio.iscoroutine(coro):
            raise TypeError('A corotinue is required!')
        self._coro = coro
        self.loop = loop or asyncio.get_event_loop()
        self.loop.call_soon(self._step)

    def _step(self, exc=None, data=None):
        # coro 对象内部使用了 yield，所以他可以通过send，恢复协程执行，并把 value 作为当前 yield 表达式的返回值
        # throw 恢复协程执行，并在当前 yield 处抛出一个异常
        try:
            if exc is None:     # 上一个对象运行成功没有 exception
                result = self._coro.send(data)
            else:
                result = self._coro.throw(exc)
        except StopIteration as exc:
            self.set_result(exc.value)
        except Exception as exc:
            self.set_exception(exc)
        else:
            asyncio.ensure_future(result, loop=self.loop).add_done_callback(self._wakeup)

    def _wakeup(self, fut):
        try:
            data = fut.result()
        except Exception as exc:
            self._step(exc=exc)
        else:
            self._step(data=data)

def custom_run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = CustomTask(coro, loop)
    result = loop.run_until_complete(task)
    return result

async def f():
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    loop.call_later(1, future.set_result, "Hello, world!")
    data = await future
    print(data)
    future = loop.create_future()
    loop.call_later(1, future.set_result, "Goodbye, world!")
    data = await future
    print(data)
    return "Done"

result = custom_run(f())
print(result)

result = asyncio.run(f())
print(result)
