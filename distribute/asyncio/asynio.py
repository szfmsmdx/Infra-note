# https://zhuanlan.zhihu.com/p/698683843
import asyncio
import time

async def wget(host):
    print(f"wget {host}...")
    # 连接80端口:
    reader, writer = await asyncio.open_connection(host, 80)
    # 发送HTTP请求:
    header = f"GET / HTTP/1.0\r\nHost: {host}\r\n\r\n"
    writer.write(header.encode("utf-8"))
    await writer.drain()

    # 读取HTTP响应:
    while True:
        line = await reader.readline()
        if line == b"\r\n":
            break
        print("%s header > %s" % (host, line.decode("utf-8").rstrip()))
    # Ignore the body, close the socket
    writer.close()
    await writer.wait_closed()
    print(f"Done {host}.")

async def main():
    await asyncio.gather(wget("www.sina.com.cn"), wget("www.sohu.com"), wget("www.163.com"))

asyncio.run(main())

async def async_hello_world():
    now = time.time()
    await asyncio.sleep(1)
    print(time.time() - now)
    print("Hello, world!")
    await asyncio.sleep(1)
    print(time.time() - now)

async def main():
    task1 = asyncio.create_task(async_hello_world())    # 将coro转为task并立即执行
    task2 = asyncio.create_task(async_hello_world())
    task3 = asyncio.create_task(async_hello_world())
    await task1
    await task2
    await task3

now = time.time()
# run 3 async_hello_world() coroutine concurrently
# 创建一个新的event loop
# 调用 main() 得到一个coro
# 把这个coro交给loop运行（内部也会包装成 Task）
# 阻塞直到 main() 完成
asyncio.run(main())

print(f"Total time for running 3 coroutine: {time.time() - now}")