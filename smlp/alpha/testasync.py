import asyncio
from functools import partial


async def talk(name):
    print(f"talk function run.")
    await asyncio.sleep(2)
    print(f"talk function run.-----end")
    return f"{name} talk!"


async def sleep(name):
    print(f"sleep function run.")
    await asyncio.sleep(2)
    print(f"sleep function run.----end")
    return f"{name} sleep!"


def callback(name):
    print(f"主动抛出接收：{name}")


if __name__ == "__main__":
    # 通过ensure_future获取，本质是future对象中的result方法
    # loop = asyncio.get_event_loop()
    # get_future1 = asyncio.ensure_future(talk("Dog"))
    # get_future2 = asyncio.ensure_future(sleep("Cat"))
    # loop.run_until_complete(get_future1)
    # loop.run_until_complete(get_future2)
    # print(get_future1.result())
    # print(get_future2.result())

    # 使用loop自带的create_task， 获取返回值
    # loop = asyncio.get_event_loop()
    # task1 = loop.create_task(talk("Dog"))
    # task2 = loop.create_task(sleep("Cat"))
    # loop.run_until_complete(task1)
    # loop.run_until_complete(task2)
    # print(task1.result())
    # print(task2.result())

    # 使用callback, 一旦await地方的内容运行完，就会运行callback
    # loop = asyncio.get_event_loop()
    # task1 = loop.create_task(talk("Dog"))
    # task2 = loop.create_task(sleep("Cat"))
    # task1.add_done_callback(callback)
    # task2.add_done_callback(callback)
    # loop.run_until_complete(task1)
    # loop.run_until_complete(task2)
    # print(task1.result())
    # print(task2.result())

    # 使用partial这个模块向callback函数中传入值
    loop = asyncio.get_event_loop()
    task1 = loop.create_task(talk("Dog"))
    task2 = loop.create_task(talk("Cat"))
    task1.add_done_callback(partial(callback))
    task2.add_done_callback(partial(callback))
    loop.run_until_complete(task1)
    loop.run_until_complete(task2)
    print(task1.result())
    print(task2.result())
