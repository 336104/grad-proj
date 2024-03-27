import time
from tqdm import tqdm,trange


def ge():
    for i in trange(100):
        yield i


for i in ge():
    time.sleep(1)
