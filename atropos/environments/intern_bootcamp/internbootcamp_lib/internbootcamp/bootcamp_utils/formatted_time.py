import time

def formatted_time():
    formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return formatted_time