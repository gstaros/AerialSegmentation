import multiprocessing as mp
from torch.utils.data import DataLoader
from datetime import datetime

def find_num_workers(dataset, batch_size, num_workers_range = mp.cpu_count()):
    lowest_time = 999999999999
    best_num = -1
    for num in range(num_workers_range):
        temp_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num, pin_memory=True)

        # measure time
        start = datetime.now()
        for _ in range(2):
            for i, data in enumerate(temp_dataloader, 0):
                pass
        end = datetime.now()

        time = (end - start).total_seconds()
        if time < lowest_time:
            best_num = num
            lowest_time = time

        print(f"Num workers: {num}, Time: {time}")
    return best_num
