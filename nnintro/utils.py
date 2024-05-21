
from tqdm import tqdm



def get_progress(data, name): # vykresluje loading bar v terminali
    result = tqdm(
        data,
        desc=name
    )
    return result



class Accumulator:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def step(self, x):
        self.sum += x
        self.count += 1
        self.avg = self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

class Statistics:
    def __init__(self):
        self.items = {}

    def step(self, key, value):
        if key not in self.items:
            self.items[key] = Accumulator()
        self.items[key].step(value)

    def reset(self):
        for k in self.items:
            self.items[k].reset()

    def get_stats(self):
        result = {}
        for kluc in self.items.keys():
            result[kluc] = self.items[kluc].avg
        return result