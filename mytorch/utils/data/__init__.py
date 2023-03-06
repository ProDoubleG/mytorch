import numpy

class DataLoader:
    def __init__(self,dataset, batch_size=2):
        self.dataset = dataset
        self.batch_size = batch_size

    def __new__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # (iter_num, batch_size, x, y , channel) , (iter_num, batch_size, label)
        return  
        