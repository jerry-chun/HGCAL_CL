from .CCV1 import CCV1
from torch_geometric.data import DataLoader

def Data_Loader(root, split, max_events = 100, batch_size = 1, shuffle = True, follow_batch = ['x']):
    dataset = CCV1(root = root, split = split, max_events = max_events)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, follow_batch = follow_batch)
    return loader
