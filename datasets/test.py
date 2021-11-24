import sys, os
sys.path.append(os.path.abspath('.'))

from datasets.carlaLTTM import LTTMDataset

t = LTTMDataset(root_path="lists", town=None, weather=None, time_of_day=None)

for e in t:
    pass
