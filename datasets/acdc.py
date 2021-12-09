import numpy as np
from PIL import Image
from datasets.cityscapes import CityDataset

class ACDCDataset(CityDataset):
    def __init__(self,
                 weather=None,
                 time_of_day=None,
                 **kwargs): # whether to use city19 or city36 class set

        super(ACDCDataset, self).__init__(**kwargs)

        if time_of_day is not None:
            self.items = [e for e in self.items if time_of_day in e[0]]

        if weather is not None:
            self.items = [e for e in self.items if weather in e[0]]

    