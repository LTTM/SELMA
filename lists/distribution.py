import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

towns = ["01", "02", "03", "04", "05", "06", "07", "10HD"]
tods = ["noon", "sunset", "night"]
weathers = [
            "clear",    "midfog",    "hardfog",
            "cloudy",   "wetcloudy", "wet",
            "softrain", "midrain",   "hardrain"
          ]

counts = np.zeros((3, len(towns), len(tods), len(weathers)), dtype=int)
for i, split in enumerate([train, val, test]):
    for j, town in enumerate(towns):
        t_town = split[split["Town"]==town]
        for k, tod in enumerate(tods):
            t_tod = t_town[t_town["Time-of-Day"]==tod]
            for l, wea in enumerate(weathers):
                t_wea = t_tod[t_tod["Weather"]==wea]
                counts[i,j,k,l] = len(t_wea)

split_tod = counts.sum(axis=(1,3))/np.expand_dims(counts.sum(axis=(1,2,3)), 1)
split_wea = counts.sum(axis=(1,2))/np.expand_dims(counts.sum(axis=(1,2,3)), 1)

town_tod = counts.sum(axis=(0,3))/np.expand_dims(counts.sum(axis=(0,2,3)), 1)
town_wea = counts.sum(axis=(0,2))/np.expand_dims(counts.sum(axis=(0,2,3)), 1)

wea_tod = counts.sum(axis=(0,1)).T/np.expand_dims(counts.sum(axis=(0,1,2)), 1)
tod_wea = counts.sum(axis=(0,1))/np.expand_dims(counts.sum(axis=(0,1,3)), 1)

print(split_tod.shape, split_wea.shape)
print(town_tod.shape, town_wea.shape)
print(wea_tod.shape, tod_wea.shape)

split_names = ["train", "val", "test"]

fig, axs = plt.subplots(3, 2)
for i in range(3):
    axs[0,0].bar(np.arange(3), split_tod[i], label=split_names[i], alpha=.3)
axs[0,0].set_xticklabels(tods)
axs[0,0].set_xticks(np.arange(3))
#axs[0,0].legend()

for i in range(3):
    axs[0,1].bar(np.arange(9), split_wea[i], label=split_names[i], alpha=.3)
axs[0,1].set_xticklabels(weathers, rotation=30, ha='right')
axs[0,1].set_xticks(np.arange(9))
#axs[0,1].legend()

for i in range(8):
    axs[1,0].bar(np.arange(3), town_tod[i], label=towns[i], alpha=.3)
axs[1,0].set_xticklabels(tods)
axs[1,0].set_xticks(np.arange(3))
#axs[1,0].legend()

for i in range(8):
    axs[1,1].bar(np.arange(9), town_wea[i], label=towns[i], alpha=.3)
axs[1,1].set_xticklabels(weathers, rotation=30, ha='right')
axs[1,1].set_xticks(np.arange(9))
#axs[1,1].legend()

for i in range(3):
    axs[2,1].bar(np.arange(9), tod_wea[i], label=tods[i], alpha=.3)
axs[2,1].set_xticklabels(weathers, rotation=30, ha='right')
axs[2,1].set_xticks(np.arange(9))
#axs[0,1].legend()

for i in range(9):
    axs[2,0].bar(np.arange(3), wea_tod[i], label=weathers[i], alpha=.3)
axs[2,0].set_xticklabels(tods)
axs[2,0].set_xticks(np.arange(3))

fig.tight_layout()
plt.show()
