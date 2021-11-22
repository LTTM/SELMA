import json
import random
from numpy import random as npr

random.seed(12345)
npr.seed(12345)

towns = ["0%d"%(i+1) for i in range(7)]+["10HD"]
tods = ["noon", "sunset", "night"]
weathers = [
            "clear",    "midfog",    "hardfog",
            "cloudy",   "wetcloudy", "wet",
            "softrain", "midrain",   "hardrain"
          ]
header = ["Town", "Time-of-Day", "Weather", "PositionID"]

for t in towns:
    with open("Town%s_Opt_wp_4.0m_rndseed42.json"%t) as f:
        data = json.load(f)
        ways = [k for k in data]
        random.shuffle(ways)

    testval = int(len(ways)*.1)
    train_slice = ways[:-2*testval]
    val_slice = ways[-2*testval:-testval]
    test_slice = ways[-testval:]
    # tot = len(ways)
    # train = int(tot*.8)
    # test = int(tot*.1)
    # val = tot-train-test
    #
    # train_slice = ways[0:train]
    # val_slice = ways[train:train+val]
    # test_slice = ways[train+val:]

    with open("train.csv", 'a') as f:
        f.write(','.join(header)+'\n')
        for w in train_slice:
            tod = npr.choice(tods, 1, p=[.5, .25, .25])[0]
            wea = npr.choice(weathers, 1, p=[.35, .035, .035, .2, .1, .1, .06, .06, .06])[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            #f.write("Town%s_Opt "%t+w+'\n')
    with open("val.csv", 'a') as f:
        f.write(','.join(header)+'\n')
        for w in val_slice:
            tod = npr.choice(tods, 1, p=[.5, .25, .25])[0]
            wea = npr.choice(weathers, 1, p=[.35, .035, .035, .2, .1, .1, .06, .06, .06])[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            #f.write("Town%s_Opt "%t+w+'\n')
    with open("test.csv", 'a') as f:
        f.write(','.join(header)+'\n')
        for w in test_slice:
            tod = npr.choice(tods, 1, p=[.5, .25, .25])[0]
            wea = npr.choice(weathers, 1, p=[.35, .035, .035, .2, .1, .1, .06, .06, .06])[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            #f.write("Town%s_Opt "%t+w+'\n')
