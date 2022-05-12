import json
import random
from numpy import random as npr

random.seed(12345)
npr.seed(12345)

towns = ["0%d"%(i+1) for i in range(7)]+["10HD"]
tods = ["noon", "sunset", "night"]
ptods = [.5, .25, .25]
mc = ["clear", "cloudy",   "wetcloudy", "wet"]
pmc = [.25, .25, .25, .25]
rain = ["softrain", "midrain",   "hardrain"]
prain = [.34, .33, .33]
fog = ["midfog",    "hardfog"]
pfog = [.5, .5]
rand = [
            "clear",    "midfog",    "hardfog",
            "cloudy",   "wetcloudy", "wet",
            "softrain", "midrain",   "hardrain"
       ]
prand = [.35, .035, .035, .2, .1, .1, .06, .06, .06]
header = ["Town", "Time-of-Day", "Weather", "PositionID"]

train_rand = open("train_rand.csv", 'w')
train_rand.write(','.join(header)+'\n')
val_rand = open("val_rand.csv", 'w')
val_rand.write(','.join(header)+'\n')
test_rand = open("test_rand.csv", 'w')
test_rand.write(','.join(header)+'\n')

train_mc = open("train_mc.csv", 'w')
train_mc.write(','.join(header)+'\n')
val_mc = open("val_mc.csv", 'w')
val_mc.write(','.join(header)+'\n')
test_mc = open("test_mc.csv", 'w')
test_mc.write(','.join(header)+'\n')

train_rain = open("train_rain.csv", 'w')
train_rain.write(','.join(header)+'\n')
val_rain = open("val_rain.csv", 'w')
val_rain.write(','.join(header)+'\n')
test_rain = open("test_rain.csv", 'w')
test_rain.write(','.join(header)+'\n')

train_fog = open("train_fog.csv", 'w')
train_fog.write(','.join(header)+'\n')
val_fog = open("val_fog.csv", 'w')
val_fog.write(','.join(header)+'\n')
test_fog = open("test_fog.csv", 'w')
test_fog.write(','.join(header)+'\n')

for t in towns:
    with open("Town%s_Opt_wp_4.0m_rndseed42.json"%t) as f:
        data = json.load(f)
        ways = [k for k in data]
        random.shuffle(ways)

    testval = int(len(ways)*.1)
    train_slice = ways[:-2*testval]
    val_slice = ways[-2*testval:-testval]
    test_slice = ways[-testval:]
    
    for w in train_slice:
        tod = npr.choice(tods, 1, p=ptods)[0]
        randwea = npr.choice(rand, 1, p=prand)[0]
        train_rand.write(','.join([t, tod, randwea, w])+'\n')
        mcwea = npr.choice(mc, 1, p=pmc)[0]
        train_mc.write(','.join([t, tod, mcwea, w])+'\n')
        rainwea = npr.choice(rain, 1, p=prain)[0]
        train_rain.write(','.join([t, tod, rainwea, w])+'\n')
        fogwea = npr.choice(fog, 1, p=pfog)[0]
        train_fog.write(','.join([t, tod, fogwea, w])+'\n')
        
    for w in val_slice:
        tod = npr.choice(tods, 1, p=ptods)[0]
        randwea = npr.choice(rand, 1, p=prand)[0]
        val_rand.write(','.join([t, tod, randwea, w])+'\n')
        mcwea = npr.choice(mc, 1, p=pmc)[0]
        val_mc.write(','.join([t, tod, mcwea, w])+'\n')
        rainwea = npr.choice(rain, 1, p=prain)[0]
        val_rain.write(','.join([t, tod, rainwea, w])+'\n')
        fogwea = npr.choice(fog, 1, p=pfog)[0]
        val_fog.write(','.join([t, tod, fogwea, w])+'\n')

    for w in test_slice:
        tod = npr.choice(tods, 1, p=ptods)[0]
        randwea = npr.choice(rand, 1, p=prand)[0]
        test_rand.write(','.join([t, tod, randwea, w])+'\n')
        mcwea = npr.choice(mc, 1, p=pmc)[0]
        test_mc.write(','.join([t, tod, mcwea, w])+'\n')
        rainwea = npr.choice(rain, 1, p=prain)[0]
        test_rain.write(','.join([t, tod, rainwea, w])+'\n')
        fogwea = npr.choice(fog, 1, p=pfog)[0]
        test_fog.write(','.join([t, tod, fogwea, w])+'\n')



train_rand.close()
val_rand.close()
test_rand.close()
train_mc.close()
val_mc.close()
test_mc.close()
train_rain.close()
val_rain.close()
test_rain.close()
train_fog.close()
val_fog.close()
test_fog.close()

"""
    with open("train.csv", 'a') as f:
        for w in train_slice:
            tod = npr.choice(tods, 1, p=)[0]
            wea = npr.choice(weathers, 1, p=)[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            # f.write("Town%s_Opt "%t+w+'\n')
    with open("val.csv", 'a') as f:
        for w in val_slice:
            tod = npr.choice(tods, 1, p=[.5, .25, .25])[0]
            wea = npr.choice(weathers, 1, p=[.35, .035, .035, .2, .1, .1, .06, .06, .06])[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            # f.write("Town%s_Opt "%t+w+'\n')
    with open("test.csv", 'a') as f:
        for w in test_slice:
            tod = npr.choice(tods, 1, p=[.5, .25, .25])[0]
            wea = npr.choice(weathers, 1, p=[.35, .035, .035, .2, .1, .1, .06, .06, .06])[0]
            f.write(','.join([t, tod, wea, w])+'\n')
            # f.write("Town%s_Opt "%t+w+'\n')
"""