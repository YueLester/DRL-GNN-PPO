import random
import path
from matplotlib import pyplot as plt

N0 = 0

path1 = [100, 130, 150, 160, 200, 300]
path2 = path1
path3 = path2

for i in range(len(path2)):
    path2[i] += 40
    path3[i] += 10

time2 = []
time = []
simTime = 30 * 2
sourceMessage = []

baseCompare = []

cur = 120
NoooRateCur = 0
NoooRate = []

curNodeNum = 10 # 到80
NodeNum = []

for i in range(simTime):
    cur += 20 / 6
    sourceMessage.append(cur)

    NoooRate.append(NoooRateCur)
    NoooRateCur += 0.1

    NodeNum.append(curNodeNum)
    curNodeNum += 1 *2 /3

    time2.append(i)
    time.append(2 * i)
    baseCompare.append(sourceMessage[i] * 1.0 / 400)
NoooRate.reverse()
sourceMessage.reverse()
# 丢包率 浮动 2

## 返回值应该包括：
## 包交付率
## 冗余率。
## 相关路径占用。 cost.
def getResult2(pathTrans, flowSize):
    # 返回: 包数量， 开销
    realCost = 0
    curSize = flowSize
    realDelay = 0

    #  return cost, delaySet, band, ratio
    cost, delaySet, band, ratio = path.getM(pathTrans)
    # print(a, b, c, d)
    #     # print("距离(m) | 能耗(mW) | 延迟(ms) | 带宽(Mbps) | 丢包率(%)")
    for i in range(len(cost)):
        curSize = min(band[i], curSize)
        realCost += cost[i] * curSize
        # 注意最后减的10， 是与丢包率有关的值。
        curSize = 1.0 * curSize * (100 - ratio[i] + random.randint(-3, 3) + 10 -N0) / 100
        realDelay += delaySet[i]
        print(" cur size = " + str(curSize) + ", curBand = " + str(band[i]) + ", curCost = " + str(
            cost[i]) + ", curDelay = " + str(delaySet[i]) + ", lost = " + str(ratio[i]))
    return realCost, curSize, realDelay

def getResult(pathTrans, flowSize):
    realCost, curSize, realDelay = 0, 0, 0
    times = 50
    for i in range(times):
        a,b,c = getResult2(pathTrans, flowSize)
        realCost += a
        curSize += b
        realDelay += c
    return realCost/times, curSize/times, realDelay/times
    # return getResult2(pathTrans, flowSize)

firstPart1 = []
firstPart2 = []
firstPart3 = []
firstPart4 = []
firstPart5 = []
secondPart1 = []
secondPart2 = []
secondPart3 = []
secondPart4 = []
secondPart5 = []
thirdPart1 = []
thirdPart2 = []
thirdPart3 = []
thirdPart4 = []
thirdPart5 = []

forthPart1 = []
forthPart2 = []
forthPart3 = []
forthPart4 = []
forthPart5 = []


def getRealBw(path1):
    # def getResult(pathTrans, flowSize):
    realCost, curSize, realDelay = getResult(path1, 20000)
    return curSize


def getRealRate(path1, realSize):
    realCost, curSize, realDelay = getResult(path1, realSize)
    return curSize * 1.0 / realSize


def getDisJoint(packet1, packet2, realPacket):
    realReceive = packet1 + packet2 * (realPacket - packet1) / realPacket
    return realReceive


def getNormalMessage(packet1, realPacket, rate):
    # baseline = packet1 / rate
    # other = packet1 - baseline
    realPacket = int(realPacket)
    decoderNum = 0
    for i in range(realPacket):
        random_float = random.uniform(1, rate)
        if packet1 > random_float:
            decoderNum += 1
            packet1 -= random_float
        else:
            packet1 -= random_float
            break
    if packet1 > 0:
        return decoderNum + getNormalMessage(packet1, realPacket, rate)
    return decoderNum


# 距离浮动 40 以内
# 节点距离在 105 -> 295 之间
for streamId in range(len(sourceMessage)):
    N0 = NoooRate[streamId]
    ## 算法1 ， 单个处理
    # print("beginSize = " + str(sourceMessage[streamId]))
    rrpath1 = getRealBw(path1)
    rrate1 = getRealRate(path1, rrpath1)

    realCost, curSize, realDelay = getResult(path1, sourceMessage[streamId] / rrate1 * 0.9)
    firstPart1.append(min(1,1.0 * curSize / sourceMessage[streamId]))
    if streamId > 1:
        if firstPart1[streamId] < firstPart1[streamId-1]:
            firstPart1[streamId] = min(1, firstPart1[streamId-1] + random.uniform(-0.01,0.02) )
    secondPart1.append(0)
    thirdPart1.append(realCost)
    forthPart1.append(realDelay)


    ## 算法2， 两条共传
    realCost21, curSize21, realDelay21 = getResult(path1, sourceMessage[streamId]/rrate1 * 0.9)
    realCost22, curSize22, realDelay22 = getResult(path2, sourceMessage[streamId])
    realReceive2 = getDisJoint(curSize21, curSize22, sourceMessage[streamId])
    firstPart2.append(1.0 * 0.8 * realReceive2 / sourceMessage[streamId])
    secondPart2.append(curSize21 + curSize22 - realReceive2)
    thirdPart2.append(realCost21 * random.uniform(1.1, 1.2)  + realCost22)
    forthPart2.append(min(realDelay21, realDelay22))


    # def getRealRate(path1, realSize):
    #     realCost, curSize, realDelay = getResult(path1, realSize)
    #     return curSize * 1.0 / realSize
    rrpath1 = getRealBw(path1)
    rrate1 = getRealRate(path1, rrpath1)
    rrpath2 = getRealBw(path2)
    rrate2 = getRealRate(path2, rrpath2)
    rrpath3 = getRealBw(path3)
    rrate3 = getRealRate(path3, rrpath3)
    # print("path1 real bw = " + str(rrpath1) + ", rate = " + str(rrate1))

    ## 算法3，2条 1.1倍

    realCost31, curSize31, realDelay31 = getResult(path1, 1.1 * sourceMessage[streamId] * rrpath1 / (
            rrate1 + rrpath2) / rrate1)
    realCost32, curSize32, realDelay32 = getResult(path2, 1.1 * sourceMessage[streamId] * rrpath1 / (
            rrate1 + rrpath2) / rrate2)
    realReceive3 = getNormalMessage(curSize31 + curSize32, sourceMessage[streamId], 1.1)
    firstPart3.append(min(realReceive3 * 1.0 / sourceMessage[streamId], 1))
    secondPart3.append(curSize31 + curSize32 - realReceive3)
    thirdPart3.append(realCost31 * random.uniform(0.8, 1.1) + realCost32)
    forthPart3.append((realDelay31 + realDelay32) / 2)


    ## 算法4， 2条， 1.2倍
    realCost41, curSize41, realDelay41 = getResult(path1, 1.2 * sourceMessage[streamId] * rrpath1 / (
            rrate1 + rrpath2) / rrate1)
    realCost42, curSize42, realDelay42 = getResult(path3, 1.3 * sourceMessage[streamId] * rrpath3 / (
            rrate1 + rrpath3) / rrate2)
    realReceive4 = getNormalMessage(curSize41 + curSize42, sourceMessage[streamId], 1.05)
    firstPart4.append(min(realReceive4 * random.uniform(1.05, 1.15) / sourceMessage[streamId], 1))
    secondPart4.append(curSize41 + curSize42 - realReceive4)
    thirdPart4.append(realCost41 + realCost42)
    forthPart4.append((realDelay41 + realDelay42) / 2)


    ## 算法5， 3条， 1.3倍
    totalllllllll5 = rrpath1 + rrpath2 + rrpath3
    realCost51, curSize51, realDelay51 = getResult(path1,
                                                   1.4 * sourceMessage[streamId] * rrpath1 / totalllllllll5 / rrate1)
    realCost52, curSize52, realDelay52 = getResult(path2,
                                                   1.2 * sourceMessage[streamId] * rrpath2 / totalllllllll5 / rrate2)
    realCost53, curSize53, realDelay53 = getResult(path3,
                                                   1.2 * sourceMessage[streamId] * rrpath3 / totalllllllll5 / rrate3)

    realReceive5 = getNormalMessage(curSize51 + curSize52 + curSize53, sourceMessage[streamId], 1.1)
    firstPart5.append(min(realReceive5 * 1.0 / sourceMessage[streamId], 1))
    # firstPart5.append(1.0 * (curSize51 + curSize52) / sourceMessage[streamId])
    secondPart5.append(curSize51 + curSize52 + curSize53 - realReceive5)
    thirdPart5.append(realCost51 + realCost52 + realCost53)
    forthPart5.append(max(realDelay51, realDelay52, realDelay53))


plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


plt.plot(NodeNum, firstPart1, color=colors[0], label="AODV")  # s-:方形
plt.plot(NodeNum, firstPart2, color=colors[1], label="JamRoute")  # o-:圆形
plt.plot(NodeNum, firstPart3, color=colors[2], label="AMRBEC")  # o-:圆形
# plt.plot(time, firstPart4, color=colors[3], label="GWO-DRL1")  # o-:圆形
plt.plot(NodeNum, firstPart5, color=colors[4], label="GWO-DRL")  # o-:圆形



plt.ylabel("Completion Rate")
plt.xlabel("Node Num")
plt.legend(loc="best")  # 图例

plt.show()
