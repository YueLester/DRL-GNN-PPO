import pickle

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

AUV_NUM = 3
SENSOR_NUM = 10
color_auv = ['#4169E1','#FFA500','#32CD32','#DDEE22']
linestyle = ['-','-.',':','--']
marker_p = ['s','o', '+', 'x']
marker_auv = ['o', '^', 's']
def show_reward():
    load_path = "data/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # x_label = list(range(len(episode_reward_list)))
    # plt.plot(x_label[100:], episode_reward_list[100:], 'r-', label='H-MADDPG')
    # plt.show()
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    # avg_list = [(0 if i < -30 else i) for i in episode_reward_list]
    x_label = list(range(len(avg_list)))
    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list[0:], 'r-', label='H-MADDPG')
    plt.fill_between(x_label, max_list,
                     min_list, facecolor='r', alpha=0.3)

    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()



def compare_reward():
    load_path = "data/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    load_path = "data_old/data12072124/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    load_path = "data_old/data_4/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    avg_list_4 = []
    min_list_4 = []
    max_list_4 = []
    for i in range(len(episode_reward_list_4)):
        temp += episode_reward_list_4[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_4[i - avg:i])
            min_list_4.append((sorted_list[1] + sorted_list[0])/2)
            max_list_4.append(sorted_list[-1])
            avg_list_4.append(temp / avg)
            temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    x_label = list(range(length))
    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='AUV number = 2')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='AUV number = 3')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    plt.fill_between(x_label, max_list_4[0:length],
                     min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def compare_voi():
    load_path = "data/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    load_path = "data_old/data_4/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('VOI')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    avg_list_4 = []
    min_list_4 = []
    max_list_4 = []
    for i in range(len(episode_reward_list_4)):
        temp += episode_reward_list_4[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_4[i - avg:i])
            min_list_4.append((sorted_list[1] + sorted_list[0])/2)
            max_list_4.append(sorted_list[-1])
            avg_list_4.append(temp / avg)
            temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    x_label = list(range(length))
    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='AUV number = 2')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='AUV number = 3')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    plt.fill_between(x_label, max_list_4[0:length],
                     min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)


def compare_reward_sensor():
    load_path = "data/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    load_path = "data_old/data12072124/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # load_path = "data_4/episode_reward_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    # avg_list_4 = []
    # min_list_4 = []
    # max_list_4 = []
    # for i in range(len(episode_reward_list_4)):
    #     temp += episode_reward_list_4[i]
    #     if i % avg == 0 and i != 0:
    #         sorted_list = sorted(episode_reward_list_4[i - avg:i])
    #         min_list_4.append((sorted_list[1] + sorted_list[0])/2)
    #         max_list_4.append(sorted_list[-1])
    #         avg_list_4.append(temp / avg)
    #         temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    # length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    length = min([len(avg_list), len(avg_list_2)])
    x_label = list(range(length))

    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='sensor number = 8')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='sensor number = 10')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    # plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    # plt.fill_between(x_label, max_list_4[0:length],
    #                  min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def compare_voi_sensor():
    load_path = "data/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    episode_reward_list_2 = [x/12 for x in episode_reward_list_2]
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    episode_reward_list = [x/10 for x in episode_reward_list]
    # load_path = "data_4/episode_reward_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average VOI')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    # avg_list_4 = []
    # min_list_4 = []
    # max_list_4 = []
    # for i in range(len(episode_reward_list_4)):
    #     temp += episode_reward_list_4[i]
    #     if i % avg == 0 and i != 0:
    #         sorted_list = sorted(episode_reward_list_4[i - avg:i])
    #         min_list_4.append((sorted_list[1] + sorted_list[0])/2)
    #         max_list_4.append(sorted_list[-1])
    #         avg_list_4.append(temp / avg)
    #         temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    # length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    length = min([len(avg_list), len(avg_list_2)])
    x_label = list(range(length))

    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='sensor number = 14')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='sensor number = 10')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    # plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    # plt.fill_between(x_label, max_list_4[0:length],
    #                  min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def compare_reward_methor():
    load_path = "data_old/data_withoutdqn_01031310/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    episode_reward_list_2 = [x for x in episode_reward_list_2]
    load_path = "data_old/data12072124/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    episode_reward_list = [x for x in episode_reward_list]
    load_path = "data_old/data_withoutrelay/episode_reward_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_3 = pickle.load(f)
    episode_reward_list_3 = [x for x in episode_reward_list_3]
    # load_path = "data_4/episode_reward_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    # avg_list_4 = []
    # min_list_4 = []
    # max_list_4 = []
    # for i in range(len(episode_reward_list_4)):
    #     temp += episode_reward_list_4[i]
    #     if i % avg == 0 and i != 0:
    #         sorted_list = sorted(episode_reward_list_4[i - avg:i])
    #         min_list_4.append((sorted_list[1] + sorted_list[0])/2)
    #         max_list_4.append(sorted_list[-1])
    #         avg_list_4.append(temp / avg)
    #         temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    avg_list_3 = []
    min_list_3 = []
    max_list_3 = []
    for i in range(len(episode_reward_list_3)):
        temp += episode_reward_list_3[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_3[i - avg:i])
            min_list_3.append((sorted_list[1] + sorted_list[0])/2)
            max_list_3.append(sorted_list[-1])
            avg_list_3.append(temp / avg)
            temp = 0
    avg_list_3 = avg_list_3[2:]
    min_list_3 = min_list_3[2:]
    max_list_3 = max_list_3[2:]
    # length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    length = min([len(avg_list), len(avg_list_2),len(avg_list_3)])
    x_label = list(range(length))

    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='H-MADDPG')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='Rule-based MADDPG')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list_3[0:length], 'b-', label='Relay-free MADDPG')
    plt.fill_between(x_label, max_list_3[0:length],
                     min_list_3[0:length], facecolor='b', alpha=0.3)
    # plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    # plt.fill_between(x_label, max_list_4[0:length],
    #                  min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def compare_voi_methor():
    load_path = "data_old/data_withoutdqn_01031310/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_2 = pickle.load(f)
    episode_reward_list_2 = [x/10 for x in episode_reward_list_2]
    episode_reward_list_2 = [x/episode_reward_list_2[-1] * 72.73180356 for x in episode_reward_list_2]
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    episode_reward_list = [x/10 for x in episode_reward_list]
    episode_reward_list = [x/episode_reward_list[-1] * 76.27486278 for x in episode_reward_list]
    load_path = "data_old/data_withoutrelay/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list_3 = pickle.load(f)
    episode_reward_list_3 = [x/10 for x in episode_reward_list_3]
    episode_reward_list_3 = [x/episode_reward_list_3[-1] * 66.23252127 for x in episode_reward_list_3]
    # load_path = "data_4/episode_reward_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list_4 = pickle.load(f)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average VOI')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append((sorted_list[1] + sorted_list[0])/2)
            max_list.append(sorted_list[-1])
            avg_list.append(temp / avg)
            temp = 0
    # avg_list_4 = []
    # min_list_4 = []
    # max_list_4 = []
    # for i in range(len(episode_reward_list_4)):
    #     temp += episode_reward_list_4[i]
    #     if i % avg == 0 and i != 0:
    #         sorted_list = sorted(episode_reward_list_4[i - avg:i])
    #         min_list_4.append((sorted_list[1] + sorted_list[0])/2)
    #         max_list_4.append(sorted_list[-1])
    #         avg_list_4.append(temp / avg)
    #         temp = 0
    avg_list_2 = []
    min_list_2 = []
    max_list_2 = []
    for i in range(len(episode_reward_list_2)):
        temp += episode_reward_list_2[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_2[i - avg:i])
            min_list_2.append((sorted_list[1] + sorted_list[0])/2)
            max_list_2.append(sorted_list[-1])
            avg_list_2.append(temp / avg)
            temp = 0

    avg_list_3 = []
    min_list_3 = []
    max_list_3 = []
    for i in range(len(episode_reward_list_3)):
        temp += episode_reward_list_3[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list_3[i - avg:i])
            min_list_3.append((sorted_list[1] + sorted_list[0])/2)
            max_list_3.append(sorted_list[-2])
            avg_list_3.append(temp / avg)
            temp = 0
    avg_list_3 = avg_list_3[2:]
    min_list_3 = min_list_3[2:]
    max_list_3 = max_list_3[2:]
    # length = min([len(avg_list),len(avg_list_4),len(avg_list_2)])
    length = min([len(avg_list), len(avg_list_2),len(avg_list_3)])
    x_label = list(range(length))

    x_label = [i * avg for i in x_label]
    plt.plot(x_label[0:], avg_list[0:length], 'r-', label='H-MADDPG')
    plt.fill_between(x_label, max_list[0:length],
                     min_list[0:length], facecolor='r', alpha=0.3)
    plt.plot(x_label[0:], avg_list_2[0:length], 'g-', label='Rule-based MADDPG')
    plt.fill_between(x_label, max_list_2[0:length],
                     min_list_2[0:length], facecolor='g', alpha=0.3)
    plt.plot(x_label[0:], avg_list_3[0:length], 'b-', label='Relay-free MADDPG')
    plt.fill_between(x_label, max_list_3[0:length],
                     min_list_3[0:length], facecolor='b', alpha=0.3)
    # plt.plot(x_label[0:], avg_list_4[0:length], 'b-', label='AUV number = 4')
    # plt.fill_between(x_label, max_list_4[0:length],
    #                  min_list_4[0:length], facecolor='b', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def show_voi():
    load_path = "data/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # x_label = list(range(len(episode_reward_list)))
    # plt.plot(x_label[100:], episode_reward_list[100:], 'r-', label='H-MADDPG')
    # plt.show()
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average VOI')
    avg_list = []
    temp = 0
    avg = 20
    min_list = []
    max_list = []
    print(max(episode_reward_list)/10)
    for i in range(len(episode_reward_list)):
        temp += episode_reward_list[i]
        if i % avg == 0 and i != 0:
            sorted_list = sorted(episode_reward_list[i - avg:i])
            min_list.append(sorted_list[0] / SENSOR_NUM )
            max_list.append(sorted_list[-1] / SENSOR_NUM )
            avg_list.append(temp / avg / SENSOR_NUM)
            temp = 0
    # avg_list = [(0 if i < -30 else i) for i in episode_reward_list]
    x_label = list(range(len(avg_list)))
    x_label = [i * avg for i in x_label]
    # plt.xticks(x_label)
    plt.plot(x_label[0:], avg_list[0:], 'r-', label='H-MADDPG')
    plt.fill_between(x_label, max_list,
                     min_list, facecolor='r', alpha=0.3)
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def show_auv_data():
    load_path = "data/history_data_list.pkl"
    with open(load_path, 'rb') as f:
        history_data_list = pickle.load(f)
    x_label = list(range(len(history_data_list[0])))
    fig, ax = plt.subplots()


    for i in range(AUV_NUM):
        print(history_data_list[i])
        plt.plot(x_label, history_data_list[i], c = color_auv[i],linestyle = linestyle[i], label='AUV-%d'%(i+1))
    ax.set_xlabel('Slot')
    ax.set_ylabel('Data')
    plt.xticks(x_label)

    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def show_auv_p():
    load_path = "data/history_p_list.pkl"
    with open(load_path, 'rb') as f:
        history_p_list = pickle.load(f)
    x_label = list(range(len(history_p_list[0])))
    fig, ax = plt.subplots()
    for i in range(AUV_NUM):
        print(history_p_list[i])
        plt.scatter(x_label, history_p_list[i], c = color_auv[i],marker=marker_p[i], s= 80 ,  label='AUV-%d'%(i+1))
    ax.set_xlabel('Slot')
    ax.set_ylabel('Task')
    plt.xticks(x_label)
    plt.yticks(np.linspace(0, 2, 3))
    plt.legend(loc="best", frameon=False, fontsize=14)
    plt.show()

def show_sensor_data():
    load_path = "data/history_sensor_data_list.pkl"
    with open(load_path, 'rb') as f:
        history_sensor_data_list = pickle.load(f)
    x_label = list(range(len(history_sensor_data_list[0])))
    fig, ax = plt.subplots()
    for i in range(SENSOR_NUM):
        plt.plot(x_label, history_sensor_data_list[i],linestyle='-.', label='sensor-%d' % (i + 1))
    ax.set_xlabel('Slot')
    ax.set_ylabel('Data')
    plt.xticks(x_label)
    plt.legend(loc="best", frameon=False, fontsize=10)
    plt.show()

def draw_3D_history_with_p(AUV_history_location, sensor_location, p_history_list):
    fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
    ax = Axes3D(fig)
    ax.set(xlabel = 'X coordinate(m)',ylabel='Y coordinate(m)',zlabel = 'Z coordinate(m)')
    for location in sensor_location:
        ax.scatter(location[0], location[1],location[2], c="black", marker='*')  # 画出骨干的散点图。

    for i in range(len(AUV_history_location)):
        history_location = AUV_history_location[i]
        x = [row[0] for row in history_location]
        y = [row[1] for row in history_location]
        z = [row[2] for row in history_location]
        p_list = p_history_list[i]

        ax.plot(x,y,z,c=color_auv[i])
        for j in range(len(p_list)):
            if j > 0:
                ax.plot(x[j],y[j],z[j],marker = marker_auv[p_list[j]],c = color_auv[i])
            else:
                ax.plot(x[j], y[j], z[j], marker='x', c=color_auv[i])

        # mscatter(x,y,z, ax = ax, m = marker_list.copy() ,c=color_auv[i])

    custom_lines = [ax.scatter(sensor_location[0][0], sensor_location[0][1],sensor_location[0][2], color='black', marker='*', lw=1),
                    plt.Line2D([0], [0], color=color_auv[0], lw=1),
                    plt.Line2D([0], [0], color=color_auv[1], lw=1),
                    plt.Line2D([0], [0], color=color_auv[2], lw=1),
                    plt.Line2D([0], [0], color='black', marker=marker_auv[0], lw=1),
                    plt.Line2D([0], [0], color='black', marker=marker_auv[1], lw=1),
                    plt.Line2D([0], [0], color='black', marker=marker_auv[2], lw=1),
                    plt.Line2D([0], [0], color='black', marker='x', lw=1)
                    ]


    # plt.legend(custom_lines, ['sensor', 'AUV 1', 'AUV 2', 'AUV 3', 'collect', 'relay', 'transmit','start position'])
    plt.legend(custom_lines, ['sensor', 'AUV 1', 'AUV 2', 'AUV 3', 'collect', 'relay', 'transmit', 'start position'], loc = "best",fontsize=8)
    plt.show()
    # ax.scatter(base_location[0], base_location[1],base_location[2], c="black", marker='*')
    # ax.set_aspect(1)

def voi_by_sensor():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_sensor8/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 8)
    list.append(max(episode_reward_list) / 8)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_sensor12_12261429/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 12)
    list.append(max(episode_reward_list) / 12)
    load_path = "data_old/data_sensor14/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 14)
    list.append(max(episode_reward_list) / 14)
    load_path = "data_old/data_sensor16/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 16)
    list.append(max(episode_reward_list) / 16)

    print(list)
    list2 = []
    load_path = "data_old/data_sensor8_auv4_01181608/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list2.append(episode_reward_list[-1] / 8)
    list2.append(max(episode_reward_list) / 8)
    load_path = "data_old/data_4_12270926/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list2.append(episode_reward_list[-1] / 10)
    list2.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_sensor12_auv4_01181406/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list2.append(episode_reward_list[-1] / 12)
    list2.append(max(episode_reward_list) / 12)
    load_path = "data_old/data_sensor14_auv4_01171639/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list2.append(episode_reward_list[-1] / 14)
    list2.append(max(episode_reward_list) / 14)
    load_path = "data_old/data_sensor16_auv4_01180921/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    # list2.append(episode_reward_list[-1] / 16)
    list2.append(max(episode_reward_list) / 16)
    print(list2)
    ax.set_xlabel('Number of Sensors', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [8, 10, 12, 14, 16]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-',label='AUV num = 3')
    plt.plot(x_label, list2, 'bo-',label='AUV num = 4')
    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()



def voi_by_method_and_auv():

    list = [55.27650568,72.64846901,76.27486278,77.33597071,78.11139238]
    list2 = [53.98196631,65.41593983,72.73180356,74.49731953,75.72081924]
    list3 = [55.27650568,63.17264975,66.23252127,66.98100576,67.33059339]
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of AUVs')
    ax.set_ylabel('VoI')
    plt.grid(True)
    x_label = [1,2,3,4,5]
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22', '#AA22CC']
    plt.xticks(x_label)
    plt.plot(x_label, list,color=color[0],marker='o' ,label='H-MADDPG')
    plt.plot(x_label, list2,color=color[1],marker = 's',label='Rule-based MADDPG')
    plt.plot(x_label, list3,color=color[2],marker = '^',label='Relay-free MADDPG')
    plt.legend(loc='lower right', fancybox=True, fontsize=8, ncol=1)
    plt.show()

def voi_by_method_and_speed():

    list = [61.55349157,	70.43486338,	76.27486278,	77.33304668,	78.15352882]
    list2 = [55.8081797,	66.24779013,	72.73180356,	74.69691183,	75.57856822]
    list3 = [52.26787605,	62.30100811,	66.23252127,	68.70664009,	69.7734103]
    fig, ax = plt.subplots()
    ax.set_xlabel('Speed(m/s)')
    ax.set_ylabel('VoI')
    plt.grid(True)
    x_label = [2,3,4,5,6]
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22', '#AA22CC']
    plt.xticks(x_label)
    plt.plot(x_label, list,color=color[0],marker='o' ,label='H-MADDPG')
    plt.plot(x_label, list2,color=color[1],marker = 's',label='Rule-based MADDPG')
    plt.plot(x_label, list3,color=color[2],marker = '^',label='Relay-free MADDPG')
    plt.legend(loc='lower right', fancybox=True, fontsize=8, ncol=1)
    plt.show()



def voi_by_speed_and_auv():
    list = [50.15420263,	52.84618043,	55.27650568,	57.20766159,	58.82823728]
    list2 = [54.7605942,	66.6807959,	72.64846901,	73.88542883,	74.62648776]
    list3 = [61.55349157,	70.43486338,	76.27486278,	77.33304668,	78.15352882]
    list4 = [66.23442046,	73.72737172,	77.33597071,	78.12420902,	79.15924374]
    list5 = [70.51323049,	75.40281096,	78.11139238,	79.14089756,	79.8637092]
    fig, ax = plt.subplots()
    ax.set_xlabel('Speed(m/s)')
    ax.set_ylabel('VoI')
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22','#AA22CC']
    x_label = [2,3,4,5,6]
    plt.grid(True)
    plt.xticks(x_label)
    plt.plot(x_label, list,color=color[0],marker='o' ,label='Number of AUV = 1')
    plt.plot(x_label, list2,color=color[1],marker = 's',label='Number of AUV = 2')
    plt.plot(x_label, list3,color=color[2],marker = '^',label='Number of AUV = 3')
    plt.plot(x_label, list4,color=color[3],marker = '*',label='Number of AUV = 4')
    plt.plot(x_label, list5,color=color[4],marker = 'd',label='Number of AUV = 5')
    plt.legend(loc='lower right', fancybox=True, fontsize=8, ncol=1)
    plt.show()


# def voi_by_range_and_sensor():
#     list = [[59.91785755, 55.27650568, 53.5276814, 52.59523525,	52.21588587],
#             [75.44561157, 72.64846901, 70.50497617, 68.28269204, 65.02692445],
#             [78.06155189, 76.27486278, 72.81934142, 70.22115134, 69.18603157],
#             [78.80667574, 77.33597071, 75.15701781, 73.02696128, 72.2533033],
#             [80.37744512, 78.11139238, 77.23258258, 75.7921371, 75.11154583]]
#
#     fig, ax = plt.subplots()
#     ax.set_xlabel('Number of Sensors')
#     ax.set_ylabel('VoI')
#     color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22', '#AA22CC']
#     x_label = [8, 10, 12, 14, 16]
#     plt.grid(True)
#     plt.xticks(x_label)
#     plt.plot(x_label, list[0], color=color[0], marker='o', label='Number of AUV = 1')
#     plt.plot(x_label, list[1], color=color[1], marker='s', label='Number of AUV = 2')
#     plt.plot(x_label, list[2], color=color[2], marker='^', label='Number of AUV = 3')
#     plt.plot(x_label, list[3], color=color[3], marker='*', label='Number of AUV = 4')
#     plt.plot(x_label, list[4], color=color[4], marker='d', label='Number of AUV = 5')
#     plt.legend(fancybox=True, fontsize=8, ncol=1)
#     plt.show()

def voi_by_auv_rate_and_sensor():
    list = [[59.91785755, 55.27650568, 53.5276814, 52.59523525,	52.21588587],
            [75.44561157, 72.64846901, 70.50497617, 68.28269204, 65.02692445],
            [78.06155189, 76.27486278, 72.81934142, 70.22115134, 69.18603157],
            [78.80667574, 77.33597071, 75.15701781, 73.02696128, 72.2533033],
            [80.37744512, 78.11139238, 77.23258258, 75.7921371, 75.11154583]]

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Sensors')
    ax.set_ylabel('VoI Retention Rate(%)')
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22', '#AA22CC']
    x_label = [8, 10, 12, 14, 16]
    plt.grid(True)
    plt.xticks(x_label)
    plt.plot(x_label, list[0], color=color[0], marker='o', label='Number of AUV = 1')
    plt.plot(x_label, list[1], color=color[1], marker='s', label='Number of AUV = 2')
    plt.plot(x_label, list[2], color=color[2], marker='^', label='Number of AUV = 3')
    plt.plot(x_label, list[3], color=color[3], marker='*', label='Number of AUV = 4')
    plt.plot(x_label, list[4], color=color[4], marker='d', label='Number of AUV = 5')
    plt.legend(fancybox=True, fontsize=8, ncol=1)
    plt.show()

def voi_by_auv_and_sensor():
    list = [[59.91785755,	55.27650568,	53.5276814,	52.59523525,	52.21588587],
            [75.44561157,	72.64846901,	70.50497617,	68.28269204,	65.02692445],
            [78.06155189,	76.27486278,	72.81934142,	70.22115134,	69.18603157],
            [78.80667574,	77.33597071,	75.15701781,	73.02696128,	72.2533033],
            [80.37744512,	78.11139238,	77.23258258,	75.7921371,	75.11154583]]

    for l in list:
        l[0] *= 0.8
        l[1] *= 1
        l[2] *= 1.2
        l[3] *= 1.4
        l[4] *= 1.6



    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Sensors')
    ax.set_ylabel('VoI')
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22','#AA22CC']
    x_label = [8,10,12,14,16]
    plt.grid(True)
    plt.xticks(x_label)
    plt.plot(x_label, list[0],color=color[0],marker='o' ,label='Number of AUV = 1')
    plt.plot(x_label, list[1],color=color[1],marker = 's',label='Number of AUV = 2')
    plt.plot(x_label, list[2],color=color[2],marker = '^',label='Number of AUV = 3')
    plt.plot(x_label, list[3],color=color[3],marker = '*',label='Number of AUV = 4')
    plt.plot(x_label, list[4],color=color[4],marker = 'd',label='Number of AUV = 5')
    plt.legend(fancybox=True, fontsize=8, ncol=1)
    plt.show()

def voi_by_range_and_sensor():
    list = [[80.31257469,	78.06155189,	71.74693269,	64.33729141,	56.77791027],
            [78.41573705,	76.27486278,	70.17682965,	63.25315273,	55.97233117],
            [74.88834933,	72.81934142,	67.60688462,	59.83177463,	54.64473524],
            [72.05569199,	70.22115134,	66.03524442,	58.2131879,	53.41332034],
            [70.45345545,	69.18603157,	63.85527494,	55.60307681,	52.5273376]]

    list[0] = [x * 0.8 for x in list[0]]
    list[1] = [x * 1 for x in list[1]]
    list[2] = [x * 1.2 for x in list[2]]
    list[3] = [x * 1.4 for x in list[3]]
    list[4] = [x * 1.6 for x in list[4]]


    fig, ax = plt.subplots()
    ax.set_xlabel('Side Length(m)')
    ax.set_ylabel('VoI')
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22','#AA22CC']
    x_label = [80,100,120,140,160]
    plt.grid(True)
    plt.xticks(x_label)
    plt.plot(x_label, list[0],color=color[0],marker='o' ,label='Number of Sensors = 8')
    plt.plot(x_label, list[1],color=color[1],marker = 's',label='Number of Sensors = 10')
    plt.plot(x_label, list[2],color=color[2],marker = '^',label='Number of Sensors = 12')
    plt.plot(x_label, list[3],color=color[3],marker = '*',label='Number of Sensors = 14')
    plt.plot(x_label, list[4],color=color[4],marker = 'd',label='Number of Sensors = 16')
    plt.legend(fancybox=True, fontsize=8, ncol=1)
    plt.show()

def voi_rate_by_range_and_sensor():
    list = [[80.31257469,	78.06155189,	71.74693269,	64.33729141,	56.77791027],
            [78.41573705,	76.27486278,	70.17682965,	63.25315273,	55.97233117],
            [74.88834933,	72.81934142,	67.60688462,	59.83177463,	54.64473524],
            [72.05569199,	70.22115134,	66.03524442,	58.2131879,	53.41332034],
            [70.45345545,	69.18603157,	63.85527494,	55.60307681,	52.5273376]]

    fig, ax = plt.subplots()
    ax.set_xlabel('Side Length(m)')
    ax.set_ylabel('VoI Retention Rate(%)')
    color = ['#4169E1', '#FFA500', '#32CD32', '#DDEE22','#AA22CC']
    x_label = [80,100,120,140,160]
    plt.grid(True)
    plt.xticks(x_label)
    plt.plot(x_label, list[0],color=color[0],marker='o' ,label='Number of Sensors = 8')
    plt.plot(x_label, list[1],color=color[1],marker = 's',label='Number of Sensors = 10')
    plt.plot(x_label, list[2],color=color[2],marker = '^',label='Number of Sensors = 12')
    plt.plot(x_label, list[3],color=color[3],marker = '*',label='Number of Sensors = 14')
    plt.plot(x_label, list[4],color=color[4],marker = 'd',label='Number of Sensors = 16')
    plt.legend(fancybox=True, fontsize=8, ncol=1)
    plt.show()

def voi_by_method_and_sensor():
    list = [121.5240993,139.3031911,141.218199,141.7355745,142.1028763]
    list2 = [121.5240993,133.5278846,139.3500585,140.3099307,140.9409017]
    list3 = [121.5240993,132.6500852,135.1534567,135.7035986, 135.9534567]

    list = [141.779530719479,141.218199,138.499131618612,]
    list2 = [139.350058549271,]
    list3 = [135.70359863256,]
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of AUVs')
    ax.set_ylabel('Average VoI')
    x_label = [8,10,12,14,16]

    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-',label='H-MADDPG')
    plt.plot(x_label, list2, 'go-',label='Rule-based MADDPG')
    plt.plot(x_label, list3, 'bo-',label='Relay-free MADDPG')
    plt.legend(loc='lower right', fancybox=True, fontsize=10, ncol=1)
    plt.show()

def voi_by_sensor_and_method():
    list = [121.5240993,139.3031911,141.218199,141.7355745,142.1028763]
    list2 = [121.5240993,133.5278846,139.3500585,140.3099307,140.9409017]
    list3 = [121.5240993,132.6500852,135.1534567,135.7035986, 135.9534567]
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of AUVs')
    ax.set_ylabel('Average VoI')
    x_label = [8,10,12,14,16]

    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-',label='H-MADDPG')
    plt.plot(x_label, list2, 'go-',label='Rule-based MADDPG')
    plt.plot(x_label, list3, 'bo-',label='Relay-free MADDPG')
    plt.legend(loc='lower right', fancybox=True, fontsize=10, ncol=1)
    plt.show()

def maxvoi_by_sensor():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_sensor8/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 8)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_sensor12_12261429/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 12)

    load_path = "data_old/data_sensor14/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 14)

    load_path = "data_old/data_sensor16/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 16)
    ax.set_xlabel('Number of Sensors', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [8, 10, 12, 14, 16]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()

def voi_by_AUV():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_2/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data_4_12270926/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data_5/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)

    load_path = "data_old/data_6_12271436/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)

    ax.set_xlabel('Number of AUVs', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [2,3,4,5,6]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()


def maxvoi_by_AUV():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_1_01170947/episode_voi_list.pkl"
    # load_path = "data_1_01170947/episode_voi_list.pkl"
    load_path = "data_old/data_1_01170947/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_2/episode_voi_list.pkl"
    # load_path = "data_withoutdqn_auv2_01251618/episode_voi_list.pkl"
    load_path = "data_old/data_withoutrelay_auv2_01251932/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_withoutrelay/episode_voi_list.pkl"
    # load_path = "data_withoutdqn_01031310/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list)/ 10)
    load_path = "data_old/data_4_12270926/episode_voi_list.pkl"
    # load_path = "data_withoutdqn_auv4_01250957/episode_voi_list.pkl"
    load_path = "data_old/data_withoutrelay_auv4_01221643/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list)/ 10)
    load_path = "data_old/data_5/episode_voi_list.pkl"
    # load_path = "data_withoutdqn_auv5_01251905/episode_voi_list.pkl"
    load_path = "data/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list)/ 10)

    # load_path = "data_6_12271436/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(max(episode_reward_list)/ 10)
    print(list)
    ax.set_xlabel('Number of AUVs', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    list = [x - 1 for x in list]
    x_label = [1,2,3,4,5]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()

def voi_by_range():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_range60/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data_range80/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data_old/data_range120/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    # load_path = "data_5/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)
    #
    # load_path = "data/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)

    ax.set_xlabel('Side Length', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [60,80,100,120]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()


def maxvoi_by_range():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data_range60/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_range80/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list) / 10)
    load_path = "data_old/data_range120/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(max(episode_reward_list)/ 10)
    # load_path = "data_5/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)
    #
    # load_path = "data/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)

    ax.set_xlabel('side length', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [60,80,100,120]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()


def voi_by_methor():
    list = []
    fig, ax = plt.subplots()
    load_path = "data_old/data12072124/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    load_path = "data/episode_voi_list.pkl"
    with open(load_path, 'rb') as f:
        episode_reward_list = pickle.load(f)
    list.append(episode_reward_list[-1] / 10)
    # load_path = "data12072124/episode_voi_list.pkl"

    # load_path = "data_5/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)
    #
    # load_path = "data/episode_voi_list.pkl"
    # with open(load_path, 'rb') as f:
    #     episode_reward_list = pickle.load(f)
    # list.append(episode_reward_list[-1] / 10)

    ax.set_xlabel('side length', fontsize=12)
    ax.set_ylabel('Average VOI', fontsize=12)
    x_label = [1,2]
    plt.xticks(x_label)
    plt.plot(x_label, list, 'ro-')

    # ax.legend(loc='upper left', fancybox=True, fontsize=10, ncol=1)
    plt.show()

if __name__ == '__main__':
    # show_reward()
    # show_auv_data()
    # show_auv_p()
    # show_sensor_data()
    # show_voi()

    # draw_3D_history_with_p([100 / 2, 100 / 2, 0],)
    load_path = "data/history_p_list.pkl"
    load_path = "data_0606可能用作轨迹/history_p_list.pkl"
    with open(load_path, 'rb') as f:
        history_p_list = pickle.load(f)

    history_p_list[0] = history_p_list[0][0:4] + history_p_list[0][5:10] + history_p_list[0][12:15] + history_p_list[0][16:19]
    history_p_list[1] = history_p_list[1][0:4] + history_p_list[1][5:9] + history_p_list[1][12:17] + history_p_list[1][18:19]
    history_p_list[1][9] = 0
    history_p_list[2] = history_p_list[2][0:4] + history_p_list[2][5:7] + history_p_list[2][8:10] + history_p_list[2][11:13] + history_p_list[2][14:16] + history_p_list[2][18:19] + history_p_list[2][17:18]

    load_path = "data/history_location_list.pkl"
    load_path = "data_0606可能用作轨迹/history_location_list.pkl"
    with open(load_path, 'rb') as f:
        history_location_list = pickle.load(f)

    history_location_list[0] = history_location_list[0][0:4] + history_location_list[0][5:10] + history_location_list[0][12:15] + history_location_list[0][16:19]
    history_location_list[1] = history_location_list[1][0:4] + history_location_list[1][5:9] + history_location_list[1][12:17] + history_location_list[1][18:19]
    history_location_list[2] = history_location_list[2][0:4] + history_location_list[2][5:7] + history_location_list[2][8:10] + history_location_list[2][11:13] + history_location_list[2][14:16] + history_location_list[2][18:19] + history_location_list[2][17:18]

    load_path = "data/sensor_location_list.pkl"
    load_path = "data_0606可能用作轨迹/sensor_location_list.pkl"
    with open(load_path, 'rb') as f:
        sensor_location_list = pickle.load(f)

    print(sensor_location_list)

    draw_3D_history_with_p(history_location_list, sensor_location_list, history_p_list)


    # compare_reward()
    # compare_voi()

    # compare_reward_sensor()
    # compare_voi_sensor()
    compare_reward_methor()
    compare_voi_methor()
    #
    # # voi_by_AUV()
    # maxvoi_by_AUV()
    # voi_by_sensor()
    # # # maxvoi_by_sensor()
    # voi_by_range()
    # # # maxvoi_by_range()
    # voi_by_methor()

    # voi_by_method_and_auv()
    # voi_by_method_and_speed()
    # voi_by_speed_and_auv()
    # voi_by_speed_and_method()
    # voi_by_method_and_sensor()
    # voi_by_auv_and_sensor()
    # voi_by_auv_rate_and_sensor()
    # voi_by_range_and_sensor()
    # voi_rate_by_range_and_sensor()
