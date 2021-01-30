# Learning Dynamics / Computational Game Theory Assignment 3
# Exercise 1

# 22 December 2020
# Horatiu Luci

# Year 1 - M-SECU-C @ ULB
# VUB Guest-Student ID: 0582214
# ULB Student ID: 000516512

from tqdm import tqdm
import numpy as np
from random import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns



class Random:
    def __init__(self, act = 4, t = False):
        self.act = 4
        self.t   = False
    def pull(self, reward_med):
        return randint(0, self.act - 1)


class Softmax:
    def __init__(self, tau, act = 4, t = False):
        self.act = act
        self.tau = tau
        self.t   = t
    def pull(self, Q_vals):
        probas = [0 for i in range(0, self.act)]
        for a in range(0, self.act):
            n = np.exp(Q_vals[a] / self.tau)
            s = 0
            for aaction in range(0, self.act):
                s += np.exp(Q_vals[aaction] / self.tau)
            probas[a] = n/s
        return np.random.choice(self.act, p = probas)


class Egreed:
    def __init__(self, eps, act = 4, t = False):
        self.act = act
        self.eps = eps
        self.t   = t
    def pull(self, Q_vals):
        if(uniform(0,1) < self.eps):
            return randint(0, self.act - 1)
        elif(self.eps):
                return np.argmax(Q_vals)
        else:
            return randint(0, self.act - 1)


class Simulate:
    def __init__(self, algo, act = 4, act_dict = {0 : [2.2, 0.4], 1 : [2.6, 0.8], 2 : [1.0, 2.0], 3 : [1.4, 0.6]}, iter = 1000):
        self.algo                 = algo
        self.act                  = act
        self.act_dict             = act_dict
        self.iter                 = iter
        self.act_progress         = [0 for i in range(0, iter)]
        self.av_progress          = [0 for i in range(0, iter)]
        self.reward_progress      = [0 for i in range(0, iter)]
        self.act_k                = [0 for i in range(0, self.act)]
        self.reward_med           = np.array([0. for i in range(0, act)])
        self.rewards_progress     = [[0 for ii in range(0, act)] for i in range(0, iter)]
        self.rewards_ca           = np.zeros(iter)
        self.norms                = []
        for i in range(act):
            self.norms.append(np.random.normal(act_dict[i][0], act_dict[i][1], iter))

    def get_reward(self, act):
        m, s = self.act_dict[act]
        return np.random.normal(m, s)

    def sim(self):
        for r in range(0, self.iter):
            if(self.algo.t):
                if(isinstance(self.algo, Softmax)):
                    self.algo.tau = 4 * (self.iter - r) / self.iter
                else:
                    self.algo.eps = 1 / np.sqrt(r + 1)
            if(isinstance(self.algo, Egreed) and r == 0):
                if(self.algo.eps == 0):
                    self.algo.eps = -1
                act = self.algo.pull(self.reward_med)
                if(self.algo.eps == -1):
                    self.algo.eps = 0
            else:
                act = self.algo.pull(self.reward_med)
            reward = self.get_reward(act)
            self.reward_progress[r] = reward
            self.av_progress[r] = (sum(self.av_progress) + reward) / (r + 1)
            self.act_progress[r] = act
            self.act_k[act] += 1
            s = [0 for i in range(0, self.act)]
            cumulative = 0
            for action in range(0, self.act):
                if(self.act_progress[r] == action):
                    self.reward_med[action] += (reward - self.reward_med[action]) / (self.act_k[action])
                cumulative += self.reward_med[action]
                self.rewards_progress[r][action] = self.reward_med[action]
            self.rewards_ca[r] = sum(self.av_progress)

    def get_reward_av(self, arm):
        reward_med = np.zeros(self.iter)
        for i in range(self.iter):
            reward_med[i] = self.rewards_progress[i][arm]
        return reward_med

    def get_act_k(self):
        return self.act_k

    def get_crewards(self):
        return self.rewards_ca



def plot_cumul(cumulative, t = False):
    x = np.arange(1000)
    plt.figure()
    plt.grid()
    plt.title("Cumulative rewards over time")
    for i in range(0, len(cumulative)):
        plt.plot(x, cumulative[i])
    if not t:
        plt.legend(str_names)
    else:
        plt.legend(namelistTF)
    plt.savefig("cumulative_plot.png")



def plot_hist(selection_k, t = False):
    act = len(selection_k[0])
    ax_x = np.arange(act)
    k = 0
    for algorithm in selection_k:
        plt.figure()
        plt.grid()
        if not t:
            plt.title("Select counter for {}".format(name_struct[i]))
        else:
            plt.title("Select counter for {}".format(nameTF[i]))
        plt.bar(ax_x, algorithm, alpha = 0.8, align = 'center')
        plt.xticks(ax_x)
        plt.savefig("histogram-{}.png".format(k + 1))
        k += 1



def plot_arm(average_reward, arm, t = False):
    plt.figure()
    plt.grid()
    plt.axhline(y = act_struct[arm][0], color='black', linestyle = 'solid')
    x = np.arange(1000)
    plt.title("Arm number {}'s rewards over time".format(arm+1))
    for i in range(0, len(average_reward)):
        plt.plot(x, average_reward[i][arm])
    if not t:
        plt.legend([r"$Q_{}$ - {}".format(arm, act_struct[arm][0])] + str_names)
    else:
        plt.legend([r"$Q_{}$ - {}".format(arm, act_struct[arm][0])] + namelistTF)
    plt.savefig("arm-{}.png".format(arm + 1))



def run_sim(sim_type, sim_dict):
    selected_ac = np.array([0 for i in range(sim_dict[sim_type].act)])
    selected_c = []
    selected_aa = [[] for i in range(sim_dict[sim_type].act)]
    selected = []
    c_av = np.array([0. for i in range(0, Simulate(sim_dict[sim_type]).iter)])
    average = np.array([ [0. for i in range (Simulate(sim_dict[sim_type]).iter)] for ii in range(len(selected_aa))])

    for i in range(0, 30):
        technik = sim_dict[sim_type]
        s = Simulate(technik, act_dict = act_struct)
        s.sim()
        selected_c.append(s.get_crewards())
        selected.append(s.get_act_k())
        for a in range(0, sim_dict[sim_type].act):
            selected_aa[a].append(s.get_reward_av(a))

    for i in range(0, len(c_av)):
        avg_cumulative_position = 0
        for trial in selected_c:
            avg_cumulative_position += trial[i]
        c_av[i] = avg_cumulative_position
    c_av = c_av / len(selected_c)

    for arm in range(len(selected_aa)):
        for r in range(0, len(average[0])):
            arm_sum = 0
            for trial in selected_aa[arm]:
                arm_sum += trial[r]
            average[arm][r] = round(arm_sum / len(selected_aa[arm]), 5)

    for trial in selected:
        for i in range(0, len(trial)):
            selected_ac[i] += trial[i]
    selected_ac = selected_ac/len(selected)

    return c_av, selected_ac, average




# 1.1 - Last number = 2, selected Table 3
# uncomment to run
'''
sim_struct = {0: Random(), 1: Egreed(0), 2: Egreed(0.1), 3: Egreed(0.2), 4: Softmax(1), 5: Softmax(0.1)}
name_struct = {0: "Random", 1: r"$\epsilon$ - greedy $\epsilon - 0$", 2: r"$\epsilon$ - greedy $\epsilon - 0.1$", 3: r"$\epsilon$ - greedy $\epsilon - 0.2$", 4: r"Softmax $\tau - 1$", 5: r"Softmax $\tau - 0.1$"}
str_names = []

for key, value in name_struct.items():
    str_names.append(value)

act_struct = {0: [2.2, 0.4], 1: [2.6, 0.8], 2: [1.0, 2.0], 3: [1.4, 0.6]}
av_s, av_c, av_a = [], [], []


for i in tqdm(range(0, 6)):
    res = run_sim(i, sim_struct)
    av_c.append(res[0])
    av_s.append(res[1])
    av_a.append(res[2])


for arm in range(0, len(list(act_struct.keys()))):
    plot_arm(av_a, arm, t = False)
plot_hist(av_s, t = False)
plot_cumul(av_c, t = False)
'''






# 1.2 - Last number = 2, selected Table 3
# uncomment to run


sim_struct = {0: Random(), 1: Egreed(0), 2: Egreed(0.1), 3: Egreed(0.2), 4: Softmax(1), 5: Softmax(0.1)}
name_struct = {0: "Random", 1: r"$\epsilon$ - greedy $\epsilon - 0$", 2: r"$\epsilon$ - greedy $\epsilon - 0.1$", 3: r"$\epsilon$ - greedy $\epsilon - 0.2$", 4: r"Softmax $\tau - 1$", 5: r"Softmax $\tau - 0.1$"}
str_names = []

for key, value in name_struct.items():
    str_names.append(value)

act_struct = {0: [2.2, 0.8], 1: [2.6, 1.6], 2: [1.0, 4.0], 3: [1.4, 1.2]}
av_s, av_c, av_a = [], [], []


for i in tqdm(range(0, 6)):
    res = run_sim(i, sim_struct)
    av_c.append(res[0])
    av_s.append(res[1])
    av_a.append(res[2])


for arm in tqdm(range(0, len(list(act_struct.keys())))):
    plot_arm(av_a, arm, t = False)
plot_hist(av_s, t = False)
plot_cumul(av_c, t = False)




# 1.3 - Last number = 2, selected Table 3
# uncomment to run
'''
sim_struct = {0: Random(), 1: Egreed(0), 2: Egreed(0.1), 3: Egreed(0.2), 4: Softmax(1), 5: Softmax(0.1), 6: Egreed(True, t = True), 7: Softmax(True, t = True)}
nameTF = {0: "Random", 1: r"$\epsilon$ - greedy $\epsilon - 0$", 2: r"$\epsilon$ - greedy $\epsilon - 0.1$", 3: r"$\epsilon$ - greedy $\epsilon - 0.2$", 4: r"Softmax $\tau - 1$", 5: r"Softmax $\tau - 0.1$", 6: r"$\epsilon$ - greedy $\epsilon - 1 / \sqrt{t}$", 7: r"Softmax - $\tau - 4 * (100-t)/100$"}

namelistTF = []

for key, value in nameTF.items():
    namelistTF.append(value)

act_struct = {0: [2.2, 0.4], 1: [2.6, 0.8], 2: [1.0, 2.0], 3: [1.4, 0.6]}
av_s, av_c, av_a = [], [], []
for i in tqdm(range(0, 8)):
    res = run_sim(i, sim_struct)
    av_c.append(res[0])
    av_s.append(res[1])
    av_a.append(res[2])
for arm in tqdm(range(0, len(list(act_struct.keys())))):
    plot_arm(av_a, arm, t = True)
plot_hist(av_s, t = True)
plot_cumul(av_c, t = True)
'''
