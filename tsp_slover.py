import random
import math
import numpy as np
import matplotlib.pyplot as plt


class GA_PSO(object):
    def __init__(self, num_city, num_total, iteration, data):
        """
        初始化遗传算法（GA）和粒子群优化算法（PSO）类。

        参数：
        - num_city: 城市数量
        - num_total: 种群大小
        - iteration: 迭代次数
        - data: 城市坐标数据
        """
        self.num_city = num_city
        self.num_total = num_total
        self.iteration = iteration
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, data)  # 计算城市间距离矩阵
        self.fruits = self.greedy_init(
            self.dis_mat, num_total, num_city
        )  # 使用贪婪算法初始化种群
        self.particals = self.greedy_init(
            self.dis_mat, num_total, num_city
        )  # 使用贪婪算法初始化粒子群
        self.local_best = self.particals.copy()  # 本地最优解
        self.local_best_len = self.compute_paths(
            self.local_best
        )  # 本地最优解的路径长度
        self.global_best = self.fruits[0]  # 全局最优解
        self.global_best_len = self.compute_pathlen(
            self.global_best, self.dis_mat
        )  # 全局最优解的路径长度
        self.best_l = self.global_best_len  # 最佳路径长度
        self.best_path = self.global_best  # 最佳路径
        self.iter_x = [0]  # 迭代次数的记录
        self.iter_y = [self.best_l]  # 最佳路径长度的记录

    # 随机初始化种群
    def random_init(self, num_total, num_city):
        """
        随机初始化种群。
        """
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 使用贪婪算法初始化种群
    def greedy_init(self, dis_mat, num_total, num_city):
        """
        使用贪婪算法初始化种群。
        """
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算城市间距离矩阵
    def compute_dis_mat(self, num_city, location):
        """
        计算城市间距离矩阵。
        """
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        """
        计算路径长度。
        """
        result = dis_mat[path[-1]][path[0]]
        for i in range(len(path) - 1):
            result += dis_mat[path[i]][path[i + 1]]
        return result

    # 计算种群的路径长度
    def compute_paths(self, paths):
        """
        计算种群的路径长度。
        """
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 交叉操作
    def ga_cross(self, x, y):
        """
        遗传算法中的交叉操作。
        """
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)
        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)
        assert len(x_conflict_index) == len(y_confict_index)

        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]
        return list(x), list(y)

    # 变异操作
    def ga_mutate(self, gene):
        """
        遗传算法中的变异操作。
        """
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    # 选择父代
    def ga_parent(self, scores, ga_choose_ratio):
        """
        选择父代。
        """
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0 : int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    # 选择操作
    def ga_choose(self, genes_score, genes_choose):
        """
        选择操作。
        """
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def distance(self, tu):
        """
        计算两个城市之间的距离。
        """
        x, y = zip(*tu)
        return np.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))

    # PSO算法
    def pso(self):
        """
        粒子群优化算法（PSO）。
        """
        for cnt in range(1, self.iteration):
            for i, one in enumerate(self.particals):
                tmp_l = self.local_best_len[i]
                new_one, new_l = self.ga_cross(one, self.local_best[i])
                new_l = self.compute_pathlen(new_l, self.dis_mat)
                if new_l < self.best_l:
                    self.best_l = new_l
                    self.best_path = new_one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                new_one, new_l = self.ga_cross(one, self.global_best)
                new_l = self.compute_pathlen(new_l, self.dis_mat)
                if new_l < self.best_l:
                    self.best_l = new_l
                    self.best_path = new_one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                one = self.ga_mutate(one)
                tmp_l = self.compute_pathlen(one, self.dis_mat)
                if new_l < self.best_l:
                    self.best_l = new_l
                    self.best_path = one
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                self.particals[i] = one
                self.local_best_len[i] = tmp_l
            self.eval_particals()
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            print(cnt, self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    # 评估粒子群
    def eval_particals(self):
        """
        评估粒子群。
        """
        min_lenth = min(self.local_best_len)
        min_index = self.local_best_len.index(min_lenth)
        cur_path = self.particals[min_index]
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        for i, l in enumerate(self.local_best_len):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 运行算法
    def run(self):
        """
        运行算法。
        """
        best_length, best_path = self.pso()
        return best_path, best_length


# 读取TSP数据
def read_tsp(path):
    """
    读取TSP数据。
    """
    lines = open(path, "r").readlines()
    assert "NODE_COORD_SECTION\n" in lines
    index = lines.index("NODE_COORD_SECTION\n")
    data = lines[index + 1 : -1]
    tmp = []
    for line in data:
        line = line.strip().split(" ")
        if line[0] == "EOF":
            continue
        tmpline = []
        for x in line:
            if x == "":
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


# 读取TSP数据
data = read_tsp("data/eil101.tsp")
# data = read_tsp('data/att48.tsp')
data = np.array(data)
data = data[:, 1:]
# 初始化模型并运行
model = GA_PSO(num_city=data.shape[0], num_total=200, iteration=100, data=data.copy())
Best_path, Best = model.run()
Best_path.append(Best_path[0])
show_data = []
for i in range(len(data) + 1):
    show_data.append(data[Best_path[i]])
x, y = zip(*show_data)
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(x, y)
axs[0].plot(x, y)
axs[0].set_title("Planning Results")
iterations = model.iter_x
best_record = model.iter_y
axs[1].plot(iterations, best_record)
axs[1].set_title("Convergence Curve")
plt.show()
