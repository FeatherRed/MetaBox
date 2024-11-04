import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import Basic_Optimizer

class DE_current_to_best_1(Basic_Optimizer):
    def __init__(self, config):
        super(DE_current_to_best_1, self).__init__(config)
        self.__dim = config.dim
        self.__MaxFEs = config.maxFEs
        self.__F = 0.9
        self.__Cr = 0.1
        self.__NP = 100

        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval

    def __get_cost(self, x, problem):
        if problem.optimum is None:
            cost = problem.eval(x)
        else:
            cost = problem.eval(x) - problem.optimum
        return cost

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]


    def __init_population(self, problem):
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb)
        self.__cost = self.__get_cost(self.__population, problem)
        self.__FEs = self.__NP
        self.gbest = np.min(self.__cost)
        self.log_index = 1
        self.cost = [self.gbest]

    def __update(self, problem):
        self.__sort()
        NP, dim = self.__NP, self.__dim
        v = self.__current_to_best_1(self.__population)

        v[v < problem.lb] = (v[v < problem.lb] + problem.lb) / 2
        v[v > problem.ub] = (v[v > problem.ub] + problem.ub) / 2

        u = self.__binomial(self.__population, v)

        new_cost = self.__get_cost(u, problem)
        self.__FEs += NP
        optim = np.where(new_cost < self.__cost)[0]
        self.__population[optim] = u[optim]
        self.__cost = np.minimum(self.__cost, new_cost)
        self.__sort()

        if np.min(self.__cost) < self.gbest:
            self.gbest = np.min(self.__cost)

        if self.__FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest)

        if problem.optimum is None:
            return False
        else:
            return self.gbest <= 1e-8

    def __current_to_best_1(self, group):
        NP, dim = group.shape
        # rb
        rb = 0

        # r1
        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((rb == r1) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((rb == r1) + (r1 == np.arange(NP)))[0]
            count += 1

        # r2
        count = 0
        r2 = np.random.randint(NP, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.arange(NP)))[0]
            count += 1

        xb = group[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = group + self.__F * (xb - group) + self.__F * (x1 - x2)
        return v

    def __binomial(self, x, v):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size = NP)
        u = np.where(np.random.rand(NP, dim) <= self.__Cr, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def run_episode(self, problem):
        self.__init_population(problem)
        while self.__FEs < self.__MaxFEs:
            is_done = self.__update(problem)
            if is_done:
                break
        if len(self.cost) >= self.__n_logpoint + 1:
            self.cost[-1] = self.gbest
        else:
            self.cost.append(self.gbest)
        return {'cost': self.cost, 'fes': self.__FEs}