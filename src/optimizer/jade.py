import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import Basic_Optimizer

class JADE(Basic_Optimizer):
    def __init__(self, config):
        super(JADE, self).__init__(config)
        self.__dim = config.dim
        self.__c = 0.1
        self.__p = 0.05

        self.__MaxFEs = config.maxFEs
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval

    # DE/current-to-pbest/1 with archive
    def __ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        # p-best
        count = 0
        rb = np.random.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        # r1
        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        # r2
        count = 0
        r2 = np.random.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)
        return v

    def __get_cost(self, x, problem):
        if problem.optimum is None:
            cost = problem.eval(x)
        else:
            cost = problem.eval(x) - problem.optimum
        return cost

    def __init_population(self, problem):
        if self.__dim <= 10:
            self.__NP = 30
        elif self.__dim <= 30:
            self.__NP = 100
        else:
            self.__NP = 400
        self.__NA = self.__NP
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb)
        self.__cost = self.__get_cost(self.__population, problem)

        self.__mean_f = 0.5
        self.__mean_cr = 0.5
        self.__archive = np.array([])

        self.__FEs = self.__NP
        self.gbest = np.min(self.__cost)

        self.log_index = 1
        self.cost = [self.gbest]

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __choose_F_Cr(self):
        pop_size = self.__NP
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc = self.__mean_cr, scale = 0.1, size = pop_size)))
        F = stats.cauchy.rvs(loc = self.__mean_f, scale = 0.1, size = pop_size)
        err = np.where(F < 0)[0]
        F[err] = 2 * self.__mean_f - F[err]
        return C_r, np.minimum(1, F)

    def __binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size = NP)
        u = np.where(np.random.rand(NP, dim) <= Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.__population[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[np.random.randint(self.__archive.shape[0])] = self.__population[old_id]

    def __mean_L(self, s):
        if np.sum(s) > 0.000001:
            return np.sum((s ** 2)) / np.sum(s)
        else:
            return 0.5

    def __update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_L = self.__mean_L(SF)
            self.__mean_f = (1 - self.__c) * self.__mean_f + self.__c * mean_L
            mean_A = np.mean(SCr)
            self.__mean_cr = (1 - self.__c) * self.__mean_cr + self.__c * mean_A
        else:
            self.__mean_f = 0.5
            self.__mean_cr = 0.5

    def __update(self, problem):
        self.__sort()
        NP, dim = self.__NP, self.__dim
        Cr, F = self.__choose_F_Cr()
        pbest = self.__population[:max(int(self.__p * NP), 2)]
        Fs = F.repeat(dim).reshape(NP, dim)

        v = self.__ctb_w_arc(self.__population, pbest, self.__archive, Fs)
        # 越界处理
        v[v < problem.lb] = (v[v < problem.lb] + problem.lb) / 2
        v[v > problem.ub] = (v[v > problem.ub] + problem.ub) / 2

        Crs = Cr.repeat(dim).reshape(NP, dim)
        # binomial
        u = self.__binomial(self.__population, v, Crs)

        new_cost = self.__get_cost(u, problem)
        self.__FEs += NP
        optim = np.where(new_cost < self.__cost)[0]
        for i in optim:
            self.__update_archive(i)
        SF = F[optim]
        SCr = Cr[optim]
        df = np.maximum(0, self.__cost - new_cost)
        self.__update_M_F_Cr(SF, SCr, df[optim])

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