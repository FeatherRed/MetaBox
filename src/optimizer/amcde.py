import numpy as np
import scipy.stats as stats
from optimizer.basic_optimizer import Basic_Optimizer
from scipy.optimize import minimize



class AMCDE(Basic_Optimizer):
    def __init__(self, config):
        super(AMCDE, self).__init__(config)
        self.__dim = config.dim

        self.__F0 = 0.5
        self.__Cr0 = 0.5
        self.__a = 2.6
        self.__p = 0.4
        self.__pbc1 = 0.4
        self.__pbc2 = 0.4
        self.__pw = 0.2
        self.__pr = 0.01
        self.__pls = 0.1
        self.__Gn = 5
        self.__FEs = 0
        self.__H = 20 * self.__dim
        self.__Nmax = 6 * self.__dim * self.__dim
        self.__Nmin = 4

        self.__MaxFEs = config.maxFEs
        # maxfes * 0.85
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval
    # DE/current-to-faibest/1 with archive
    def __ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        # fai-best
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
    # DE/current-to-faibest/1
    def __ctb_w(self, group, best, Fs):
        NP, dim = group.shape
        NB = best.shape[0]

        # fai-best
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
        r2 = np.random.randint(NP, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.array(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.array(NP)))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]

        v = x1 + Fs * (xb - group + x1 - x2)
        return v
    # DE/weighted-rand-to-faibest/1
    def __weighted_rtb(self, group, best, Fs):
        NP, dim = group.shape
        NB = best.shape[0]

        # fai-best
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
        r2 = np.random.randint(NP, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.array(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == r1) + (r2 == np.array(NP)))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * (xb - x2)
        return v

    def __choose_F_Cr(self):
        pop_size = self.__NP
        ind_r = np.random.randint(0, self.__MF.shape[0], size = pop_size) # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc = self.__MCr[ind_r], scale = 0.1, size = pop_size)))
        cauchy_locs = self.__MF[ind_r]
        F = stats.cauchy.rvs(loc = cauchy_locs, scale = 0.1, size = pop_size)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    def __mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    def __update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.__mean_wL(df, SF)
            self.__MF[self.__k] = mean_wL
            mean_wL = self.__mean_wL(df, SCr)
            self.__MCr[self.__k] = mean_wL
            self.__k = (self.__k + 1) % self.__MF.shape[0]
        else:
            self.__MF[self.__k] = 0.5
            self.__MCr[self.__k] = 0.5

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __get_cost(self, x, problem):
        if problem.optimum is None:
            cost = problem.eval(x)
        else:
            cost = problem.eval(x) - problem.optimum
        return cost

    def __init_population(self, problem):
        self.__NP = self.__Nmax
        self.__NA = int(self.__a * self.__NP) # |A|_max = NP
        self.__population = np.random.rand(self.__NP, self.__dim) * (problem.ub - problem.lb)
        self.__cost = self.__get_cost(self.__population, problem)
        self.__FEs = self.__NP
        self.__archive = np.array([])

        self.__MF = np.ones(self.__H) * self.__F0
        self.__MCr = np.ones(self.__H) * self.__Cr0
        self.__k = 0

        self.__gn = 0
        self.g = 1
        self.gbest = np.min(self.__cost)

        self.__monopolizer = 0 # [0, 1, 2] ------> []
        self.log_index = 1
        self.cost = [self.gbest]

        self.__f = np.zeros(3) # x -> u 总改进量
        self.__fn = np.zeros(3) # 控制数目
        self.__flag = 0 # competition is first

        self.__fes_sqp = 0
        self.__S = [np.array([]), np.array([]), np.array([])]

    def __binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size = NP)
        u = np.where(np.random.rand(NP, dim) <= Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def __exponential(self, x, v, Crs):
        NP, dim = x.shape
        l = np.random.randint(dim, size = NP)
        Lrand = np.where(np.random.rand(NP, dim) <= Crs, 1, 0)
        r = Lrand.argmin(axis = -1) + l
        r = np.where(r < dim, r, dim)
        mask = np.zeros((NP, dim), dtype = bool)
        for i in range(NP):
            idx = np.arange(l[i], r[i])
            mask[i, idx] = True

        u = np.copy(x)
        u[mask] = v[mask]
        return u

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.__population[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[np.random.randint(self.__archive.shape[0])] = self.__population[old_id]

    def __cal_U(self, S, p):
        u = S
        for pi in p:
            if not np.all(u == pi):
                pi = pi[None,:]
                u = np.concatenate((u, pi), axis = 0)
        return u

    def __cal_div(self, S, p):
        u = S
        x = np.array([])
        # U
        for pi in p:
            if not np.all(u == pi):
                pi = pi[None, :]
                u = np.concatenate((u, pi), axis = 0)
            else:
                pi = pi[None, :]
                x = np.concatenate((x, pi), axis = 0)

        # -
        for xi in x:
            idx = np.where((u == xi).all(axis = 1))
            u = np.delete(u, idx, axis = 0)
        return u

    def objective_function(self, x, problem):
        self.__fes_sqp += 1
        return self.__get_cost(x, problem)

    def __sqp(self, x_best, problem):
        self.__fes_sqp = 0
        dim = len(x_best)
        bounds = [(problem.lb, problem.ub)] * dim
        result = minimize(self.objective_function, x_best, method = 'SLSQP',
                          args = (problem,), bounds = bounds)
        return result.x, result.fun

    def __update(self, problem):
        self.__sort()
        NP, dim = self.__NP, self.__dim
        Cr, F = self.__choose_F_Cr()
        pbest = self.__population[:max(int(self.__p * NP), 2)]
        Fs = F.repeat(dim).reshape(NP, dim)
        p = self.__population
        if self.__gn < self.__Gn:
            # monopolize
            if self.__monopolizer == 0:
                v = self.__ctb_w_arc(p, pbest, self.__archive, Fs)
            elif self.__monopolizer == 1:
                v = self.__ctb_w(p, pbest, Fs)
            else:
                v = self.__weighted_rtb(p, pbest, Fs)

            # 越界处理
            v[v < problem.lb] = (v[v < problem.lb] + problem.lb) / 2
            v[v > problem.ub] = (v[v > problem.ub] + problem.ub) / 2
            rvs = np.random.rand(NP)
            u = np.zeros((NP, dim))

            Crs = Cr.repeat(dim).reshape(NP, dim)
            # binomial
            bu = v[rvs <= self.__pbc1]
            bu = self.__binomial(p[rvs <= self.__pbc1], bu, Crs[rvs <= self.__pbc1])
            u[rvs <= self.__pbc1] = bu
            # exponential
            eu = v[rvs > self.__pbc1]
            eu = self.__exponential(p[rvs > self.__pbc1], eu, Crs[rvs > self.__pbc1])
            u[rvs > self.__pbc1] = eu
            new_cost = self.__get_cost(u, problem)
            new_gbest = np.min(new_cost)
            optim = np.where(new_cost < self.__cost)[0]
            for i in optim:
                self.__update_archive(i)


            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, self.__cost - new_cost)
            self.__update_M_F_Cr(SF, SCr, df[optim])


            self.__population[optim] = u[optim]
            self.__cost = np.minimum(self.__cost, new_cost)
            if new_gbest < self.gbest:
                self.gbest = new_gbest
                self.__gn = 0
            else:
                self.__gn += 1
        else:
            # competition

            # 先判断是不是空
            if self.__fn[0] == 0:
                c = np.ones(3) / 3

            else:
                aver_f = self.__f / self.__fn
                c = aver_f / np.sum(aver_f)
                while np.min(c) < 0.1:
                    i_max = np.argmax(c)
                    i_min = np.argmin(c)
                    c[i_max] -= 0.1 - c[i_min]
                    c[i_min] = 0.1

            index = np.arange(NP)
            np.random.shuffle(index)
            if self.__flag == 0:
                # 分配相等个数
                idx = np.array_split(index, 3)
            else:
                # 第二代
                n1 = int(c[0] * NP)
                n2 = int(c[1] * NP)
                idx = np.array_split(index, [n1, n1 + n2])

            p1 = self.__population[idx[0]]
            p2 = self.__population[idx[1]]
            p3 = self.__population[idx[2]]

            win = np.argmax(c)

            # if self.__flag == 1:
            #     for i in range(3):
            #         if i == win:
            #             self.__S[i] = self.__cal_U(self.__S[i], self.__population[idx[i]])
            #         else:
            #             self.__S[i] = self.__cal_div(self.__S[i], self.__population[idx[i]])
            # else:
            #     self.__flag = 1
            #     self.__S[0] = p1
            #     self.__S[1] = p2
            #     self.__S[2] = p3
            #     pass

            v1 = self.__ctb_w_arc(p1, pbest, self.__archive, Fs[idx[0]])
            v2 = self.__ctb_w(p2, pbest, Fs[idx[1]])
            v3 = self.__weighted_rtb(p3, pbest, Fs[idx[2]])
            v = np.zeros((NP, dim))
            v[idx[0]] = v1
            v[idx[1]] = v2
            v[idx[2]] = v3

            v[v < problem.lb] = (v[v < problem.lb] + problem.lb) / 2
            v[v > problem.ub] = (v[v > problem.ub] + problem.ub) / 2

            Crs = 0

            if self.__FEs >= 0.5 * self.__MaxFEs:
                Crs = 2 * self.__FEs / self.__MaxFEs -1

            rvs = np.random.rand(NP)
            u = np.zeros((NP, dim))
            bu = v[rvs <= self.__pbc2]
            bu = self.__binomial(p[rvs <= self.__pbc2], bu, Crs)
            u[rvs <= self.__pbc2] = bu
            eu = v[rvs > self.__pbc2]
            eu = self.__exponential(p[rvs > self.__pbc2], eu, Crs)
            u[rvs > self.__pbc2] = eu

            new_cost = self.__get_cost(u, problem)


            new_gbest = np.min(new_cost)
            new_gbest_idx = np.argmin(new_cost)

            optim = np.where(new_cost < self.__cost)[0]
            for i in optim:
                self.__update_archive(i)

            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, self.__cost - new_cost)
            self.__update_M_F_Cr(SF, SCr, df[optim])

            for i in range(3):
                self.__f[i] += np.sum(df[idx[i]])
                self.__fn[i] += len(idx[i])



            self.__population[optim] = u[optim]
            self.__cost = np.minimum(self.__cost, new_cost)
            ind = np.argsort(self.__cost)
            worst_ind = ind[-max(int(self.__pw * NP), 2):]
            if np.random.rand() < self.__pr:
                self.__population[worst_ind] = u[worst_ind]
                self.__cost[worst_ind] = new_cost[worst_ind]


            if new_gbest < self.gbest:
                self.__gn = 0
                self.__Gn += 1
                for i in range(3):
                    if new_gbest_idx in idx[i]:
                        self.__monopolizer = i
                self.gbest = new_gbest
                # self.__S = [np.array([]), np.array([]), np.array([])]



        self.__FEs += NP
        self.g += 1
        # local search
        self.__sort()

        if self.__FEs >= 0.85 * self.__MaxFEs:
            x_best = self.__population[0]
            if np.random.rand() < self.__pls:
                x_sqp, y_sqp = self.__sqp(x_best, problem)
                if y_sqp < self.__cost[0]:
                    self.__pls = 0.1
                    self.__population[0] = x_sqp
                    self.__cost[0] = y_sqp
                else:
                    self.__pls = 0.0001
            self.__FEs += self.__fes_sqp

        self.__NP = int(np.round(self.__Nmax + (self.__Nmin - self.__Nmax) * self.__FEs / self.__MaxFEs))
        self.__NA = int(self.__a * self.__NP)
        self.__population = self.__population[: self.__NP]
        self.__cost = self.__cost[: self.__NP]
        self.__archive = self.__archive[: self.__NA]

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
