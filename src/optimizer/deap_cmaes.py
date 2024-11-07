import numpy as np
import cma
from optimizer.basic_optimizer import Basic_Optimizer

class DEAP_CMAES(Basic_Optimizer):
    def __init__(self, config):
        super(DEAP_CMAES, self).__init__(config)
        config.NP = 50
        self.__config = config
        self.log_interval = config.log_interval

    def run_episode(self, problem):

        def problem_eval(x):
            if problem.optimum is None:
                fitness = problem.eval(x)
            else:
                fitness = problem.eval(x) - problem.optimum
            return fitness

        dim = self.__config.dim
        centroid = [problem.ub] * dim
        sigma = 0.5
        lambda_ = self.__config.NP
        FEs = self.__config.maxFEs
        strategy = cma.CMAEvolutionStrategy(centroid, sigma, {'popsize': lambda_,
                                                                'bounds': [problem.lb, problem.ub],
                                                                'maxfevals': FEs,
                                                                'verbose': False})
        fes = 0
        log_index = 0
        cost = []
        while True:
            X = strategy.ask()
            X = np.ascontiguousarray(X)
            new_cost = problem_eval(X)
            strategy.tell(X, new_cost)
            fes += self.__config.NP
            gbest = np.min(new_cost)
            if fes >= log_index * self.log_interval:
                log_index += 1
                cost.append(gbest)
            if problem.optimum is None:
                done = fes >= self.__config.maxFEs
            else:
                done = fes >= self.__config.maxFEs or gbest <= 1e-8

            if done:
                if len(cost) >= self.__config.n_logpoint + 1:
                    cost[-1] = gbest
                else:
                    cost.append(gbest)
                break
        return {'cost': cost, 'fes': fes}

