import numpy as np
import time

class Obj_func:

    def __init__(self, data_generator):
        
        self.data_generator = data_generator

    def obj_func(self, weights, *args):

        list_of_score  = []
        
        while True: 
            try:
                data_batch  = next(self.data_generator)
                pred        = np.square(data_batch[:, 0] - weights[0])
                rmse        = np.sqrt(np.square(data_batch[:, 1] - pred))
                list_of_score.append(rmse)

            except StopIteration:
                break

        avg_score      = np.mean(list_of_score)

        return avg_score

class BatchOptimization:

    def __init__(self, optimizer, optimizer_params, Obj_func):

        self.data_generator = self.create_dataGenerator()
        self.obj_depression = Obj_func(data_generator=self.data_generator)
        self.opt            = optimizer(obj_func=self.obj_func, **optimizer_params)

    def create_dataGenerator(self, seed=1234, param_to_be_estimated=0.7325):
        
        np.random.seed(seed=seed)

        for i in range(10):
            data        = np.random.random(size=(10000,2))
            data[:, 1]  = np.square(data[:, 0] - param_to_be_estimated)
            yield data

    def obj_func(self, weights, *args):

        score = self.obj_depression.obj_func(weights, *args)

        return score

    def train(self, *args):

        results             = self.opt.optimize(*args)
        self.best_score     = results[0]
        self.best_weights   = results[1]

        return self.best_score, self.best_weights

class BruteForce:

    def __init__(self, obj_func, npar, n_iteration):

        self.obj_func    = obj_func
        self.npar        = npar
        self.n_iteration = n_iteration

    def optimize(self, *args):

        self.list_of_score = []

        for i in range(self.n_iteration):
            np.random.seed()
            weights   = np.random.random(size=(self.npar, ))
            score     = self.obj_func(weights, *args)
            self.list_of_score.append((score, weights))

        self.list_of_score = sorted(self.list_of_score)
        best_score         = self.list_of_score[0][0]
        best_weight        = self.list_of_score[0][1]

        return best_score, best_weight




BO  = BatchOptimization(optimizer        = BruteForce, 
                        optimizer_params = {'npar'        : 1, 
                                            'n_iteration' : 100},
                        Obj_func         = Obj_func)
best_score, best_weight = BO.train()

print('------------[Result]-------------')
print('score:', best_score)
print('weight:', best_weight)