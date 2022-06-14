import numpy as np
import matplotlib.pyplot as plt
from utils import *
from algo import *
from metrics import *
from sample import *
from tqdm import tqdm

# Settings

mode = 1 # 0 for performance; 1 for communication cost
t = 30
xi = 4
ss = [20, 40, 60, 80, 100, 120, 140, 160]
costs = [5, 10, 15, 20, 25]
epoch = 1

lin_costs = [Linear(0.00015, 0), Linear(0.0005, 0), Linear(0.001, 0),Linear(0.0016, 0), Linear(0.003, 0)]
quad_costs = [Quaratic(0.000001, 0, 0), Quaratic(0.000002, 0, 0), Quaratic(0.000003, 0, 0), Quaratic(0.000004, 0, 0), Quaratic(0.000008, 0, 0)]

if __name__=="__main__":
    errors = []
    if mode == 0:
        for s in tqdm(ss):
            l_svs = SVS(s, g=lin, eps=None, cost=20)
            q_svs = SVS(s, g=quad, eps=None, cost=20)
            efd = eFD(s, eps=None, cost=20)
            rs = RS(s, eps=None, cost=20)
            error = [0 for i in range(4)]
            for i in range(epoch):
                A, A_hat = generate_matrix(1000*s, 500, t, xi)
                error[0] += sketch_error(l_svs.run(A_hat), A_hat) / epoch
                error[1] += sketch_error(q_svs.run(A_hat), A_hat) / epoch
                error[2] += sketch_error(efd.run(A_hat), A_hat) / epoch
                error[3] += sketch_error(rs.run(A_hat), A_hat) / epoch

            errors.append(error)

        errors = np.array(errors)
        plt.plot(ss, errors[:, 0], marker='s')
        plt.plot(ss, errors[:, 1], marker='^')
        plt.plot(ss, errors[:, 2], marker='o')
        plt.plot(ss, errors[:, 3], marker='*')

        plt.legend(["l_svs", "q_svs", "efd", "rs"])
        plt.show()

    elif mode == 1:
        for k, cost in tqdm(enumerate(costs)):
            l_svs = SVS(128, g=lin_costs[k], eps=None, cost=cost)
            q_svs = SVS(128, g=quad_costs[k], eps=None, cost=cost)
            efd = eFD(128, eps=None, cost=cost)
            rs = RS(128, eps=None, cost=cost)
            error = [0 for i in range(4)]
            for i in range(epoch):
                A, A_hat = generate_matrix(1000*128, 500, t, xi)
                error[0] += sketch_error(l_svs.run(A_hat), A_hat) / epoch
                error[1] += sketch_error(q_svs.run(A_hat), A_hat) / epoch
                error[2] += sketch_error(efd.run(A_hat), A_hat) / epoch
                error[3] += sketch_error(rs.run(A_hat), A_hat) / epoch

            errors.append(error)

        errors = np.array(errors)
        plt.plot(costs, errors[:, 0], marker='s')
        plt.plot(costs, errors[:, 1], marker='^')
        plt.plot(costs, errors[:, 2], marker='o')
        plt.plot(costs, errors[:, 3], marker='*')

        plt.legend(["l_svs", "q_svs", "efd", "rs"])
        plt.show()

    else:
        raise Exception("mode error")