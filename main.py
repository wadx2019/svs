import numpy as np
import matplotlib.pyplot as plt
from utils import *
from algo import *
from metrics import *
from sample import *
from tqdm import tqdm

# Settings

mode = 0 # 0 for performance; 1 for communication cost
t = 30
xi = 4
ss = [20, 40, 60, 80, 100, 120, 140, 160]
costs = [5, 10, 15, 20, 25]
epoch = 5


if __name__=="__main__":
    errors = []
    if mode == 0:
        for s in tqdm(ss):
            l_svs = SVS(s, g=iden, eps=0.01, cost=20)
            q_svs = SVS(s, g=quad, eps=0.01, cost=20)
            efd = eFD(s, eps=0.01, cost=20)
            rs = RS(s, eps=0.01, cost=20)
            error = [0 for i in range(4)]
            for i in range(epoch):
                A, A_hat = generate_matrix(t, xi)
                error[0] += sketch_error(l_svs.run(A_hat), A) / epoch
                error[1] += sketch_error(q_svs.run(A_hat), A) / epoch
                error[2] += sketch_error(efd.run(A_hat), A) / epoch
                error[3] += sketch_error(rs.run(A_hat), A) / epoch

            errors.append(error)

        errors = np.array(errors)
        plt.plot(ss, errors[:, 0], marker='s')
        plt.plot(ss, errors[:, 1], marker='s')
        plt.plot(ss, errors[:, 2], marker='s')
        plt.plot(ss, errors[:, 3], marker='s')

        plt.legend(["l_svs", "q_svs", "efd", "rs"])
        plt.show()

    elif mode == 1:
        for cost in tqdm(costs):
            l_svs = SVS(128, g=iden, eps=None, cost=cost)
            q_svs = SVS(128, g=quad, eps=0.001, cost=cost)
            efd = eFD(128, cost=cost)
            rs = RS(128, cost=cost)
            error = [0 for i in range(4)]
            for i in range(epoch):
                A, A_hat = generate_matrix(t, xi)
                error[0] += sketch_error(l_svs.run(A_hat), A) / epoch
                error[1] += sketch_error(q_svs.run(A_hat), A) / epoch
                error[2] += sketch_error(efd.run(A_hat), A) / epoch
                error[3] += sketch_error(rs.run(A_hat), A) / epoch

            errors.append(error)

        errors = np.array(errors)
        plt.plot(costs, errors[:, 0], marker='s')
        plt.plot(costs, errors[:, 1], marker='s')
        plt.plot(costs, errors[:, 2], marker='s')
        plt.plot(costs, errors[:, 3], marker='s')

        plt.legend(["l_svs", "q_svs", "efd", "rs"])
        plt.show()

    else:
        raise Exception("mode error")