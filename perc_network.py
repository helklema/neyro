import numpy as np
import math


class matr:
    def __init__(self, n, m, def_val = None):
        np.random.seed(1234)
        self.matr = np.array([ [float(def_val)] * m if def_val is not None else np.random.uniform(-1, 1, m) for _ in range(n + 1)])

    def get(self, i, j):
        return self.matr[i][j]

    def update(self, i, j, val):
        self.matr[i][j] += val

    def get_fict(self, j):
        return self.fict[j]

class graph:
    DIVIDE_COEF = 150.
    SPEED = 0.1

    def __init__(self, vert_lt):
        self.vert_lt = vert_lt
        self.edges = [matr(vert_lt[i], vert_lt[i+1]) for i in range(len(vert_lt) - 1)]

    def activate(self, val):
        return 1.0 / (1.0 + math.exp(-2 * 1 * val))

    def deactivate(self, val):
        return self.activate(val) * (1 - self.activate(val))

    def go_forward(self, v, step):
        for to in range(self.vert_lt[step + 1]):
            neuron = self.neurons[step][v] if step == 0 else self.activate(self.neurons[step][v]) if v != self.vert_lt[step] else 1
            self.neurons[step + 1][to] += neuron * self.edges[step].get(v, to)

    def forward(self, _cords):
        self.cords = [i / self.DIVIDE_COEF for i in _cords]
        self.neurons = [ [0] * i + [1] for i in self.vert_lt]
        self.neurons[0] = self.cords + [1]
        self.neurons[-1].pop()
        for step in range(len(self.vert_lt) - 1):
            for v in range(self.vert_lt[step] + 1):
                self.go_forward(v, step)

        last_res = []
        for val in self.neurons[-1]:
            last_res.append(self.activate(val))
        return last_res

    def go_back(self, v, step):
        for to in range(self.vert_lt[step - 1]):
            neuron = self.neurons[step - 1][to] if step - 1 == 0 else self.deactivate(self.neurons[step - 1][to])
            self.bneurons[step - 1][to] += self.bneurons[step][v] * self.edges[step - 1].get(to, v) * neuron

    def back(self, _res):
        self.bneurons = [[0] * i for i in self.vert_lt]
        self.bneurons[-1] = [self.deactivate(self.neurons[-1][i]) * (_res[i] - self.activate(self.neurons[-1][i])) for i in range(self.vert_lt[-1])]

        for step in range(len(self.vert_lt) - 1, 0, -1):
            for v in range(self.vert_lt[step]):
                self.go_back(v, step)

        return self.neurons[len(self.vert_lt) - 1]

    def update_edges(self, fast_update=False):
        for step in range(len(self.vert_lt) - 1):
            for v in range(self.vert_lt[step] + 1):
                for to in range(self.vert_lt[step + 1]):
                    neuron = self.neurons[step][v] if step == 0 else self.activate(self.neurons[step][v]) if v != self.vert_lt[step] else 1
                    if fast_update:
                        self.edges[step].update(v, to, self.SPEED * self.bneurons[step + 1][to] * neuron)
                    else:
                        self.weight[step].update(v, to, self.SPEED * self.bneurons[step + 1][to] * neuron)

    def apply_weight(self):
        for step in range(len(self.vert_lt) - 1):
            for v in range(self.vert_lt[step] + 1):
                for to in range(self.vert_lt[step + 1]):
                    self.edges[step].update(v, to, self.weight[step].get(v, to))

    def get_result(self, _res):
        last_res = []
        for val in self.neurons[-1]:
            last_res.append(self.activate(val))
        return last_res

    def get_mistake(self, points, _res):
        cnt_tp = 0
        for i in range(len(points)):
            self.forward(points[i])
            nm, mx = 0, -11
            for n, p in enumerate(self.neurons[-1]):
                if p > mx:
                    mx, nm = p, n
            if _res[i][nm] == 1:
                cnt_tp += 1
        return cnt_tp / len(_res)


    def era(self, points, _res, debug=False):
        self.res = _res
        self.weight = [matr(self.vert_lt[i], self.vert_lt[i+1], 0) for i in range(len(self.vert_lt) - 1)]
        for i in range(len(points)):
            self.forward(points[i])
            self.back(self.res[i])
            self.update_edges(True)
            if debug:
                self.get_result(self.res[i])

        #self.apply_weight()
