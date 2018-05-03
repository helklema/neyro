from network import graph
from generator import generate_props
import math
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(123)

FROM = -120
TO = 120
CNT_EPOCH = 25
## касание
CNT_EPOCH = 60
## пересечение
num_features = 3
CNT_EPOCH = 100
## много точек
#num_features = 20
#CNT_EPOCH = 20
col = [0] * num_features
col = [0] * num_features
col_base = [0] * num_features
conf = [2, 3]

eps = 1e-7

def check_color(res, pt):
    nm, mx = 0, -11
    for n, p in enumerate(res):
        if p > mx:
            mx, nm = p, n
    if col[nm] == 0:
        col[nm] = [[pt[0]], [pt[1]]]
    else:
        col[nm][0].append(pt[0])
        col[nm][1].append(pt[1])

def check_color_base(res, pt):
    nm = 0
    mx = -11
    for n, p in enumerate(res):
        if p > mx:
            mx = p
            nm = n
    if col_base[nm] == 0:
        col_base[nm] = [[pt[0]], [pt[1]]]
    else:
        col_base[nm][0].append(pt[0])
        col_base[nm][1].append(pt[1])

def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def add_points():
    vl = 28
    cnt = [-32, -78]
    for i in range(90):
        x = cnt[0] + random.randint(-vl, vl)
        y = cnt[1] + random.randint(-vl, vl)
        dt = dist([x, y], cnt)
        while dt < 25.5 or dt > 27:
            x = cnt[0] + random.randint(-vl, vl)
            y = cnt[1] + random.randint(-vl, vl)
            dt = dist([x, y], cnt)
        col_base[0][0].append(x)
        col_base[0][1].append(y)

def get_sets():
    #касание
    points, res = generate_props(num_features)
    #не касение
    #points, res = generate_props(num_features, min_dist=40, max_dist = 2000)
    #пересечение
#    points, res = generate_props(num_features, min_dist=0, max_dist = 1000)
    #много точек
    #points, res = generate_props(num_features, min_dist=40, max_dist = 2000)

    vals = [i for i in range(len(res))]
    random.shuffle(vals)
    c_tr = (len(vals) // 100) * 60
    c_val = (len(vals) - c_tr) // 2
    tr_pt, tr_res = [points[vals[i]] for i in range(c_tr)], [res[vals[i]] for i in range(c_tr)]
    tt_pt, tt_res = [points[vals[i]] for i in range(c_tr, c_tr + c_val)], [res[vals[i]] for i in range(c_tr, c_tr + c_val)]
    vl_pt, vl_res = [points[vals[i]] for i in range(c_tr + c_val, len(vals))], [res[vals[i]] for i in range(c_tr + c_val, len(vals))]
    return tr_pt, tr_res, tt_pt, tt_res, vl_pt, vl_res


if __name__ == "__main__":
    g = graph([2, 3, 4, num_features])
    #points = [[0.3, 0.5], [-0.4, -0.7], [-0.5, 0.5]]

    data = []
    for i in range(FROM, TO):
        for j in range(FROM, TO):
            data.append([i, j])

    tr_pt, tr_res, tt_pt, tt_res, vl_pt, vl_res = get_sets()
    tt_mis = []
    vl_mis = []
    print ("Generate")

    for i in range(CNT_EPOCH):
        g.era(tr_pt, tr_res)
        tt_mis.append(1 - g.get_mistake(tt_pt, tt_res))
        vl_mis.append(1 - g.get_mistake(vl_pt, vl_res))
        print("epochN {0} mist {1}".format(i, vl_mis[-1]))
        if vl_mis[-1] < eps:
            break

    plt.subplot(211)
    plt.plot(range(len(tt_mis)), tt_mis, color='red')
    plt.plot(range(len(vl_mis)), vl_mis, color='blue')
    plt.title("mistakes")
    plt.grid(True)

    for pt in data:
        cl = g.forward(pt)
        check_color(cl, pt)

    print ("Color")
    plt.subplot(212)
    plt.title("Field")

    for pt in col:
        if pt != 0:
            plt.scatter(pt[0], pt[1], s=6)

    for i in range(len(tr_res)):
        check_color_base(tr_res[i], tr_pt[i])
    for i in range(len(vl_res)):
        check_color_base(vl_res[i], vl_pt[i])
    for i in range(len(tt_res)):
        check_color_base(tt_res[i], tt_pt[i])
    add_points()

    for pt in col_base:
        if pt != 0:
            plt.scatter(pt[0], pt[1], s=3)
    plt.show()

    print ("OK")
