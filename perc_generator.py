from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import math
import numpy as np
import random

MAX_DIMENSION_SIZE = 10
np.random.seed(12342)


@dataclass
class Properties:
    radius: float = 5
    num_samples: int = 1000   # number of points
    num_features: int = 2    # dimensions
    center: List[float] = None
    scale_coefs: List[float] = None


class Set:
    def __init__(self, properties: Properties):
        self.props = properties
        self.max_cords = [-1e6] * self.props.num_features
        self.min_cords = [1e6] * self.props.num_features
        self.points = np.ndarray(shape=(self.props.num_samples, self.props.num_features), dtype=float)
        if self.props.scale_coefs is None or not len(self.props.scale_coefs) == self.props.num_features:
            self.props.scale_coefs = [1] * self.props.num_features

        if self.props.center is None or not len(self.props.center) == self.props.num_features:
            self.props.center = [0] * self.props.num_features

    def create_points(self):
        for point_id in range(self.props.num_samples):
            vector = np.random.uniform(-1, 1, self.props.num_features)
            radius = np.random.uniform(0, self.props.radius)
            norm = math.sqrt(sum([el * el for el in vector]))
            self.points[point_id] = [cord / norm * radius + self.props.center[ix] for ix, cord in enumerate(vector)]
            self.points[point_id] *= self.props.scale_coefs
            self.max_cords = [max(self.max_cords[i], self.points[point_id][i]) for i in range(self.props.num_features)]
            self.min_cords = [min(self.min_cords[i], self.points[point_id][i]) for i in range(self.props.num_features)]


class World:
    def __init__(self, prop_list: List[Properties]):
        self.prop_list = prop_list
        self.sets = []
        self.max_cords = [-1e6] * MAX_DIMENSION_SIZE
        self.min_cords = [1e6] * MAX_DIMENSION_SIZE

    def create_sets(self):
        for prop in self.prop_list:
            figure = Set(prop)
            figure.create_points()
            self.sets.append(figure)
            for i in range(MAX_DIMENSION_SIZE):
                if i >= figure.props.num_features:
                    break
                self.max_cords[i] = max(self.max_cords[i], figure.max_cords[i])
                self.min_cords[i] = min(self.min_cords[i], figure.min_cords[i])

    def plot_2d(self, cord_x=0, cord_y=1):
        plt.figure(figsize=(7, 7))
        plt.axis([self.min_cords[cord_x] - 5, self.max_cords[cord_x] + 5, self.min_cords[cord_y] - 5, self.max_cords[cord_y] + 5])

        for s in self.sets:
            x = s.points[:, cord_x] if s.points.shape[1] >= cord_x + 1 else np.zeros(s.points.shape[0])
            y = s.points[:, cord_y] if s.points.shape[1] >= cord_y + 1 else np.zeros(s.points.shape[0])
            plt.scatter(x, y, 20)
        plt.show()

    def save_csv(self, csv_file):
        if len(self.sets) == 0:
            return

        point_arrays = []
        for ix, s in enumerate(self.sets):
            label_column = np.ndarray(shape=(s.points.shape[0], 1))
            label_column.fill(ix)
            points = np.append(s.points, label_column, axis=1)
            point_arrays.append(points)

        merged_array = np.concatenate(point_arrays, axis=0)
        np.savetxt(csv_file, X=merged_array, fmt="%.4f", delimiter=",")

    def get_data_all(self):
        point_arrays = []
        ans_arrays = []
        for ix, s in enumerate(self.sets):
            res = [0.] * len(self.sets)
            res[ix] = 1.
            for pt in s.points:
                point_arrays.append(pt)
                ans_arrays.append(res)
        return point_arrays, ans_arrays


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def check_min(lt, pt):
    if len(lt) == 0:
        return 0
    ans = dist(lt[0], pt)
    for p in lt:
        ans = min(ans, dist(p, pt))
    return ans

def generate_props(num_sets, size=100, min_dist=15, max_dist=35, need_print=False):
    num_samples_set = (50, 60, 70, 80, 90, 100)
    num_features = 2
    props = []
    centers = []

    for i in range(num_sets):
        num_samples = num_samples_set[random.randint(0, len(num_samples_set) - 1)] * 5
        center = [random.randint(-size, size) for _ in range(num_features)]
        dt = check_min(centers, center)
        while dt > min_dist or dt < max_dist:
            center = [random.randint(-size, size) for _ in range(num_features)]
            dt = check_min(centers, center)
        print (dt)
        print (center)

        centers.append(center)
        scale_coefs = [random.randint(1, 1) for _ in range(num_features)]
        print ("new_pt")

        props.append(Properties(radius=random.randint(25, 25),
                                num_samples=num_samples,
                                num_features=num_features,
                                center=center,
                                scale_coefs=scale_coefs))
    world = World(props)
    world.create_sets()
    if need_print:
        print (centers)
        world.plot_2d()
    return world.get_data_all()


if __name__ == "__main__":
    a, b = generate_props(5, need_print=True)
