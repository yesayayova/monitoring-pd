from itertools import combinations
import numpy as np

threshold = 200

def calc_distance(points):
    detect = np.zeros(len(points))
    seq = combinations(points, 2)
    for point in list(seq):
        distance = euclid(point[0], point[1])
        if int(distance) < threshold:
            detect[points.index(point[0])] = 1
            detect[points.index(point[1])] = 1
    return detect

def euclid(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(1/2)