# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from cv2 import findHomography

def build_M(worldPoints, planePoints):
    points_size = len(worldPoints)
    m = []
    for i in range(points_size):
        BIGS = worldPoints[i]
        smalls = planePoints[i]
        ax = [-BIGS[0],-BIGS[1],-BIGS[2],-1,0,0,0,0,
              smalls[0]*BIGS[0],smalls[0]*BIGS[1],
              smalls[0]*BIGS[2],smalls[0]]
        m.append(ax)
        ay = [0, 0, 0, 0,-BIGS[0], -BIGS[1], -BIGS[2], -1,
              smalls[1] * BIGS[1], smalls[1] * BIGS[1],
              smalls[1] * BIGS[2], smalls[1]]
        m.append(ay)
    return np.array(m)

def DLT(worldPoints, planePoints):
    M = build_M(worldPoints, planePoints)
    u,sig,VT = np.linalg.svd(M)
    p = VT[-1].reshape(3,4)
    H = p[:3, :3]
    h = p[:,3:]
    H_inv = np.linalg.inv(H)
    x_0 = -(H_inv @ h)
    Q,R = np.linalg.qr(H_inv)
    rotate = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    rotation_matrix = rotate @ Q.T
    return x_0 ,rotation_matrix








if __name__ == '__main__':
    im1 = np.array(
        [[430,521,1], [763,708, 1], [664,408, 1], [967,551, 1],[367,86, 1], [758,184, 1], [627,34, 1], [995,98, 1]])
    real = np.array([[0,0,0,1],[1,0,0,1],[0,0.7,0,1],[1,0.7,0,1],[0,0,0,1],[1,0,1,1],[0,0.7,1,1],[1,0.7,1,1]])
    translation,rotation = DLT(worldPoints=real,planePoints=im1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
