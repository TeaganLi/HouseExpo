import numpy as np


def transform_coord(y_coordMat, x_coordMat, rotationCenter, transformVect):
    """ Transform x-y coordinate (y_mat & x_mat) by transformVect | round to int | return rotated y & x coord as vector"""
    """ y_mat and x_mat are the coord to be rotated | rotationCenter [y;x] or [y;x;phi] are the centre of rotation by theta
      transformVect [y;x;theta]: y & x are relative to rotationCenter if center [y;x], or relative to world ref frame if center [y;x;phi],
      theta is the angle in rad which the coord to be rotated """

    y_rc= rotationCenter[0]
    x_rc= rotationCenter[1]

    y_translate= transformVect[0]
    x_translate= transformVect[1]
    # change transform to be relative to rotationCenter frame if in form of [y;x;phi]
    if rotationCenter.shape[0]>2:
        y_translate= y_translate*np.cos(rotationCenter[2]) + x_translate*np.sin(rotationCenter[2])
        x_translate= x_translate*np.cos(rotationCenter[2]) - y_translate*np.sin(rotationCenter[2])

    theta= transformVect[2]
    sthe = np.sin(theta)
    cthe = np.cos(theta)
    y_rot = sthe*x_coordMat + cthe*y_coordMat + (1-cthe)*y_rc - sthe*x_rc + y_translate
    x_rot = cthe*x_coordMat - sthe*y_coordMat + (1-cthe)*x_rc + sthe*y_rc + x_translate

    y_ind = np.round(y_rot).astype(int).reshape(y_rot.size, 1)
    x_ind = np.round(x_rot).astype(int).reshape(x_rot.size, 1)
    return y_ind, x_ind


def rad2deg(rad):
    return 180.0/np.pi*rad


def deg2rad(deg):
    return np.pi/180*deg


def angle_within_360(theta):
    """ cast angle into range 0 < theta < 360"""
    theta = np.mod(theta, 360)
    if theta > 360:
        theta -= 360
    return theta

def angel_within_pi(theta):
    """ cast angle into range 0 < theta < 2pi"""
    theta= np.mod(theta, 2*np.pi)
    if theta > 2*np.pi:
        theta -= 2*np.pi
    return theta

def meter2pixel(x_in_m, m2p_ratio):
    """ convert world meter into pixel"""
    return np.round(x_in_m*m2p_ratio).astype(int)

def pixel2meter(x_in_pixel, m2p_ratio):
    """ convert pixel in world meter"""
    return x_in_pixel*1.0/m2p_ratio

def world2mapCoord(p_world, worldOrigin, m2p_ratio=1):
    """ convert world coordinate into map coordinate
    world coord: (origin= worldOrigin & y-axis is upward) | map coord: (origin=top-left corner & y-axis is downward)
    worldOrigin: [y,x] in pixel in img coord | p_world: [y,x] in meter in world coord
    return p_map: [y,x] in pixel in img coord """

    p_map_y= worldOrigin[0] - p_world[0]*m2p_ratio
    p_map_x= worldOrigin[1] + p_world[1]*m2p_ratio
    return np.array([p_map_y,p_map_x])

def map2worldCoord(p_map, worldOrigin, m2p_ratio=1):
    """ convert map coordinate into world coordinate
    map coord: (origin=top-left corner & y-axis is downward) | world coord: (origin= worldOrigin & y-axis is upward)
    worldOrigin: [y,x] in pixel in img coord | p_map: [y,x] in pixel in img coord
    return p_world: [y,x] in meter in world coord"""

    p_world_y= (worldOrigin[0] - p_map[0])*1.0/m2p_ratio
    p_world_x= (-worldOrigin[1] + p_map[1])*1.0/m2p_ratio
    return np.array([p_world_y,p_world_x])

def within_bound(p,shape,r=0):
    """ check if point p [y;x] or [y;x;theta] with radius r is inside world of shape (h,w)
    return bool if p is single point | return bool matrix (vector) if p: [y;x] where y & x are matrix (vector) """
    return (p[0] >= r) & (p[0] < shape[0]-r) & (p[1] >= r) & (p[1] < shape[1]-r)

def make_circle(r, pixelValue):
    """ make a patch of circle with pixelValue """
    patch = np.zeros([2*r+1, 2*r+1])
    angles = np.arange(361).reshape(361, 1) * np.pi / 180
    radius = np.linspace(0, r, num=30).reshape(1, 30)

    y_mat = r + np.matmul(np.sin(angles), radius)
    x_mat = r + np.matmul(np.cos(angles), radius)
    y_ind = np.round(y_mat).astype(int).reshape(y_mat.size, 1)
    x_ind = np.round(x_mat).astype(int).reshape(x_mat.size, 1)
    patch[y_ind, x_ind] = pixelValue
    return patch, r


def gauss_noise(mu=0, sigma=0.1):
    """ return value sampled from Gaussian distribution """
    return np.random.normal(mu,sigma)
