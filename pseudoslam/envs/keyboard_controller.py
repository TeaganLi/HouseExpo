import numpy as np
from matplotlib import pyplot as plt
import yaml
from os import path

from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam

import cv2

import sys, select, termios, tty, os

moveBindings = {'i':(1,0),
                'j':(0,1),
                'l':(0,-1),
                'k':(-1,0)}

def getKey():
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    i,o,e=select.select([sys.stdin], [], [], 1)
    # if i:
    #     print('')
    key = sys.stdin.read(1)
    # else:
    #     return None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def getMotion():

    key = getKey()
    if key in moveBindings.keys():
        robot_x = moveBindings[key][0]
        th = moveBindings[key][1]
        return robot_x,th
    if (key == '\x03') or (key=='`'):
            return None,None
    else:
        return 0,0

move= {1:'forward', 2:'left', 3:'right'}

def main():
    config_file= path.join(path.dirname(__file__), "config.yaml")
    sim= pseudoSlam(config_file)
    
    print("Start simulation.\nThe configuration file locates at {}".format(config_file))
    print("========================================")
    print("Configuration:")
    print("-------------------------------------")
    print("|Robot Radius        |{} pixels      |".format(sim.robotRadius))
    print("|Forward Step Length |{} pixels      |".format(sim.stepLength_linear))
    print("|Rotation Step Length|{:.2} rad      |".format(sim.stepLength_angular))
    print("|Laser Range         |{} pixels    |".format(sim.laser_range))
    print("|Laser FoV           |{:.3} rad      |".format(sim.laser_fov))
    print("|Map Resolution      |{} pixel/meter|".format(sim.m2p))
    print("|Laser Noise         |{}           |".format(sim.laser_noiseSigma))
    print("|SLAM Error          |{}, {:.3}     |".format(sim.slamErr_linear, sim.slamErr_angular))
    print("|Obstacle Number     |{}             |".format(sim.obs_num))
    print("-------------------------------------")
    print("Action command: ")
    print("-------------------------------------")
    print("i: move forward.")
    print("j: rotate clockwise.")
    print("l: rotate anti-clockwise.")
    print("Esc: stop simulation and quit.")
    print("-------------------------------------")

    i=0
    while 1:
        slamMap= sim.get_state()
        pose = sim.get_pose()

        plt.clf()
        plt.imshow(slamMap,cmap="gray")
        plt.draw()
        plt.pause(0.001)

        v,w= getMotion()
        if v==1:
            sim.moveRobot("forward")
            motion= 1
        elif w==1:
            sim.moveRobot("left")
            motion= 2
        elif w==-1:
            sim.moveRobot("right")
            motion= 3
        else:
            print("Terminate the simulation and quit.")
            break
        i=i+1

    return


if __name__=='__main__':
    main()