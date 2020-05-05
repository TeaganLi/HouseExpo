import numpy as np
from matplotlib import pyplot as plt
import yaml
from os import path

from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam

import cv2

import sys, select, termios, tty, os

moveBindings = {'w':(1,0),
                'a':(0,1),
                'd':(0,-1),
                's':(-1,0)}

def getKey():
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    i,o,e=select.select([sys.stdin], [], [], 1)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def getMotion():
    key = getKey()
    if key in moveBindings.keys():
        robot_x = moveBindings[key][0]
        th = moveBindings[key][1]
        return robot_x,th
    if (key in ['q', 'r']):
            return key, key
    else:
        return 0,0

move= {1:'forward', 2:'left', 3:'right'}

def main():
    config_file= path.join(path.dirname(__file__), "config", "config_exploration.yaml")
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
    print("w: move forward.")
    print("a: rotate clockwise.")
    print("d: rotate anti-clockwise.")
    print("r: reset the environment")
    print("q: stop simulation and quit.")
    print("-------------------------------------")

    epi_num = 0
    fig_map = plt.gcf()
    while 1:
        slamMap= sim.get_state()
        pose = sim.get_pose()

        plt.clf()
        plt.imshow(slamMap,cmap="gray")
        plt.draw()
        plt.pause(0.001)
        # fig_map.savefig("test.png")	# uncomment this if you want save figure

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
        elif w=='q':
            print("Terminate the simulation and quit.")
            break
        elif w=='r':
            print("Reset the environment.")
            sim.reset()
        else:
            continue

    return


if __name__=='__main__':
    main()