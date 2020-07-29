# HouseExpo Dataset & PseudoSLAM Simulator (IROS2020)
![samples](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/collection.png)
by [Tingguang Li](http://www.ee.cuhk.edu.hk/~tgli/) at Robotics, Perception, and AI Laboratory, The Chinese University of Hong Kong. The paper and video can be found at [Paper](https://arxiv.org/abs/1903.09845), [Video](https://youtu.be/v7XPzj62OfE).

If you think our work is useful, please consider citing use with
```
@article{"li2019houseexpo",
  title={HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms on Mobile Robots},
  author={Tingguang, Li and Danny, Ho and Chenming, Li and Delong, Zhu and Chaoqun, Wang and Max Q.-H. Meng},
  journal={arXiv preprint arXiv:1903.09845},
  year={2019}
}
```

## Overview
HouseExpo is a large-scale dataset of indoor layout built on [SUNCG dataset](http://suncg.cs.princeton.edu/#). The dataset contains 35,126 2D floor plans with 252,550 rooms in total, together with the category label of each room. Check out all floor plans as .png images at:
https://drive.google.com/file/d/1gEmTdgZD1pIa8UtaLXz8vm_301zL4L1J/view?usp=sharing.

PseudoSLAM is a high-speed OpenAI Gym-compatible simulation platform that simulates SLAM and the navigation process in an unknown 2D environment. It reads data from HouseExpo, creates the corresponding 2D environment and generates a mobile robot to carry on different tasks in this environment. 

## Prerequisite
The code has been tested under 
* python 3.6
* tensorflow 1.15
* Ubuntu 16.04

## Getting Started
- Clone the repo and cd into it:
  ```
  git clone https://github.com/TeaganLi/HouseExpo.git
  cd HouseExpo
  ```
- Install pseudoSLAM package
  ```
  pip install -e .
  ```
- Uncompress HouseExpo data:
  ```
  cd HouseExpo
  tar -xvzf json.tar.gz
  ```
- There are two demos for you. You can try the exploration demo using keyboard by running 
  ```
  python pseudoslam/envs/keyboard_exploration.py
  ```
  You will see an exploration (partially known/unknown) environment (left figure). Activate the terminal and then you can control the robot by pressing w (move forward), a (rotate clockwise) and d (rotate anti-clockwise). 
- You can try the navigation demo using keyboard by running
  ```
  python pseudoslam/envs/keyboard_navigation.py
  ```
  You will see a navigation (fully known) environment. Activate the terminal and you can save the traversed trajectory by pressing s.
  ![demo](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/simulator_demo.png)

- To train your model, you can check the observation and adjust your configuration file by running the following code. You will see a global map and a local observation.
  ```
  python pseudoslam/envs/robot_exploration_v0.py
  ```
  
## Training Models
- One Gym-compatible environment of robot exploration has been implemented. To train a model using OpenAI baselines, first install OpenAI Gym and Baselines. Then add our environment in Gym by registering in `/gym/envs/__init__.py`:
  ```
  register(
      id='RobotExploration-v0',
      entry_point='pseudoslam.envs.robot_exploration_v0:RobotExplorationT0',
      max_episode_steps=200,
  )
  ```
- Start to train the model using baselines, for example, run
  ```
  python -m baselines.run --alg=ppo2 --env=RobotExploration-v0 --network=cnn --num_timesteps=2e7 --save_path=~/models/exploration_20M_ppo2 --save_interval=10 --log_path=~/logs/exploration --save_video_interval=10000
  ```
- After the training is finished, check your trained model by running
  ```
  python -m baselines.run --alg=ppo2 --env=RobotExploration-v0 --num_timesteps=0 --load_path=~/models/exploration_20M_ppo2 --play
  ```
Please refer to https://github.com/openai/baselines for more detailed introduction on how to use baselines, like launching multiple environments.

## Details of HouseExpo Dataset
### Data Format
The floor plans are stored in `./HouseExpo/json/` in the form of .json files. The data format is as follows
* id (string): the unique house ID number.
* room_num (int): the number of rooms of this house.
* bbox (dict): bounding box of the whole house
   * "min": (x1, y1)
   * "max": (x2, y2)
* verts (list): each element (x, y) represents a vertex location (in meter).
* room_category (dict): the room categories and its bounding box, for example
   * "kitchen": (x1, y1, x2, y2), bounding box of each kitchen.

### Map Visualization
First randomly sample a subset of maps by running
```
python pseudoslam/viz/map_id_set_generator.py --num 100
```
Then visualize the sampled maps as images 
```
python pseudoslam/viz/vis_maps.py PATH_TO_MAPID_FILE
```
   
### Room Category
HouseExpo inherits room category labels from SUNCG dataset and  also provides a flexible way to define your own room type labels (as defined in `pseudoslam/envs/simulator/jsonreader.py`). Some samples are (different colors represent different room categories)

![room_label](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/room_label.png) 

### Statistics
There are 35,126 2D floor plans with 252,550 rooms, with mean of 7.14 and median of 7.0 rooms per house. The distribution of rooms is
![Room number distribution](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/room_label_dist.png)

Feel free to download all floor plans as .png images at:
https://drive.google.com/file/d/1gEmTdgZD1pIa8UtaLXz8vm_301zL4L1J/view?usp=sharing

## Usage of PseudoSLAM
### Simulator Parameters
The parameters of the simulator are specified in `./pseudoslam/envs/config.yaml` including
* **json_dir**: path to json files
* **map_id_set**: path to map_id_set. PseudoSLAM supports to utilize a subset of the dataset to train a model. map_id_set is a .txt file recording all map ids you want to train/test on.
* **meter2pixel**: the ratio from meter to pixel, i.e. 1 meter in real world corresponds to n pixels in simulator. Note that a larger value will present a more precise simulation, but with a low simulation speed.
* **mode**: 0 for exploration (unknown environment), 1 for navigation (fully known environment)
* **obstacle**
  * **number**: number of obstacles. The obstacle is dynamically generated when initializing the map.
  * **size** [**max**, **min**]: size of obstacle (in meter).
* **robotRadius**: robot radius (in meter) for collision checking.
* **stepLength**:
  * **linear**: robot linear movement in each step (in meter).
  * **angular**: robot angular movement in each step (in degree).
* **startPose** [**x**, **y**, **theta**]: robot start pose (in meter) in world coordinate with (0,0) at the center of the house.
* **resetRandomPose**: flag of whether randomly set robot init pose when reset, if 0, robot reset to startPose.
* **laser**:
  * **range**: laser range in meter.
  * **fov**: laser field of view in degree.
  * **resolution**: laser rotation resolution in degree.
  * **noiseSigma**: sigma of Gaussian distributino of laser noise.
* **slamError**:
  * **linear**: slam linear registration error in pixel.
  * **angular**: slam rotational registration error in degree.
* **stateSize** [**x**, **y**]: state size in meter

## Benchmark Reproduction
The simulation configurations and the training/testing map ids for obstacle avoidance and autonomous exploration are located at ./experiments for reproduction purpose.

## Dynamic Obstacles
We collected a database for dynamic obstacles (moving humans) from the real world and are incorporating this part into our simulator. A demo is like this. Please stay tuned for this part!

![dynamic obstacles](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/dynamic_obstacles.gif)



