# HouseExpo Dataset & PseudoSLAM Simulator
![samples](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/collection.png)

## Overview
HouseExpo is a large-scale dataset of indoor layout built on [SUNCG dataset](http://suncg.cs.princeton.edu/#). The dataset contains 35,357 2D floor plans with 252,550 rooms in total, together with the category label of each room.

PseudoSLAM is a high-speed OpenAI Gym-compatible simulation platform that simulates SLAM and the navigation process in an unknown 2D environment. It reads data from HouseExpo, creates the corresponding 2D environment and generates a mobile robot to carry on different tasks in this environment. 

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
- Then you can try the simulator using keyboard by running 
  ```
  python pseudoslam/envs/keyboard_controller.py
  ```
  You can control the robot by pressing i (move forward), j (rotate clockwise) and l (rotate anti-clockwise).

  
## Training Models
- One Gym-compatible environment of robot exploration has been implemented. To train a model using OpenAI baselines, first add our environment in Gym by registering in `/gym/envs/__init__.py`:
  ```
  register(
      id='RobotExploration-v0',
      entry_point='pseudoslam.envs.robot_exploration_v0:RobotExplorationT0',
      max_episode_steps=100,
  )
  ```
- Start to train the model using baselines, for example, run
  ```
  python -m baselines.run --alg=ppo2 --env=RobotExploration-v0 --network=cnn --num_timesteps=2e7
  ```
- Check your trained model by running
  ```
  python -m baselines.run --alg=ppo2 --env=RobotExploration-v0 --num_timesteps=0 --play
  ```
Please refer to https://github.com/openai/baselines for more detailed introduction on how to use baselines.

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
There are 35,357 2D floor plans with 252,550 rooms, with mean of 7.14 and median of 7.0 rooms per house. The distribution of rooms is
![Room number distribution](http://www.ee.cuhk.edu.hk/~tgli/TingguangLi_files/room_label_dist.png)

Feel free to download all floor plans as .png images at:
https://drive.google.com/file/d/12CU51Fhi-_WubsPUsq5KCijfucW5HG7G/view?usp=sharing

## Usage of PseudoSLAM
### Simulator Parameters
The parameters of the simulator are specified in `./pseudoslam/envs/config.yaml` including
* **json_dir**: path to json files
* **map_id_set**: path to map_id_set. PseudoSLAM supports to utilize a subset of the dataset to train a model. map_id_set is a .txt file recording all map ids you want to train/test on.
* **meter2pixel**: the ratio from meter to pixel, i.e. 1 meter in real world corresponds to n pixels in simulator. Note that a larger value will present a more precise simulation, but with a low simulation speed.
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

## Citation
If you use our HouseExpo dataset or pseudoSLAM simulator, you can cite us with
```
@article{"",
  title={},
  author={},
  journal={},
  year={}
}
```
