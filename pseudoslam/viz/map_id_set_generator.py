import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate map id set by randomly select a subset of the maps.")
    parser.add_argument("--path", type=str, default='../../HouseExpo/json/',
                        help="path to HouseExpo json files")
    parser.add_argument("--num_map", type=int, default=100,
                        help="number of the generated map set")
    parser.add_argument("--save_path", type=str, default='../../HouseExpo/')
    parser.add_argument("--complement_flag", type=bool, default=False,
                        help="if true, generate map set from the complement of existing_set")
    parser.add_argument("--existing_set_path", type=str, default='../../HouseExpo/test_1000.txt',
                        help="the path to existing set, if complement is False, existing path will not be used")
    result = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.path))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.save_path))
    print("---------------------------------------------------------------------")
    print("|json file path        |{}".format(json_path))
    print("---------------------------------------------------------------------")
    print("|Num of target map set | {}".format(result.num_map))
    print("---------------------------------------------------------------------")
    print("|Save path             | {}".format(save_path))
    print("---------------------------------------------------------------------")
    if result.complement_flag:
        exist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.existing_set_path))
        print("|Complement flag       | {}".format(result.complement_flag))
        print("---------------------------------------------------------------------")
        print("|Existing set path     | {}".format(exist_path))
        print("---------------------------------------------------------------------")

    map_ids = os.listdir(json_path)
    map_ids = [map_id.split('.')[0] for map_id in map_ids]

    if result.complement_flag:
        complement_set = np.loadtxt(exist_path, str)
        map_ids = list(set(map_ids) - set(complement_set))
    assert len(map_ids) >= result.num_map, "Total number of maps less than target number!"

    map_ids = np.random.choice(map_ids, result.num_map, replace=False)

    save_path = save_path + "/map_id_" + str(result.num_map) + ".txt"
    np.savetxt(save_path, map_ids, fmt="%s")
    print("Successfully save the map subset ids to {}.".format(save_path))
