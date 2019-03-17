import os
import json
import cv2
import argparse
import numpy as np

meter2pixel = 100
border_pad = 25

def draw_map(file_name, json_path, save_path):
    print("Processing ", file_name)

    with open(json_path + '/' + file_name + '.json') as json_file:
        json_data = json.load(json_file)

    # Draw the contour
    verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
    x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
    cnt_map = np.zeros((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2))

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad
    cv2.drawContours(cnt_map, [verts], 0, 255, -1)

    # Save map
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_path + "/" + file_name + '.png', cnt_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the subset of maps in .png.")
    parser.add_argument("map_id_set_file", help="map id set (.txt)")
    parser.add_argument("--json_path", type=str, default="./HouseExpo/json/", help="json file path")
    parser.add_argument("--save_path", type=str, default='./png')
    result = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.getcwd(), result.json_path))
    map_file = os.path.abspath(os.path.join(os.getcwd(), result.map_id_set_file))
    save_path = os.path.abspath(os.path.join(os.getcwd(), result.save_path))
    print("---------------------------------------------------------------------")
    print("|map id set file path        |{}".format(map_file))
    print("---------------------------------------------------------------------")
    print("|json file path              |{}".format(json_path))
    print("---------------------------------------------------------------------")
    print("|Save path                   | {}".format(save_path))
    print("---------------------------------------------------------------------")

    map_ids = np.loadtxt(map_file, str)

    for map_id in map_ids:
        draw_map(map_id, json_path, save_path)

    print("Successfully draw the maps into {}.".format(save_path))
