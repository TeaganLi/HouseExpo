import json, cv2, os
import numpy as np

ALLOWED_ROOM_TYPES = {'indoor':1, 'kitchen':2, 'dining_room':4, 'living_room':8,
                      'bathroom':16, 'bedroom':32, 'office':64, 'hallway':128}

def _get_room_tp_id(room):
    room = room.lower()
    if room == 'toilet':
        room = 'bathroom'
    elif room == 'guest_room':
        room = 'bedroom'
    if room not in ALLOWED_ROOM_TYPES:
        return ALLOWED_ROOM_TYPES['indoor']
    return ALLOWED_ROOM_TYPES[room]

class jsonReader():
    def __init__(self, json_prefix, meter2pixel):
        self.json_prefix = os.path.join(os.path.dirname(__file__), "../", json_prefix)      # json data dir
        self.meter2pixel = meter2pixel      # meter to pixel. E.g. meter2pixel=100 represents 1 meter
                                            # equals 100 pixels in the map
        self.border_pad = 1                 # border padding for each side

        self.json_data = {}                 # house info read from json file
        self.cnt_map = []                   # contour map, 0-obstacle, 255-free space
        self.tp_map = []                    # room type map

    def read_json(self, file_name):
        """
        Read json file, generate and return contour map (cnt_map) and room type map (tp_map)
        :param file_name: name of a json file, e.g. 12345678.json
        :return: cnt_map, tp_map
        """
        # print("Processing ", file_name)

        with open(self.json_prefix + file_name.split('.')[0] + '.json') as json_file:
            self.json_data = json.load(json_file)

        # Draw the contour
        verts = (np.array(self.json_data['verts']) * self.meter2pixel).astype(np.int)
        x_max, x_min, y_max, y_min = np.max(verts[:,0]), np.min(verts[:,0]), np.max(verts[:, 1]), np.min(verts[:,1])
        self.cnt_map = np.zeros((y_max - y_min + self.border_pad * 2,
                        x_max - x_min + self.border_pad * 2))

        verts[:, 0] = verts[:, 0] - x_min + self.border_pad
        verts[:, 1] = verts[:, 1] - y_min + self.border_pad
        cv2.drawContours(self.cnt_map, [verts], 0, 255, -1)
        # self.display(self.cnt_map)

        # Merge the tps into an allowed subset
        self.tp_map = np.ones_like(self.cnt_map, dtype=np.uint8)
        for tp in self.json_data['room_category']:
            tp_id = _get_room_tp_id(tp)
            for bbox_tp in self.json_data['room_category'][tp]:
                bbox_tp = (np.array(bbox_tp) * self.meter2pixel).astype(np.int)
                bbox = [np.max([bbox_tp[0] - x_min + self.border_pad, 0]),
                        np.max([bbox_tp[1] - y_min + self.border_pad, 0]),
                        np.min([bbox_tp[2] - x_min + self.border_pad, self.cnt_map.shape[1]]),
                        np.min([bbox_tp[3] - y_min + self.border_pad, self.cnt_map.shape[0]])]
                self.tp_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] |= tp_id
        self.tp_map[self.cnt_map==0] = 0
        # self.display(self.tp_map, tp_flag=True)

        return self.cnt_map.copy(), self.tp_map.copy()

    def get_room_tp(self, x, y):
        """
        Get room type, given x,y coordinates in array coor, (row, column).
        :return: list of room types, e.g. ['kitchen', 'dining room', 'indoor']
        """
        assert (type(x) is int) and (type(y) is int), "Input coordinates are not int!!"
        assert (x >= 0) and (x < self.tp_map.shape[0]) and (y >= 0) and (y < self.tp_map.shape[1]), "Input coordinates exceeds map boundary!!"
        if self.tp_map[x, y] == 0:
            return ['obstacle']

        tp = []
        for i in range(len(ALLOWED_ROOM_TYPES)):
            if (self.tp_map[x, y] & np.power(2, i).astype(np.uint8)) != 0:
                tp.append(list(ALLOWED_ROOM_TYPES.keys())[i])

        return tp


if __name__ == '__main__':
    json_prefix = './resource/json/'

    reader = jsonReader(json_prefix)
    files = os.listdir(json_prefix)
    for file in files:
        reader.read_json(file)

