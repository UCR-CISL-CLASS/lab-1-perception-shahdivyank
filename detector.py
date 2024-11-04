from mmdet3d.apis import init_model, inference_detector
import numpy as np

class Detector:
    def __init__(self):
        config_file = 'mmdet3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        checkpoint_file = 'mmdet3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        self.model = init_model(config_file, checkpoint_file)

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the detector. The location is defined with respect to the actor center
        -- x axis is longitudinal (forward-backward)
        -- y axis is lateral (left and right)
        -- z axis is vertical
        Unit is in meters

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 
                      'rotation_frequency': 20, 'channels': 64,
                      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                      'id': 'LIDAR'},

            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]

        return sensors

    def detect(self, sensor_data):
        """
        Add your detection logic here
            Input: sensor_data, a dictionary containing all sensor data. Key: sensor id. Value: tuple of frame id and data. For example
                'Right' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'Left' : (frame_id, numpy.ndarray)
                    The RGBA image, shape (H, W, 4)
                'LIDAR' : (frame_id, numpy.ndarray)
                    The lidar data, shape (N, 4)
            Output: a dictionary of detected objects in global coordinates
                det_boxes : numpy.ndarray
                    The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
                det_class : numpy.ndarray
                    The object class for each predicted bounding box, shape (N, 1) corresponding to the above bounding box. 
                    0 for vehicle, 1 for pedestrian, 2 for cyclist.
                det_score : numpy.ndarray
                    The confidence score for each predicted bounding box, shape (N, 1) corresponding to the above bounding box.
        """

        _, lidar_array = sensor_data["LIDAR"]

        predictions, _ = inference_detector(self.model, lidar_array)

        instances = predictions.pred_instances_3d

        labels = instances.labels_3d.cpu().numpy().reshape(-1, 1)
        scores = instances.scores_3d.cpu().numpy().reshape(-1, 1)

        boxes = instances.bboxes_3d

        print(type(boxes))

        total_objects = boxes.shape[0]

        objects = np.zeros((total_objects, 8, 3))

        for object in range(total_objects):

            x, y, z, x_size, y_size, z_size, yaw = boxes[object].cpu().numpy().flatten()

            objects[object] = [
                [x, y, z],          [x, y + y_size, z],             [x + x_size, y + y_size, z],            [x + x_size, y, z],
                [x, y, z + z_size], [x, y + y_size, z + z_size],    [x + x_size, y + y_size, z + z_size],   [x + x_size, y, z + z_size]
            ]

        print("GRRRRR", objects)

        print("RESULTS", boxes.shape, type(labels))
        print("INSTANCE", boxes[0], boxes[0][0] if len(boxes[0]) > 0 else "nothing")

        print("CLASSS", labels)
        print("SCORES", scores)


        return {
        #    "det_class": labels.cpu().numpy(),
        #    "det_score": scores.cpu().numpy()
        }

    