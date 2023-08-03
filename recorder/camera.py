#!/usr/bin/python3
import csv
import carla
import cv2 as cv
import numpy as np
import transforms3d
import math

from recorder.sensor import Sensor
from carla_dataset_tools.utils.geometry_types import Transform, Rotation
from carla_dataset_tools.utils.transform import carla_transform_to_transform
import os


class CameraBase(Sensor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 parent,
                 carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None,
                 save_bbox = False,
                 world=None,
                 camera_bp=None,
                 instance_seg=False):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.color_converter = color_converter
        self.save_bbox = save_bbox
        self.world = world

        if save_bbox:
            # Get the attributes from the camera
            image_w = camera_bp.get_attribute("image_size_x").as_int()
            image_h = camera_bp.get_attribute("image_size_y").as_int()
            fov = camera_bp.get_attribute("fov").as_float()

            # Calculate the camera projection matrix to project from 3D -> 2D
            self.K = self.build_projection_matrix(image_w, image_h, fov)



    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        save_dir_annoated = save_dir + "annotated"
        # Convert to target color template
        if self.color_converter is not None:
            sensor_data.convert(self.color_converter)

        # Convert raw data to numpy array, image type is 'bgra8'
        carla_image_data_array = np.ndarray(shape=(sensor_data.height,
                                                   sensor_data.width,
                                                   4),
                                            dtype=np.uint8,
                                            buffer=sensor_data.raw_data)

        # Save image to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].png
        success = cv.imwrite("{}/{:0>10d}.png".format(save_dir,
                                                      sensor_data.frame),
                             carla_image_data_array)

        if success and self.is_first_frame():
            self.save_camera_info(save_dir)
        # print(f"save bbox is {self.save_bbox}")
        if self.save_bbox:
            world_2_camera = np.array(self.carla_actor.get_transform().get_inverse_matrix())
            image_x_min = 0
            image_y_min = 0
            image_x_max = 1216
            image_y_max = 1216
            for npc in self.world.get_actors().filter('*vehicle*'):
                # print(f"len of npc is {len(self.world.get_actors().filter('*vehicle*'))}")
                
                # print(f"self.parent is {self.parent.carla_actor}")
                # print(f"parent id is{self.parent.carla_actor.id}")
                if npc.id != self.parent.carla_actor.id:

                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(self.parent.carla_actor.get_transform().location)

                    # print(f"distance is {dist}")
                    # Filter for the vehicles within 50m
                    if dist < 80:

                    # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the other vehicle. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                        forward_vec = self.parent.carla_actor.get_transform().get_forward_vector()
                        ray = npc.get_transform().location - self.parent.carla_actor.get_transform().location

                        if forward_vec.dot(ray) > 1:
                            # print(f"forward_vec_dot is {forward_vec.dot(ray)}")
                            p1 = self.get_image_point(bb.location, self.K, world_2_camera) #http://host.robots.ox.ac.uk/pascal/VOC/
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            x_max = -10000
                            x_min = 10000
                            y_max = -10000
                            y_min = 10000

                            for vert in verts:
                                p = self.get_image_point(vert, self.K, world_2_camera)
                                # Find the rightmost vertex
                                if p[0] > x_max:
                                    x_max = p[0]
                                # Find the leftmost vertex
                                if p[0] < x_min:
                                    x_min = p[0]
                                # Find the highest vertex
                                if p[1] > y_max:
                                    y_max = p[1]
                                # Find the lowest  vertex
                                if p[1] < y_min:
                                    y_min = p[1]
                            adj_count = 0
                            if x_max > image_x_max:
                                # x_max = image_x_max
                                adj_count += 1
                            if y_max > image_y_max:
                                # y_max = image_y_max
                                adj_count += 1
                            if x_min < image_x_min:
                                # x_min = image_x_min
                                adj_count += 1
                            if y_min < image_y_min:
                                # y_min = image_y_min
                                adj_count += 1
                            if adj_count > 2:
                                continue
                    
                            cv.line(carla_image_data_array, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                            cv.line(carla_image_data_array, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                            cv.line(carla_image_data_array, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                            cv.line(carla_image_data_array, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)


                            # Before saving the image, make sure the directory exists
                            if not os.path.exists(save_dir_annoated):
                                os.makedirs(save_dir_annoated)
                            success = cv.imwrite("{}/{:0>10d}.png".format(save_dir_annoated,
                                                      sensor_data.frame),
                             carla_image_data_array)
                            # print("saved in :", "{}/{:0>10d}.png".format(save_dir_annoated,
                            #                           sensor_data.frame))



        return success
    
    
    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]
    
    
    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    
    def save_camera_info(self, save_dir):
        with open('{}/camera_info.csv'.format(save_dir), 'w', encoding='utf-8') as csv_file:
            fieldnames = {'width',
                          'height',
                          'fx',
                          'fy',
                          'cx',
                          'cy'}
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            camera_info = self.get_camera_info()
            writer.writerow(camera_info)

    
    def get_camera_info(self):
        camera_width = int(self.carla_actor.attributes['image_size_x'])
        camera_height = int(self.carla_actor.attributes['image_size_y'])
        fx = camera_width / (
                2.0 * math.tan(float(self.carla_actor.attributes['fov']) * math.pi / 360.0))
        return {
            'width': camera_width,
            'height': camera_height,
            'cx': camera_width / 2.0,
            'cy': camera_height / 2.0,
            'fx': fx,
            'fy': fx
        }

    def get_transform(self) -> Transform:
        c_trans = self.carla_actor.get_transform()
        trans = carla_transform_to_transform(c_trans)
        quat = trans.rotation.get_quaternion()
        quat_swap = transforms3d.quaternions.mat2quat(np.matrix(
                      [[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]]))
        quat_camera = transforms3d.quaternions.qmult(quat, quat_swap)
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat_camera)
        return Transform(trans.location, Rotation(roll=math.degrees(roll),
                                                  pitch=math.degrees(pitch),
                                                  yaw=math.degrees(yaw)))


class RgbCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None, world=None, camera_bp=None):
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter, save_bbox=True, world=world, camera_bp=camera_bp)

class InstanceSegmentationCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        # color_converter = carla.ColorConverter.CityScapesPalette
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


# class RgbCameraBbox(CameraBase):
#     def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
#                  color_converter: carla.ColorConverter = None):
#         super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)
    

class SemanticSegmentationCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        color_converter = carla.ColorConverter.CityScapesPalette
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class DepthCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        color_converter = carla.ColorConverter.Raw
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)