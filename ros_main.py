import copy
import os
import time

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import tensorflow as tf
from geometry_msgs.msg import Point, Point32, Quaternion
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from lio_sam.srv import detection, detectionRequest, detectionResponse
from scipy.spatial.transform.rotation import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import config.config as cfg
import lib_cpp
from networks.model import *


def point_cloud(points, parent_frame="velodyne"):
    """ Creates a point cloud message.
    Args:
        points: Nx4 array of xyz positions (m) and intensities (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [
        PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzi')
    ]

    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(header=header,
                       height=1,
                       width=points.shape[0],
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=(itemsize * 4),
                       row_step=(itemsize * 4 * points.shape[0]),
                       data=data)



mnum = 0
marker_array = MarkerArray()
marker_array_text = MarkerArray()

DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']

Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']

Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

overlap = cfg.OVERLAP
HEIGHT = round((X_MAX - X_MIN+2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)



print(HEIGHT, WIDTH, CHANNELS)

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
              )
lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


TYPE_INDICES = {'car': 0, 'bus': 1, 'truck': 2, 'pedestrian': 3, 'bimo': 4}


class Detector(object):
    def __init__(self, *, nms_threshold=0.1, weight_file=None):
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(cfg.GPU_INDEX)):
                input_bev_img_pl = \
                    self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(input_bev_img_pl)

                saver = tf.train.Saver()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, cfg.MODEL_PATH)
                self.ops = {'input_bev_img_pl': input_bev_img_pl,  # input
                            'end_points': end_points,  # output
                            }
        rospy.init_node('livox_detection', anonymous=True)
        rospy.Service('se_ssd', detection, self.LIOSLOTCallback)


    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    def get_3d_box(self, box_size, heading_angle, center):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        R = self.roty(heading_angle)
        l, w, h = box_size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        corners_3d = np.transpose(corners_3d)
        return corners_3d

    def data2voxel(self, pclist):

        data = [i*0 for i in range(HEIGHT*WIDTH*CHANNELS)]

        for line in pclist:
            X = float(line[0])
            Y = float(line[1])
            Z = float(line[2])
            if( Y > Y_MIN and Y < Y_MAX and
                X > X_MIN and X < X_MAX and
                Z > Z_MIN and Z < Z_MAX):
                channel = int((-Z + Z_MAX)/DZ)
                if abs(X)<3 and abs(Y)<3:
                    continue
                if (X > -overlap):
                    pixel_x = int((X - X_MIN + 2*overlap)/DX)
                    pixel_y = int((-Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
                if (X < overlap):
                    pixel_x = int((-X + overlap)/DX)
                    pixel_y = int((Y + Y_MAX)/DY)
                    data[pixel_x*WIDTH*CHANNELS+pixel_y*CHANNELS+channel] = 1
        voxel = np.reshape(data, (HEIGHT, WIDTH, CHANNELS))
        return voxel

    def detect(self, batch_bev_img):
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out,\
            = self.sess.run([self.ops['end_points']['feature_out'],
                             ], feed_dict=feed_dict)
        result = lib_cpp.cal_result(feature_out[0,:,:,:], \
                                    cfg.BOX_THRESHOLD,overlap,X_MIN,HEIGHT, WIDTH, cfg.VOXEL_SIZE[0], cfg.VOXEL_SIZE[1], cfg.VOXEL_SIZE[2], cfg.NMS_THRESHOLD)
        is_obj_list = result[:, 0].tolist()

        reg_m_x_list = result[:, 5].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_z_list = result[:, 8].tolist()
        reg_h_list = result[:, 7].tolist()

        results = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            box3d_pts_3d[:, 0:3] = self.get_3d_box( \
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]), \
                reg_theta_list[i], (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T  # n*4
            if int(obj_cls_list[i]) == 0:
                cls_name = "car"
            elif int(obj_cls_list[i]) == 1:
                cls_name = "bus"
            elif int(obj_cls_list[i]) == 2:
                cls_name = "truck"
            elif int(obj_cls_list[i]) == 3:
                cls_name = "pedestrian"
            else:
                cls_name = "bimo"
            results.append([cls_name,
                            box3d_pts_3d[0][0], box3d_pts_3d[1][0], box3d_pts_3d[2][0], box3d_pts_3d[3][0],
                            box3d_pts_3d[4][0], box3d_pts_3d[5][0], box3d_pts_3d[6][0], box3d_pts_3d[7][0],
                            box3d_pts_3d[0][1], box3d_pts_3d[1][1], box3d_pts_3d[2][1], box3d_pts_3d[3][1],
                            box3d_pts_3d[4][1], box3d_pts_3d[5][1], box3d_pts_3d[6][1], box3d_pts_3d[7][1],
                            box3d_pts_3d[0][2], box3d_pts_3d[1][2], box3d_pts_3d[2][2], box3d_pts_3d[3][2],
                            box3d_pts_3d[4][2], box3d_pts_3d[5][2], box3d_pts_3d[6][2], box3d_pts_3d[7][2],
                            is_obj_list[i], (reg_m_y_list[i], -reg_m_x_list[i], -reg_m_z_list[i]),
                            (reg_l_list[i], reg_w_list[i], reg_h_list[i]), reg_theta_list[i]])
        return results

    def LIOSLOTCallback(self, request: detectionRequest):
        header = request.cloud.header

        response = detectionResponse()
        response.detections = BoundingBoxArray()
        response.detections.header = header

        points_list = []
        for point in pcl2.read_points(request.cloud, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            if np.abs(point[0]) < 2.0 and np.abs(point[1]) < 1.5:
                continue
            points_list.append(point)

        if len(points_list) == 0:
            return response

        points_list = np.asarray(points_list)
        vox = self.data2voxel(points_list)
        vox = np.expand_dims(vox, axis=0)
        t0 = time.time()
        result = self.detect(vox)
        t1 = time.time()
        print('det_time(ms): {}; det_numbers: {}'.format(1000*(t1-t0), len(result)))
        for ii in range(len(result)):
            result[ii][1:9] = list(np.array(result[ii][1:9]))
            result[ii][9:17] = list(np.array(result[ii][9:17]))
            result[ii][17:25] = list(np.array(result[ii][17:25]))
        boxes = result
        marker_array.markers.clear()
        marker_array_text.markers.clear()

        for obid in range(len(boxes)):
            ob = boxes[obid]
            detect_points_set = []
            for i in range(0, 8):
                detect_points_set.append(Point(ob[i+1], ob[i+9], ob[i+17]))

            # if ob[0] not in ["car", "bus", "truck"]:
            #     continue

            box = BoundingBox()
            box.header = header
            box.label = TYPE_INDICES[ob[0]]

            box.pose.position.x = ob[-3][0]
            box.pose.position.y = ob[-3][1]
            box.pose.position.z = ob[-3][2]

            box.dimensions.x = ob[-2][0]
            box.dimensions.y = ob[-2][1]
            box.dimensions.z = ob[-2][2]

            c = np.cos(-ob[-1] + np.pi/2)
            s = np.sin(-ob[-1] + np.pi/2)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            quat = R.from_matrix(rot).as_quat()  # (x, y, z, w)

            box.pose.orientation.x = quat[0]
            box.pose.orientation.y = quat[1]
            box.pose.orientation.z = quat[2]
            box.pose.orientation.w = quat[3]
            
            response.detections.boxes.append(box)

        return response

    def warm_up(self):
        request = detectionRequest()
        request.cloud = point_cloud(np.array([[3, 3, 3, 1]]))
        for _ in range(5):
            self.LIOSLOTCallback(request)

if __name__ == '__main__':
    livox = Detector()
    print("Warm up...")
    livox.warm_up()
    print("Start working ...")
    rospy.spin()
