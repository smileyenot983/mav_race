#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Transform, Vector3, Quaternion
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from cv_bridge import CvBridge

import numpy as np
import cv2
from cv2 import aruco
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline

from message_filters import ApproximateTimeSynchronizer, Subscriber

f_x = f_y = 205.46963709898583
c_x = 640 / 2
c_y = 480 / 2
intrinsics = np.array([[f_x, 0.0, c_x],
                        [0.0, f_y, c_y],
                        [0.0, 0.0, 1.0]])

# transformation from camera CS to robot CS
cam2rob_extrinsics = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])

# gt known transformation, taken from urdf
cam2rob_translation = np.array([0.0, 0.0, 0.0])

# axis alignment + fixed translation 
def cam2rob(position, align_axis = False):

    return np.dot(cam2rob_extrinsics, position)

class ArucoDetector:
    def __init__(self, debug=False):
        pass

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        
        self.parameters.adaptiveThreshWinSizeMin = 4
        self.parameters.adaptiveThreshWinSizeMax = 10
        self.parameters.adaptiveThreshWinSizeStep = 3

        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.maxMarkerPerimeterRate = 0.45
        self.parameters.polygonalApproxAccuracyRate = 0.05
        
        self.parameters.minCornerDistanceRate = 0.01
        self.parameters.minMarkerDistanceRate = 0.02

        self.parameters.perspectiveRemovePixelPerCell = 3
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.1
        
        self.debug = debug
        self.detection_counter = 0



    def detect(self,img):

        # Convert to grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        # visualize
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)

        if(self.debug):
            if not ids is None:
                self.detection_counter += 1
            print(f"self.detection_counter: {self.detection_counter}")

        

        return corners, frame_markers

class ArucoTracker:
    def __init__(self):
        # holds positions of 4 markers corners
        self.markers = []
        self.visited = []

        self.initial_rotation = None
        
        self.previous_pos = np.array([-1.0, -2.0])

        self.last_idx = -1

        self.min_markers_dist = 1.0
        self.marker_height = 3.5

        self.initialized = False

    # adds marker into markers list
    def add_marker(self, corners, corners_wframe, depth, pose_mat):
        
        if(len(self.markers) == 0):
            print(f"new marker: {corners_wframe}")
            self.markers.append(corners_wframe)
            self.visited.append(False)
        else:
            # check that there no nans in depth:
            corners = corners[0][0]
            for i in range(corners.shape[0]):
                depth_i = depth[int(corners[i,1]), int(corners[i,0])]

                if(np.isnan(depth_i)):
                    return

            mean_x = np.mean(corners_wframe[:,0])
            mean_y = np.mean(corners_wframe[:,1])

            new_marker = True
            for i in range(len(self.markers)):
                    
                mean_x_i = np.mean(self.markers[i][:,0])
                mean_y_i = np.mean(self.markers[i][:,1])

                # if markers are close -> it's same marker
                if(np.abs(mean_x - mean_x_i) + np.abs(mean_y - mean_y_i) < self.min_markers_dist):
                    new_marker = False
                    break


            if(new_marker): 
                print(f"new marker: {corners_wframe}")
                # add marker only in case it has enough distance from previous markers 
                self.markers.append(corners_wframe)
                self.visited.append(False)

    # returns closest unvisited marker's pose
    def get_next_marker(self):

        next_marker_id_candidates = []
        for i in range(len(self.visited)):
            if self.visited[i] == False:
                next_marker_id_candidates.append(i)

        print(f"next_marker_id_candidates: {next_marker_id_candidates}")
        print(f"self.visited: {self.visited}")
        # find closest one in case it is not first:
        next_marker_id = -1
        closest_dist = 20
        for marker_id in next_marker_id_candidates:
            
            corners_wframe_i = self.markers[marker_id]
            pose_x = np.mean(corners_wframe_i[:,0])
            pose_y = np.mean(corners_wframe_i[:,1])

            dist_i = np.abs(pose_x - self.previous_pos[0]) + np.abs(pose_y - self.previous_pos[1])
            if(dist_i < closest_dist):
                next_marker_id = marker_id
                closest_dist = dist_i

        print(f"next_marker_id: {next_marker_id}")

        self.last_idx = next_marker_id
        
        if next_marker_id == -1:
            return False, None, None
        else:
            corners_wframe_last = self.markers[next_marker_id]

            # get pose as mean of markers
            pose_x = np.mean(corners_wframe_last[:,0])
            pose_y = np.mean(corners_wframe_last[:,1])
            pose_z = self.marker_height

            corner0 = corners_wframe_last[0,:]
            corner1 = corners_wframe_last[1,:]
            corner2 = corners_wframe_last[2,:]
            corner3 = corners_wframe_last[3,:]

            axis_y = (corner3 - corner2) / np.linalg.norm(corner3 - corner2)
            axis_z = (corner1 - corner2) / np.linalg.norm(corner1 - corner2)
            axis_x = np.cross(axis_y, axis_z)
            
            rotation = np.column_stack((axis_x, axis_y, axis_z))

            self.previous_pos = np.array([pose_x, pose_y])

            if self.initial_rotation is None:
                self.initial_rotation = rotation
                return True, np.array([0.0, 0.0, 0.0, 1.0]), np.array([pose_x, pose_y, pose_z])
            # otherwise it is a difference
            else:            

                rotation_diff = np.dot(self.initial_rotation.transpose(), rotation)
                # print(f"rotation_diff: {rotation_diff}")
                r = Rotation.from_matrix(rotation_diff)

                return True, r.as_quat(), np.array([pose_x, pose_y, pose_z])

    # check whether last marker was reached
    def last_point_reached(self, pose_mat):

        # pos_rob = pose_mat[:3,3] + cam2rob_translation
        # pos_rob = pose_mat[:3,3] + cam2rob_translation
        pos_rob = pose_mat[:3,3]


        # in case of first marker -> no need to check whether previous was reached
        if(self.last_idx == -1):
            return True
        else:
            corners_wframe_i = self.markers[self.last_idx]
            pose_x = np.mean(corners_wframe_i[:,0])
            pose_y = np.mean(corners_wframe_i[:,1])
            
            dist_i = np.abs(pose_x - pos_rob[0]) + np.abs(pose_y - pos_rob[1]) #+ np.abs(self.marker_height - pos_rob[2])

            if(dist_i < 0.5):
                self.visited[self.last_idx] = True
                return True
            else:
                return False

# uses splines for smooth trajectory generation
def generate_trajectory(pt_start, quat_start, pt_end, quat_end):

    # interpolate positions
    print(f"pt_start: {pt_start} | pt_end: {pt_end}")
    control_points = np.array([pt_start, pt_end])

    t = np.linspace(0, 1, len(control_points))

    cs_x = CubicSpline(t, control_points[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t, control_points[:, 1], bc_type="clamped")
    cs_z = CubicSpline(t, control_points[:, 2], bc_type="clamped")

    num_points = 100
    t_values = np.linspace(0, 1, num_points)
    trajectory_x = cs_x(t_values)
    trajectory_y = cs_y(t_values)
    trajectory_z = cs_z(t_values)

    # interpolate angles(SLERP)
    rot1 = Rotation.from_quat(quat_start)
    rot2 = Rotation.from_quat(quat_end)
    rot_stacked = Rotation.concatenate([rot1,rot2])


    slerp = Slerp(t, rot_stacked)
    trajectory_rotations = slerp(t_values)

    trajectory_msg = MultiDOFJointTrajectory()
    trajectory_msg.header.frame_id = "firefly/odometry_sensor1_link"

    for i in range(num_points):
        pos_i = Vector3(trajectory_x[i],
                        trajectory_y[i],
                        trajectory_z[i])
        
        rot_i = Quaternion(trajectory_rotations[i].as_quat()[0],
                           trajectory_rotations[i].as_quat()[1],
                           trajectory_rotations[i].as_quat()[2],
                           trajectory_rotations[i].as_quat()[3])
        
        transform_i = Transform(pos_i, rot_i)

        trajectory_pt_i = MultiDOFJointTrajectoryPoint()
        trajectory_pt_i.transforms = [transform_i]

        trajectory_msg.points.append(trajectory_pt_i)    
    
    return trajectory_msg


# transforms marker's px coordinates into real world coordinates
def getArucoWorldCoord(corners, depth, cam2rob_extrinsics, intrinsics, pose):
    intrinsics_inv = np.linalg.inv(intrinsics)
    corners_wframe = []
    # print(corners.shape)
    for i in range(corners.shape[0]):
        corner_pxframe = np.array([corners[i,0], corners[i,1], 1.0])

        corner_depth = depth[int(corners[i,1]), int(corners[i,0])]

        corner_camframe = corner_depth * np.dot(intrinsics_inv, corner_pxframe)

        corner_robframe = cam2rob(corner_camframe, True)

        corner_wframe = np.dot(pose[:3,:3], corner_robframe) + pose[:3,3] 
        
        corners_wframe.append(corner_wframe)
    corners_wframe = np.array(corners_wframe)
    return corners_wframe



# converts quaternion to rotation matrix
def quat2mat(quaternion):
    r = Rotation.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

    return r.as_matrix()

# converts ros Pose message to SE(3) pose matrix
def pose2mat(pose_msg):
    rot_mat = quat2mat(pose_msg.orientation)

    pose_mat = np.eye(4)
    pose_mat[:3,:3] = rot_mat
    pose_mat[0,3] = pose_msg.position.x
    pose_mat[1,3] = pose_msg.position.y
    pose_mat[2,3] = pose_msg.position.z

    return pose_mat

# checks that linear and angular velocities are less than threshold
def checkZeroTwist(odometry_msg, max_linear = 0.2, max_angular = 0.2):

    v_x = odometry_msg.twist.twist.linear.x
    v_y = odometry_msg.twist.twist.linear.y
    v_z = odometry_msg.twist.twist.linear.z

    w_x = odometry_msg.twist.twist.angular.x
    w_y = odometry_msg.twist.twist.angular.y
    w_z = odometry_msg.twist.twist.angular.z

    if (np.abs(v_x) < max_linear and
        np.abs(v_y) < max_linear and
        np.abs(v_z) < max_linear and
        np.abs(w_x) < max_angular and 
        np.abs(w_y) < max_angular and
        np.abs(w_z) < max_angular):
        return True
    else:
        return False
    

aruco_detector = ArucoDetector()
aruco_tracker = ArucoTracker()

trajectory_publisher = rospy.Publisher("/firefly/command/trajectory", MultiDOFJointTrajectory, queue_size=10)

def img_depth_pose_callback(image_msg, depth_msg, odometry_msg):
    bridge = CvBridge()

    # get both
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    corners, upd_image = aruco_detector.detect(image)

    pose_mat = pose2mat(odometry_msg.pose.pose)

    if len(corners) > 0:
        
        corners_wframe = getArucoWorldCoord(corners[0][0], depth, cam2rob_extrinsics, intrinsics, pose_mat)
        aruco_tracker.add_marker(corners, corners_wframe, depth, pose_mat)

    if(aruco_tracker.last_point_reached(pose_mat) and checkZeroTwist(odometry_msg)):

        r = Rotation.from_matrix(pose_mat[:3,:3])

        new_marker, marker_rot, marker_pos = aruco_tracker.get_next_marker()

        if(new_marker):
            trajectory_new = generate_trajectory(pose_mat[:3,3], r.as_quat(),  marker_pos, marker_rot)
            trajectory_publisher.publish(trajectory_new)
        else:
            print("All markers were visited!")


    upd_image_msg = bridge.cv2_to_imgmsg(upd_image)
    pub = rospy.Publisher("upd_image", Image, queue_size=10)
    pub.publish(upd_image_msg)



def gate_processer():

    rospy.init_node('gate_processer', anonymous=True)

    image_sub = Subscriber("/firefly/vi_sensor/camera_depth/camera/image_raw", Image)
    depth_sub = Subscriber("/firefly/vi_sensor/camera_depth/depth/disparity", Image)

    # odometry is estimated in camera frame
    pose_sub = Subscriber("/firefly/ground_truth/odometry", Odometry)

    tyme_syncer = ApproximateTimeSynchronizer([image_sub, depth_sub, pose_sub], queue_size=10, slop=0.5)
    tyme_syncer.registerCallback(img_depth_pose_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    gate_processer()