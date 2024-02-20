#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np

from cv_bridge import CvBridge
import cv2

from message_filters import ApproximateTimeSynchronizer, Subscriber


cntr_img = 0
def image_saver(img_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    global cntr_img
    cv2.imwrite(f"/home/ramil/catkin_ws/src/mav_race/imgs/img_{cntr_img}.png", cv_image)
    cntr_img+=1

    print(f"cv_image.shape: {cv_image.shape}")

cntr_disp = 0
def disp_saver(img_msg):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    global cntr_disp
    np.save(f"/home/ramil/catkin_ws/src/mav_race/depths/depth_{cntr_disp}.npy", depth_image)
    cntr_disp+=1

    print(f"depth.shape: {depth_image.shape}")

cntr_cloud = 0
def cloud_saver(cloud_msg):
    # pc_array = ros_numpy.numpify(cloud_msg)
    
    pts = []
    for point in pc2.read_points(cloud_msg, skip_nans=True):
        x, y, z = point[:3]

        pts.append(point[:3])

    pts = np.array(pts)

    global cntr_cloud
    np.save(f"/home/ramil/catkin_ws/src/mav_race/clouds/cloud_{cntr_cloud}.npy", pts)
    cntr_cloud += 1
    print(f"pts.shape: {pts.shape}")

cntr_synced = 0
def synced_callback(image_msg, depth_msg):
    bridge = CvBridge()

    # get both
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # save
    global cntr_synced
    cv2.imwrite(f"/home/ramil/catkin_ws/src/mav_race/imgs/img_{cntr_synced}.png", image)
    np.save(f"/home/ramil/catkin_ws/src/mav_race/depths/depth_{cntr_synced}.npy", depth)
    cntr_synced += 1



def listener():

    rospy.init_node('img_reader', anonymous=True)

    # rospy.Subscriber("/firefly/vi_sensor/camera_depth/camera/image_raw", Image, image_saver)
    # rospy.Subscriber("/firefly/vi_sensor/camera_depth/depth/disparity", Image, disp_saver)
    # rospy.Subscriber("/firefly/vi_sensor/camera_depth/depth/points", PointCloud2, cloud_saver)

    image_sub = Subscriber("/firefly/vi_sensor/camera_depth/camera/image_raw", Image)
    depth_sub = Subscriber("/firefly/vi_sensor/camera_depth/depth/disparity", Image)

    tyme_syncer = ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.5)
    tyme_syncer.registerCallback(synced_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()