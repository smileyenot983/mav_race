
# mav_race

Autonomous MAV(drone) flying through gates in a Gazebo framework

# Prerequisites

1. ros complete installation(https://wiki.ros.org/noetic/Installation/Ubuntu)
2. rotors simulator ROS package(https://github.com/ethz-asl/rotors_simulator)
3. mav messages package(https://github.com/ethz-asl/mav_comm)
4. It uses OpenCV, numpy and scipy. These are usually installed together with ROS, but in case they aren't installed or deleted, they are added to `requirements.txt`

# Build

Building is same as for any other ROS package
1. Build packages `catkin build`
2. Source `source devel/setup.bash`

# Running

1. Run the simulator `roslaunch mav_race mav_gates.launch`
2. Move to initial point by publishing ros message 
`rostopic pub /firefly/command/pose geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: -1.0
    y: -2.0
    z: 3.5
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0"
    `

3. After robot reaches the desired position(3-5 seconds maximum) run the node which flies through gates:
`rosrun mav_race gate_processer.py`

In short time after running `gate_processer.py` it should start flying through gates. It should stop near the last gate.


There are might be some bugs, in case of abnormal behavior it is recommended to restart everything.

# Video

[![racing simulation video](https://img.youtube.com/vi/b6nN-xbmpQM/0.jpg)](https://www.youtube.com/watch?v=b6nN-xbmpQM)


# The approach

The overall approach relies on aruco markers and depth camera for gate positions estimation. The `ArucoDetector` class detects markers, whereas `ArucoTracker` performs tracking and path planning. In order to achieve smooth trajectory Cubic spline interpolation is used for both: position and orientation.

Odometry is taken as ground-truth, but there is also second launch file `mav_gates_noisy.launch`, which uses corrupted odometry, corruption is done by adding normal noise to position and orientation. Running might be done in exactly same way: `roslaunch mav_race mav_gates_noisy.launch`. Most of the time it works, but with less stability.





