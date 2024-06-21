#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
import tf
import math

class IMUPositionEstimator:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_time = None
        self.accel_bias = [0.0, 0.0, 0.0]

        rospy.init_node('imu_position_estimator', anonymous=True)
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.tf_listener = tf.TransformListener()

    def imu_callback(self, data):
        current_time = rospy.get_time()
        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time

        # Get linear acceleration from IMU
        ax = data.linear_acceleration.x - self.accel_bias[0]
        ay = data.linear_acceleration.y - self.accel_bias[1]
        az = data.linear_acceleration.z - self.accel_bias[2]

        # Log IMU data
        rospy.loginfo(f"IMU Data - ax: {ax}, ay: {ay}, az: {az}")

        # Integrate acceleration to get velocity
        self.vx += ax * dt
        self.vy += ay * dt

        # Integrate velocity to get position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Get orientation (theta) from IMU quaternion
        orientation = data.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]  # yaw angle

        # Log calculated position and orientation
        rospy.loginfo(f"Calculated Position - x: {self.x}, y: {self.y}, theta: {self.theta}")

        self.last_time = current_time

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        estimator = IMUPositionEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass

