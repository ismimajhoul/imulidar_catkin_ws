#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion

class RobotController:
    def __init__(self, robot_name='turtlebot3_waffle', reference_frame='world'):
        self.robot_name = robot_name
        self.reference_frame = reference_frame
        rospy.init_node('robot_controller', anonymous=True)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.left_torque_pub = rospy.Publisher('/apply_force_left', Wrench, queue_size=10)
        self.right_torque_pub = rospy.Publisher('/apply_force_right', Wrench, queue_size=10)

        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)

        self.orientation = None
        self.roll = 0.0
        self.pitch = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.seq_num = 0  # Sequence number for logging

        # PID coefficients
        self.kp = 1.0  # Augmenter Kp pour une réponse plus rapide
        self.ki = 0.001  # Réduire Ki pour éviter une accumulation rapide
        self.kd = 0.01

        self.prev_error = 0.0
        self.integral = 0.0

        # Thresholds for roll and pitch
        self.roll_threshold_high = -1.0
        self.roll_threshold_low = -0.5
        self.pitch_threshold = 0.05

        self.suspend_torque = False
        self.suspend_counter = 20

    def imu_callback(self, data):
        self.orientation = data.orientation
        orientation_list = [data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]
        (self.roll, self.pitch, _) = euler_from_quaternion(orientation_list)

        if self.seq_num % 10 == 0:
            rospy.loginfo(f"[Seq: {self.seq_num}] IMU raw data - Orientation: x: {data.orientation.x}, y: {data.orientation.y}, z: {data.orientation.z}, w: {data.orientation.w}, "
                          f"Linear Acc: x: {data.linear_acceleration.x}, y: {data.linear_acceleration.y}, z: {data.linear_acceleration.z}, "
                          f"Angular Vel: x: {data.angular_velocity.x}, y: {data.angular_velocity.y}, z: {data.angular_velocity.z}")
            rospy.loginfo(f"[Seq: {self.seq_num}] IMU converted data - Roll: {self.roll}, Pitch: {self.pitch}")

    def get_robot_position(self):
        try:
            state = self.get_state_service(self.robot_name, self.reference_frame)
            if state.success:
                self.x = state.pose.position.x
                self.y = state.pose.position.y
                self.theta = 2 * math.atan2(state.pose.orientation.z, state.pose.orientation.w)
                if self.seq_num % 10 == 0:
                    rospy.loginfo(f"[Seq: {self.seq_num}] Gazebo - x: {self.x}, y: {self.y}, theta: {self.theta}")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def log_positions(self, target_x, target_y):
        if self.seq_num % 10 == 0:
            gazebo_distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
            rospy.loginfo(f"[Seq: {self.seq_num}] => Distance to target (Gazebo): {gazebo_distance}, Threshold: 0.1")

    def calculate_orientation(self, target_x, target_y):
        angle = math.atan2(target_y - self.y, target_x - self.x)
        return angle

    def set_robot_orientation(self, angle):
        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.position.x = self.x
        state_msg.pose.position.y = self.y
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.z = math.sin(angle / 2.0)
        state_msg.pose.orientation.w = math.cos(angle / 2.0)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            rospy.loginfo(f"[Seq: {self.seq_num}] *** Robot orienté selon l'angle: {angle} radians ({math.degrees(angle)} degrés)")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def calculate_pid(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def move_to_target(self, target_x, target_y):
        rate = rospy.Rate(10)  # Adjusting frequency to 10 Hz
        while not rospy.is_shutdown():
            self.seq_num += 1  # Increment sequence number
            self.get_robot_position()

            distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)

            self.log_positions(target_x, target_y)

            if distance < 0.1:
                rospy.loginfo(f"[Seq: {self.seq_num}] *** Target reached (Gazebo): x={self.x}, y={self.y}")
                self.stop_robot()
                break

            # Check roll and pitch before applying torque
            if self.roll < self.roll_threshold_high:
                rospy.logwarn(f"[Seq: {self.seq_num}] Roll too high, suspending torque application")
                self.suspend_torque = True
                self.suspend_counter = 10  # Suspend for 10 iterations
            elif self.suspend_torque and self.roll > self.roll_threshold_low:
                self.suspend_counter -= 1
                if self.suspend_counter <= 0:
                    rospy.loginfo(f"[Seq: {self.seq_num}] Roll back within limits, resuming torque application")
                    self.suspend_torque = False

            if not self.suspend_torque:
                torque = self.calculate_pid(distance)
                torque = max(0.5, torque * distance)  # Augmenter le couple minimal
                torque = min(torque, 5)  # Augmenter le couple maximal

                desired_angle = self.calculate_orientation(target_x, target_y)
                angle_error = desired_angle - self.theta
                rospy.loginfo(f"[Seq: {self.seq_num}] => Orientation error: {angle_error}")

                if abs(angle_error) > 0.1:
                    self.set_robot_orientation(desired_angle)

                wrench = Wrench()
                wrench.force.x = torque * math.cos(desired_angle)
                wrench.force.y = torque * math.sin(desired_angle)
                
                self.left_torque_pub.publish(wrench)
                self.right_torque_pub.publish(wrench)
                rospy.loginfo(f"[Seq: {self.seq_num}] => Envoi de couple - x={wrench.force.x}, y={wrench.force.y}")
            else:
                rospy.loginfo(f"[Seq: {self.seq_num}] Torque application suspended due to high roll")

            rate.sleep()

    def stop_robot(self):
        # Publier un couple nul pour arrêter le robot
        wrench = Wrench()
        wrench.force.x = 0.0
        wrench.force.y = 0.0
        self.left_torque_pub.publish(wrench)
        self.right_torque_pub.publish(wrench)
        rospy.loginfo(f"[Seq: {self.seq_num}] => Robot arrêté")

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.move_to_target(0.0, 9.5)  # Déplacement vers (0,3)
    except rospy.ROSInterruptException:
        pass

