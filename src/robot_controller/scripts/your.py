#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class RobotController:
    def __init__(self, robot_name='turtlebot3_burger', reference_frame='world'):
        self.robot_name = robot_name
        self.reference_frame = reference_frame
        rospy.init_node('robot_controller', anonymous=True)

        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.left_torque_pub = rospy.Publisher('/apply_force_left', Wrench, queue_size=10)
        self.right_torque_pub = rospy.Publisher('/apply_force_right', Wrench, queue_size=10)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # PID coefficients
        self.kp = 0.5
        self.ki = 0.01
        self.kd = 0.1

        self.prev_error = 0.0
        self.integral = 0.0

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
            rospy.loginfo(f"*** Robot orienté selon l'angle: {angle} radians ({math.degrees(angle)} degrés)")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def get_robot_position(self):
        try:
            state = self.get_state_service(self.robot_name, self.reference_frame)
            if state.success:
                self.x = state.pose.position.x
                self.y = state.pose.position.y
                self.theta = 2 * math.atan2(state.pose.orientation.z, state.pose.orientation.w)
                rospy.loginfo(f"*** Position du robot mise à jour - x: {self.x}, y: {self.y}, theta: {self.theta}")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def calculate_pid(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def move_to_target(self, target_x, target_y):
        self.get_robot_position()
        angle = self.calculate_orientation(target_x, target_y)
        self.set_robot_orientation(angle)
        
        rospy.sleep(1)  # Wait for the orientation to take effect

        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.get_robot_position()

            distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
            rospy.loginfo(f"=> Distance to target: {distance}, Threshold: 0.1")

            if distance < 0.1:
                rospy.loginfo(f"*** Target reached: x={self.x}, y={self.y}")
                self.apply_reverse_torque()
                break

            # Calculate PID output for distance
            torque = self.calculate_pid(distance)

            # Adjust torque based on distance to avoid overshooting
            if distance < 1.0:
                torque = max(0.1, torque * distance)

            # Calculate orientation error and adjust
            desired_angle = self.calculate_orientation(target_x, target_y)
            angle_error = desired_angle - self.theta
            rospy.loginfo(f"=> Orientation error: {angle_error}")

            # Adjust orientation if the error is significant
            if abs(angle_error) > 0.1:
                self.set_robot_orientation(desired_angle)

            # Apply torque in the direction of the target
            wrench = Wrench()
            wrench.force.x = torque * math.cos(desired_angle)
            wrench.force.y = torque * math.sin(desired_angle)
            
            self.left_torque_pub.publish(wrench)
            self.right_torque_pub.publish(wrench)
            rospy.loginfo(f"=> Envoi de couple - x={wrench.force.x}, y={wrench.force.y}")

            rate.sleep()

    def apply_reverse_torque(self):
        rate = rospy.Rate(10)
        for _ in range(10):  # Apply reverse torque for a short period
            wrench = Wrench()
            wrench.force.x = -0.05  # Apply small reverse torque
            wrench.force.y = -0.05
            self.left_torque_pub.publish(wrench)
            self.right_torque_pub.publish(wrench)
            rospy.loginfo(f"=> Envoi de couple inverse - x={wrench.force.x}, y={wrench.force.y}")
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.move_to_target(3.0, 3.0)  # Déplacement vers (3,3)
    except rospy.ROSInterruptException:
        pass

