#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class RobotController:
    def __init__(self, robot_name='turtlebot3_burger', reference_frame='world'):
        self.robot_name = robot_name
        self.reference_frame = reference_frame
        rospy.init_node('trajectory_bot', anonymous=True)
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.ins_sub = rospy.Subscriber('/ins', Imu, self.ins_callback)
        
        self.left_torque_pub = rospy.Publisher('/apply_force_left', Wrench, queue_size=10)
        self.right_torque_pub = rospy.Publisher('/apply_force_right', Wrench, queue_size=10)
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.target_x = 3.0
        self.target_y = 3.0
        self.error_threshold = 0.1
        self.max_torque = 1.0

        self.kp = 1.0
        self.ki = 0.01
        self.kd = 0.1

        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_theta = 0.0

        self.imu_data = None
        self.ins_data = None

        self.calculate_trajectory_and_orientation()

    def imu_callback(self, data):
        self.imu_data = data

    def ins_callback(self, data):
        self.ins_data = data

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

    def calculate_trajectory_and_orientation(self):
        if self.target_x == self.x and self.target_y == self.y:
            self.slope = None
            self.desired_angle = self.theta
        elif self.target_x == self.x:
            self.slope = float('inf')
            self.desired_angle = math.pi / 2 if self.target_y > self.y else -math.pi / 2
        elif self.target_y == self.y:
            self.slope = 0.0
            self.desired_angle = 0.0 if self.target_x > self.x else math.pi
        else:
            self.slope = (self.target_y - self.y) / (self.target_x - self.x)
            self.desired_angle = math.atan2(self.target_y - self.y, self.target_x - self.x)
        
        rospy.loginfo(f"*** Coefficient directeur de la trajectoire: {self.slope}")
        rospy.loginfo(f"*** Angle entre l'orientation initiale et la direction du point cible: {self.desired_angle} radians")

    def set_robot_orientation(self):
        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.orientation.z = math.sin(self.desired_angle / 2.0)
        state_msg.pose.orientation.w = math.cos(self.desired_angle / 2.0)

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            rospy.loginfo(f"*** Robot orienté selon l'angle: {self.desired_angle} radians")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def set_target_point(self, new_x, new_y):
        self.target_x = new_x
        self.target_y = new_y
        rospy.loginfo(f"=> Nouveau point cible défini - x: {self.target_x}, y: {self.target_y}")
        self.calculate_trajectory_and_orientation()

    def apply_torque(self):
        self.set_robot_orientation()
        rospy.sleep(2)

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.get_robot_position()

            error_x = self.target_x - self.x
            error_y = self.target_y - self.y
            distance_to_target = math.sqrt(error_x**2 + error_y**2)
            rospy.loginfo(f"=> Distance to target: {distance_to_target}, Threshold: {self.error_threshold}")
            
            if distance_to_target < self.error_threshold:
                rospy.loginfo(f"*** Target reached: x={self.x}, y={self.y}")
                wrench_left = Wrench()
                wrench_right = Wrench()
                wrench_left.force.x = 0.0
                wrench_right.force.x = 0.0
                wrench_left.force.y = 0.0
                wrench_right.force.y = 0.0
                self.left_torque_pub.publish(wrench_left)
                self.right_torque_pub.publish(wrench_right)
                break

            error_theta = self.desired_angle - self.theta
            if error_theta > math.pi:
                error_theta -= 2 * math.pi
            elif error_theta < -math.pi:
                error_theta += 2 * math.pi

            delta_time = 0.1
            self.integral_x += error_x * delta_time
            self.integral_y += error_y * delta_time
            self.integral_theta += error_theta * delta_time

            derivative_x = (error_x - self.prev_error_x) / delta_time
            derivative_y = (error_y - self.prev_error_y) / delta_time
            derivative_theta = (error_theta - self.prev_error_theta) / delta_time

            torque_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
            torque_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y
            torque_theta = self.kp * error_theta + self.ki * self.integral_theta + self.kd * derivative_theta

            torque_x = min(self.max_torque, max(-self.max_torque, torque_x))
            torque_y = min(self.max_torque, max(-self.max_torque, torque_y))
            torque_theta = min(self.max_torque, max(-self.max_torque, torque_theta))

            self.prev_error_x = error_x
            self.prev_error_y = error_y
            self.prev_error_theta = error_theta

            wrench_left = Wrench()
            wrench_right = Wrench()
            
            wrench_left.force.x = torque_x
            wrench_right.force.x = torque_y
            wrench_left.force.y = torque_theta
            wrench_right.force.y = torque_theta
            
            self.left_torque_pub.publish(wrench_left)
            self.right_torque_pub.publish(wrench_right)
            
            rospy.loginfo(f"*** Position actuelle - x: {self.x}, y: {self.y}, theta: {self.theta}")
            rospy.loginfo(f"=> Position désirée - x_d: {self.target_x}, y_d: {self.target_y}")
            rospy.loginfo(f"=> Envoi de couple - Left: x={wrench_left.force.x}, y={wrench_left.force.y}, Right: x={wrench_right.force.x}, y={wrench_right.force.y}")

            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.set_target_point(3.0, 3.0)  # Exemple de point cible
        controller.apply_torque()
    except rospy.ROSInterruptException:
        pass

