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
        rospy.init_node('step_bot', anonymous=True)
        
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
        
        # Paramètres PID initiaux
        self.kp_position = 0.1
        self.kd_position = 0.0
        self.ki_position = 0.0
        self.kp_angle = 0.1
        self.kd_angle = 0.0
        self.ki_angle = 0.0
        
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self.previous_error_theta = 0.0
        self.integral_error_x = 0.0
        self.integral_error_y = 0.0
        self.integral_error_theta = 0.0
        self.previous_time = rospy.get_time()
        self.phase = 'x'
        self.imu_data = None
        self.ins_data = None
        self.x_reached = False
        self.y_reached = False

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

    def set_robot_position(self):
        state_msg = ModelState()
        state_msg.model_name = self.robot_name
        state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = 0.0
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = 0.0
        state_msg.pose.orientation.w = 1.0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            rospy.loginfo("=> Robot repositionné à l'origine: x=0.0, y=0.0, theta=0.0")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def set_target_point(self, new_x, new_y):
        self.target_x = new_x
        self.target_y = new_y
        rospy.loginfo(f"=> Nouveau point cible défini - x: {self.target_x}, y: {self.target_y}")

    def has_reached_target(self):
        if self.phase == 'x':
            rospy.loginfo(f"=> error_threshold - x: {abs(self.target_x - self.x)}, err: {self.error_threshold}")
            if abs(self.target_x - self.x) < self.error_threshold:
                self.x_reached = True
                return True
        elif self.phase == 'y':
            rospy.loginfo(f"=> error_threshold - y: {abs(self.target_y - self.y)}, err: {self.error_threshold}")
            if abs(self.target_y - self.y) < self.error_threshold:
                self.y_reached = True
                return True
        return False

    def calculate_differential(self, v, omega):
        delta_time = rospy.get_time() - self.previous_time
        self.previous_time = rospy.get_time()
        
        new_x = self.x + v * math.cos(self.theta) * delta_time
        new_y = self.y + v * math.sin(self.theta) * delta_time
        new_theta = self.theta + omega * delta_time
        
        return new_x, new_y, new_theta

    def calculate_required_torque(self):
        current_time = rospy.get_time()
        if self.previous_time is None:
            self.previous_time = current_time

        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_theta = math.atan2(error_y, error_x) - self.theta

        rospy.loginfo(f"Erreur x: {error_x}, Erreur y: {error_y}, Erreur theta: {error_theta}")

        delta_time = current_time - self.previous_time
        if delta_time == 0:
            derivative_x = 0
            derivative_y = 0
            derivative_theta = 0
        else:
            derivative_x = (error_x - self.previous_error_x) / delta_time
            derivative_y = (error_y - self.previous_error_y) / delta_time
            derivative_theta = (error_theta - self.previous_error_theta) / delta_time

        self.integral_error_x += error_x * delta_time
        self.integral_error_y += error_y * delta_time
        self.integral_error_theta += error_theta * delta_time

        torque_x = self.kp_position * error_x + self.kd_position * derivative_x + self.ki_position * self.integral_error_x
        torque_y = self.kp_position * error_y + self.kd_position * derivative_y + self.ki_position * self.integral_error_y
        torque_theta = self.kp_angle * error_theta + self.kd_angle * derivative_theta + self.ki_angle * self.integral_error_theta

        rospy.loginfo(f"PID Torque x: {torque_x}, PID Torque y: {torque_y}, PID Torque theta: {torque_theta}")

        torque_x = min(self.max_torque, max(-self.max_torque, torque_x))
        torque_y = min(self.max_torque, max(-self.max_torque, torque_y))
        torque_theta = min(self.max_torque, max(-self.max_torque, torque_theta))

        rospy.loginfo(f"Limité Torque x: {torque_x}, Limité Torque y: {torque_y}, Limité Torque theta: {torque_theta}")

        self.previous_error_x = error_x
        self.previous_error_y = error_y
        self.previous_error_theta = error_theta
        self.previous_time = current_time

        return torque_x, torque_y, torque_theta

    def apply_torque(self):
        self.set_robot_position()  # Réinitialiser la position du robot
        rospy.sleep(2)  # Attendre que le repositionnement soit effectif
        self.get_robot_position()  # Vérifier la position initiale

        rate = rospy.Rate(1)  # 1 Hz

        while not rospy.is_shutdown():
            self.get_robot_position()  # Mettre à jour la position du robot
            
            if self.has_reached_target():
                if self.phase == 'x' and self.x_reached:
                    rospy.loginfo(f"=> Position x atteinte : x={self.x}")
                    self.phase = 'y'  # Switch to regulating y
                    self.x_reached = False  # Reset the flag for next iteration
                elif self.phase == 'y' and self.y_reached:
                    rospy.loginfo(f"=> Position y atteinte : y={self.y}")
                    self.y_reached = False  # Reset the flag for next iteration
                    wrench_left = Wrench()
                    wrench_right = Wrench()
                    wrench_left.force.x = 0.0
                    wrench_right.force.x = 0.0
                    wrench_left.force.y = 0.0
                    wrench_right.force.y = 0.0
                    self.left_torque_pub.publish(wrench_left)
                    self.right_torque_pub.publish(wrench_right)
                    break
            
            torque_x, torque_y, torque_theta = self.calculate_required_torque()

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

            if self.imu_data:
                rospy.loginfo(f"IMU Data - Orientation: {self.imu_data.orientation}, Angular Velocity: {self.imu_data.angular_velocity}, Linear Acceleration: {self.imu_data.linear_acceleration}")
            
            if self.ins_data:
                rospy.loginfo(f"INS Data - Orientation: {self.ins_data.orientation}, Angular Velocity: {self.ins_data.angular_velocity}, Linear Acceleration: {self.ins_data.linear_acceleration}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.set_target_point(3.0, 3.0)  # Exemple de point cible
        controller.apply_torque()
    except rospy.ROSInterruptException:
        pass

