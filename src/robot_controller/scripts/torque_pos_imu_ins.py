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
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Subscribe to IMU and INS topics
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.ins_sub = rospy.Subscriber('/ins', Imu, self.ins_callback)
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.target_x = 3.0  # Point cible en x
        self.target_y = 2.0  # Point cible en y
        self.error_threshold = 0.5  # Seuil d'erreur pour considérer un point comme atteint
        self.max_torque = 1.0  # Couple maximum à appliquer pour limiter la vitesse
        self.kp_position = 0.5  # Coefficient proportionnel pour la position
        self.kd_position = 0.1  # Coefficient dérivé pour la position
        self.ki_position = 0.05  # Coefficient intégral pour la position
        self.kp_angle = 1.0  # Coefficient proportionnel pour l'angle
        self.kd_angle = 0.1  # Coefficient dérivé pour l'angle
        self.ki_angle = 0.05  # Coefficient intégral pour l'angle
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self.integral_error_x = 0.0
        self.integral_error_y = 0.0
        self.previous_time = None
        self.imu_data = None
        self.ins_data = None
        self.x_reached = False  # Drapeau pour indiquer que x est atteint
        self.y_reached = False  # Drapeau pour indiquer que y est atteint

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
        x_reached = abs(self.target_x - self.x) <= self.error_threshold
        y_reached = abs(self.target_y - self.y) <= self.error_threshold
        if x_reached:
            self.x_reached = True
            rospy.loginfo(f"=> Position x atteinte : x={self.x}")
        if y_reached:
            self.y_reached = True
            rospy.loginfo(f"=> Position y atteinte : y={self.y}")
        return x_reached and y_reached

    def calculate_required_torque(self, target, current, previous_error, integral_error):
        current_time = rospy.get_time()
        if self.previous_time is None:
            self.previous_time = current_time

        error = target - current
        delta_time = current_time - self.previous_time
        if delta_time == 0:
            derivative = 0
        else:
            derivative = (error - previous_error) / delta_time
        
        integral_error += error * delta_time
        integral_error = min(max(integral_error, -10), 10)  # Limiter l'accumulation de l'erreur intégrale

        # Petite zone morte pour ignorer les petites erreurs
        if abs(error) < 0.01:
            error = 0

        torque = self.kp_position * error + self.kd_position * derivative + self.ki_position * integral_error
        torque = min(self.max_torque, max(-self.max_torque, torque))

        self.previous_time = current_time
        return torque, error, integral_error

    def apply_torque(self):
        rospy.init_node('step_bot', anonymous=True)
        
        left_torque_pub = rospy.Publisher('/apply_force_left', Wrench, queue_size=10)
        right_torque_pub = rospy.Publisher('/apply_force_right', Wrench, queue_size=10)
        
        rate = rospy.Rate(10)  # 10 Hz

        self.set_robot_position()  # Réinitialiser la position du robot
        rospy.sleep(2)  # Attendre que le repositionnement soit effectif
        self.get_robot_position()  # Vérifier la position initiale

        while not rospy.is_shutdown():
            self.get_robot_position()  # Mettre à jour la position du robot
            
            if self.has_reached_target():
                wrench_left = Wrench()
                wrench_right = Wrench()
                wrench_left.force.x = 0.0
                wrench_right.force.x = 0.0
                wrench_left.force.y = 0.0
                wrench_right.force.y = 0.0
                left_torque_pub.publish(wrench_left)
                right_torque_pub.publish(wrench_right)
                break
            
            torque_x, self.previous_error_x, self.integral_error_x = self.calculate_required_torque(
                self.target_x, self.x, self.previous_error_x, self.integral_error_x)
            torque_y, self.previous_error_y, self.integral_error_y = self.calculate_required_torque(
                self.target_y, self.y, self.previous_error_y, self.integral_error_y)

            required_torque_angle = 0  # Ajuster l'angle au besoin

            wrench_left = Wrench()
            wrench_right = Wrench()

            # Appliquer les couples pour la rotation et la translation
            wrench_left.force.x = (torque_x - required_torque_angle) / 2
            wrench_right.force.x = (torque_x + required_torque_angle) / 2
            wrench_left.force.y = (torque_y - required_torque_angle) / 2
            wrench_right.force.y = (torque_y + required_torque_angle) / 2
            
            left_torque_pub.publish(wrench_left)
            right_torque_pub.publish(wrench_right)
            
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
        controller.set_target_point(3.0,2.0)  # Exemple de point cible
        controller.apply_torque()
    except rospy.ROSInterruptException:
        pass

