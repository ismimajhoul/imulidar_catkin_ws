#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class RobotController:
    def __init__(self, robot_name='turtlebot3_burger', reference_frame='world'):
        self.robot_name = robot_name
        self.reference_frame = reference_frame
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Subscribers
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        
        # Initial state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.target_x = 3.0  # Target point in x
        self.target_y = 3.0  # Target point in y
        self.error_threshold = 0.1  # Error threshold to consider a point as reached
        self.max_torque = 1.0  # Maximum torque to apply to limit speed
        self.kp_position = 0.5  # Proportional coefficient for position
        self.kd_position = 0.1  # Derivative coefficient for position
        self.kp_angle = 1.0  # Proportional coefficient for angle
        self.kd_angle = 0.1  # Derivative coefficient for angle
        self.previous_error = 0.0
        self.previous_time = None
        self.phase = 'x'  # Start by regulating x

    def imu_callback(self, data):
        orientation_q = data.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)
        rospy.loginfo(f"*** Orientation mise à jour depuis l'IMU - theta: {self.theta}")

    def get_robot_position(self):
        try:
            state = self.get_state_service(self.robot_name, self.reference_frame)
            if state.success:
                self.x = state.pose.position.x
                self.y = state.pose.position.y
                rospy.loginfo(f"*** Position du robot mise à jour - x: {self.x}, y: {self.y}, theta: {self.theta}")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def verify_robot_position(self, expected_x, expected_y, expected_theta):
        self.get_robot_position()  # Mettre à jour la position du robot
        position_correcte = (
            abs(self.x - expected_x) < self.error_threshold and
            abs(self.y - expected_y) < self.error_threshold and
            abs(self.theta - expected_theta) < self.error_threshold
        )
        if position_correcte:
            rospy.loginfo(f"=> Robot correctement repositionné: x={self.x}, y={self.y}, theta={self.theta}")
        else:
            rospy.logwarn(f"=> Erreur de repositionnement: x={self.x}, y={self.y}, theta={self.theta}")
        return position_correcte

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
            
            rospy.sleep(1)  # Attendre que le repositionnement soit effectif
            if not self.verify_robot_position(0.0, 0.0, 0.0):
                rospy.logerr("Le repositionnement du robot a échoué.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def set_target_point(self, new_x, new_y):
        self.target_x = new_x
        self.target_y = new_y
        rospy.loginfo(f"=> Nouveau point cible défini - x: {self.target_x}, y: {self.target_y}")

    def has_reached_target(self):
        if self.phase == 'x':
            return abs(self.target_x - self.x) < self.error_threshold
        elif self.phase == 'y':
            return abs(self.target_y - self.y) < self.error_threshold

    def calculate_required_torque(self):
        current_time = rospy.get_time()
        if self.previous_time is None:
            self.previous_time = current_time

        # Calcul de l'erreur de position
        if self.phase == 'x':
            error = self.target_x - self.x
        elif self.phase == 'y':
            error = self.target_y - self.y

        # Calcul de la dérivée de l'erreur de position
        delta_time = current_time - self.previous_time
        if delta_time == 0:
            derivative = 0
        else:
            derivative = (error - self.previous_error) / delta_time

        # Calcul de l'erreur d'angle
        desired_theta = math.atan2(self.target_y - self.y, self.target_x - self.x)
        error_angle = desired_theta - self.theta
        error_angle = math.atan2(math.sin(error_angle), math.cos(error_angle))  # Normalisation de l'angle

        # Calcul de la dérivée de l'erreur d'angle
        if delta_time == 0:
            derivative_angle = 0
        else:
            derivative_angle = (error_angle - self.previous_error) / delta_time

        # Calcul du couple requis
        torque = self.kp_position * error + self.kd_position * derivative
        torque_angle = self.kp_angle * error_angle + self.kd_angle * derivative_angle

        torque = min(self.max_torque, max(-self.max_torque, torque))  # Limiter le couple par max_torque
        torque_angle = min(self.max_torque, max(-self.max_torque, torque_angle))

        # Mise à jour des variables pour la prochaine itération
        self.previous_error = error
        self.previous_time = current_time

        return torque, torque_angle

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
                if self.phase == 'x':
                    rospy.loginfo(f"=> Position x atteinte : x={self.x}")
                    self.phase = 'y'  # Switch to regulating y
                    # Immobiliser le mouvement en x
                    wrench_left = Wrench()
                    wrench_right = Wrench()
                    wrench_left.force.x = 0.0
                    wrench_right.force.x = 0.0
                    left_torque_pub.publish(wrench_left)
                    right_torque_pub.publish(wrench_right)
                    rospy.sleep(1)  # Pause pour stabiliser
                elif self.phase == 'y':
                    rospy.loginfo(f"=> Position y atteinte : y={self.y}")
                    # Immobiliser le robot
                    wrench_left = Wrench()
                    wrench_right = Wrench()
                    wrench_left.force.x = 0.0
                    wrench_right.force.x = 0.0
                    left_torque_pub.publish(wrench_left)
                    right_torque_pub.publish(wrench_right)
                    break
            
            required_torque, required_torque_angle = self.calculate_required_torque()
            
            wrench_left = Wrench()
            wrench_right = Wrench()
            
            # Appliquer les couples pour la rotation et la translation
            if self.phase == 'x':
                wrench_left.force.x = required_torque - required_torque_angle
                wrench_right.force.x = required_torque + required_torque_angle
            elif self.phase == 'y':
                wrench_left.force.x = -required_torque_angle
                wrench_right.force.x = required_torque_angle
            
            left_torque_pub.publish(wrench_left)
            right_torque_pub.publish(wrench_right)
            
            rospy.loginfo(f"*** Position actuelle - x: {self.x}, y: {self.y}, theta: {self.theta}")
            rospy.loginfo(f"=> Position désirée - x_d: {self.target_x}, y_d: {self.target_y}")
            rospy.loginfo(f"=> Envoi de couple - Left: {wrench_left.force.x}, Right: {wrench_right.force.x}")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.set_target_point(3.0, 3.0)  # Exemple de point cible
        controller.apply_torque()
    except rospy.ROSInterruptException:
        pass

