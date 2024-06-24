#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import Imu, NavSatFix
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
        self.ins_sub = rospy.Subscriber('/ins', NavSatFix, self.ins_callback)

        self.orientation = None
        self.roll = 0.0
        self.pitch = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.ins_x = 0.0
        self.ins_y = 0.0
        self.ins_theta = 0.0

        self.imu_x = 0.0
        self.imu_y = 0.0
        self.imu_theta = 0.0

        self.prev_imu_time = None

        self.seq_num = 0  # Sequence number for logging

        # PID coefficients
        self.kp = 0.5
        self.ki = 0.01
        self.kd = 0.1

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

        if self.prev_imu_time is None:
            self.prev_imu_time = rospy.Time.now()
            return

        current_time = rospy.Time.now()
        dt = (current_time - self.prev_imu_time).to_sec()
        self.prev_imu_time = current_time

        # Integrate the linear acceleration to get position
        self.imu_x += data.linear_acceleration.x * dt**2 / 2
        self.imu_y += data.linear_acceleration.y * dt**2 / 2
        self.imu_theta += data.angular_velocity.z * dt

        if self.seq_num % 10 == 0:
            rospy.loginfo(f"[Seq: {self.seq_num}] IMU raw data - Orientation: x: {data.orientation.x}, y: {data.orientation.y}, z: {data.orientation.z}, w: {data.orientation.w}, "
                          f"Linear Acc: x: {data.linear_acceleration.x}, y: {data.linear_acceleration.y}, z: {data.linear_acceleration.z}, "
                          f"Angular Vel: x: {data.angular_velocity.x}, y: {data.angular_velocity.y}, z: {data.angular_velocity.z}")
            rospy.loginfo(f"[Seq: {self.seq_num}] IMU converted data - x: {self.imu_x}, y: {self.imu_y}, theta: {self.imu_theta}")

    def ins_callback(self, data):
        # Assuming a local conversion from lat/lon to meters
        lat = data.latitude
        lon = data.longitude

        # Convert latitude and longitude to meters (simple approximation)
        R = 6371000  # Radius of Earth in meters
        lat0 = 0  # Reference latitude, set as initial latitude
        lon0 = 0  # Reference longitude, set as initial longitude

        self.ins_x = R * (lat - lat0) * math.pi / 180
        self.ins_y = R * (lon - lon0) * math.cos(lat * math.pi / 180) * math.pi / 180
        self.ins_theta = 0  # Update this based on your INS data

        if self.seq_num % 10 == 0:
            rospy.loginfo(f"[Seq: {self.seq_num}] INS raw data - Latitude: {lat}, Longitude: {lon}, Altitude: {data.altitude}")
            rospy.loginfo(f"[Seq: {self.seq_num}] INS converted data - x: {self.ins_x}, y: {self.ins_y}, theta: {self.ins_theta}")

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
            ins_distance = math.sqrt((target_x - self.ins_x)**2 + (target_y - self.ins_y)**2)
            imu_distance = math.sqrt((target_x - self.imu_x)**2 + (target_y - self.imu_y)**2)
            rospy.loginfo(f"[Seq: {self.seq_num}] => Distance to target (Gazebo): {gazebo_distance}, (INS): {ins_distance}, (IMU): {imu_distance}, Threshold: 0.1")

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
            self.ins_callback(NavSatFix())  # Simulate INS callback
            self.imu_callback(Imu())  # Simulate IMU callback

            distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
            ins_distance = math.sqrt((target_x - self.ins_x)**2 + (target_y - self.ins_y)**2)
            imu_distance = math.sqrt((target_x - self.imu_x)**2 + (target_y - self.imu_y)**2)

            self.log_positions(target_x, target_y)

            if distance < 0.1:
                rospy.loginfo(f"[Seq: {self.seq_num}] *** Target reached (Gazebo): x={self.x}, y={self.y}")
                self.apply_reverse_torque()
                break

            if ins_distance < 0.1:
                rospy.loginfo(f"[Seq: {self.seq_num}] *** Target reached (INS): x={self.ins_x}, y={self.ins_y}")

            if imu_distance < 0.1:
                rospy.loginfo(f"[Seq: {self.seq_num}] *** Target reached (IMU): x={self.imu_x}, y={self.imu_y}")

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
                torque = max(0.1, torque * distance)
                torque = min(torque, 2)

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

    def apply_reverse_torque(self):
        rate = rospy.Rate(20)
        for _ in range(10):
            wrench = Wrench()
            wrench.force.x = -0.05
            wrench.force.y = -0.05
            self.left_torque_pub.publish(wrench)
            self.right_torque_pub.publish(wrench)
            rospy.loginfo(f"[Seq: {self.seq_num}] => Envoi de couple inverse - x={wrench.force.x}, y={wrench.force.y}")
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotController()
        controller.move_to_target(0.0, 3.0)  # Déplacement vers (0,3)
        controller.move_to_target(1.0, 3.0)  # Déplacement vers (1,3)
        controller.move_to_target(1.0, 0.0)  # Déplacement vers (1,0)
        controller.move_to_target(2.0, 0.0)  # Déplacement vers (2,0)
        controller.move_to_target(2.0, 3.0)  # Déplacement vers (2,3)
        controller.move_to_target(3.0, 3.0)  # Déplacement vers (3,3)
        controller.move_to_target(3.0, 0.0)  # Déplacement vers (3,0)
    except rospy.ROSInterruptException:
        pass

