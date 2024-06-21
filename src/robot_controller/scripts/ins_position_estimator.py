#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetWorldProperties
import tf

class RobotStateModifier:
    def __init__(self):
        rospy.init_node('robot_state_modifier', anonymous=True)
        rospy.loginfo("Robot State Modifier Node Initialized")
        
        # Subscribe to IMU and INS topics
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.ins_sub = rospy.Subscriber('/ins', Imu, self.ins_callback)
        
        self.imu_data = None
        self.ins_data = None
        
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        rospy.wait_for_service('/gazebo/get_world_properties')
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        self.rate = rospy.Rate(1)  # 1 Hz

    def imu_callback(self, data):
        self.imu_data = data

    def ins_callback(self, data):
        self.ins_data = data

    def check_model_exists(self, model_name):
        try:
            world_properties = self.get_world_properties()
            rospy.loginfo(f"Models in the world: {world_properties.model_names}")
            return model_name in world_properties.model_names
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def set_robot_position(self, x, y, z, yaw):
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_burger'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        state_msg.pose.orientation.x = quaternion[0]
        state_msg.pose.orientation.y = quaternion[1]
        state_msg.pose.orientation.z = quaternion[2]
        state_msg.pose.orientation.w = quaternion[3]

        try:
            self.set_model_state(state_msg)
            rospy.loginfo(f"Robot position set to x: {x}, y: {y}, z: {z}, yaw: {yaw}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def get_robot_position(self):
        try:
            state = self.get_model_state('turtlebot3_burger', '')
            if state.success:
                return state.pose.position.x, state.pose.position.y, state.pose.position.z, state.pose.orientation
            else:
                rospy.logerr("Failed to get robot position")
                return None, None, None, None
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None, None, None, None

    def run(self):
        if not self.check_model_exists('turtlebot3_burger'):
            rospy.logerr("Model 'turtlebot3_burger' does not exist in the world")
            return

        positions = [(1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0)]
        for pos in positions:
            self.set_robot_position(*pos)
            rospy.sleep(2)  # Wait for the position to be set
            
            x, y, z, orientation = self.get_robot_position()
            rospy.loginfo(f"Current Position - x: {x}, y: {y}, z: {z}, orientation: {orientation}")

            if self.imu_data:
                rospy.loginfo(f"IMU Data - Orientation: {self.imu_data.orientation}, Angular Velocity: {self.imu_data.angular_velocity}, Linear Acceleration: {self.imu_data.linear_acceleration}")
            
            if self.ins_data:
                rospy.loginfo(f"INS Data - Orientation: {self.ins_data.orientation}, Angular Velocity: {self.ins_data.angular_velocity}, Linear Acceleration: {self.ins_data.linear_acceleration}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        robot_state_modifier = RobotStateModifier()
        robot_state_modifier.run()
    except rospy.ROSInterruptException:
        pass

