#!/usr/bin/env python3
"""
Simple Robot Simulator

Simulates a differential drive robot for MPPI testing.
Subscribes to /cmd_vel and publishes /odom.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header

from mppi_controller.models.kinematic.differential_drive_kinematic import (
    DifferentialDriveKinematic,
)


class SimpleRobotSimulator(Node):
    """
    Simple robot simulator for MPPI testing

    Subscribes:
        /cmd_vel (geometry_msgs/Twist): Control commands

    Publishes:
        /odom (nav_msgs/Odometry): Simulated odometry
    """

    def __init__(self):
        super().__init__('simple_robot_simulator')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.process_noise_std = self.get_parameter('process_noise_std').value

        # Robot model
        v_max = self.get_parameter('v_max').value
        omega_max = self.get_parameter('omega_max').value
        self.model = DifferentialDriveKinematic(v_max=v_max, omega_max=omega_max)

        # State [x, y, theta]
        initial_x = self.get_parameter('initial_x').value
        initial_y = self.get_parameter('initial_y').value
        initial_theta = self.get_parameter('initial_theta').value
        self.state = np.array([initial_x, initial_y, initial_theta])

        # Control
        self.current_control = np.array([0.0, 0.0])

        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            qos_profile
        )

        # Publisher
        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            qos_profile
        )

        # Simulation timer
        self.dt = 1.0 / self.publish_rate
        self.sim_timer = self.create_timer(
            self.dt,
            self.simulation_step
        )

        self.get_logger().info('Simple Robot Simulator initialized')
        self.get_logger().info(f'  Initial state: x={initial_x:.2f}, y={initial_y:.2f}, θ={initial_theta:.2f}')
        self.get_logger().info(f'  Publish rate: {self.publish_rate} Hz')

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('process_noise_std', 0.0)  # Process noise

        # Robot parameters
        self.declare_parameter('v_max', 2.0)
        self.declare_parameter('omega_max', 2.0)

        # Initial state
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)

    def cmd_vel_callback(self, msg: Twist):
        """Control command callback"""
        self.current_control[0] = msg.linear.x
        self.current_control[1] = msg.angular.z

    def simulation_step(self):
        """Simulate one timestep"""
        # Update state
        self.state = self.model.step(self.state, self.current_control, self.dt)

        # Add process noise
        if self.process_noise_std > 0:
            noise = np.random.normal(0, self.process_noise_std, 3)
            self.state += noise

        # Publish odometry
        self.publish_odometry()

    def publish_odometry(self):
        """Publish odometry message"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        # Position
        msg.pose.pose.position.x = float(self.state[0])
        msg.pose.pose.position.y = float(self.state[1])
        msg.pose.pose.position.z = 0.0

        # Orientation (theta to quaternion)
        theta = self.state[2]
        msg.pose.pose.orientation = self._theta_to_quaternion(theta)

        # Velocity (in body frame)
        v = self.current_control[0]
        omega = self.current_control[1]
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = float(omega)

        self.odom_pub.publish(msg)

    def _theta_to_quaternion(self, theta: float) -> Quaternion:
        """Convert theta to quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = float(np.sin(theta / 2.0))
        q.w = float(np.cos(theta / 2.0))
        return q


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRobotSimulator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
