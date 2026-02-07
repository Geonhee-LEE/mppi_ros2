#!/usr/bin/env python3
"""
MPPI Visualizer ROS2 Node

Visualizes MPPI controller information in RVIZ.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Optional

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class MPPIVisualizerNode(Node):
    """
    ROS2 Node for MPPI Visualization

    Subscribes:
        /mppi/sample_trajectories (custom): Sample trajectories
        /mppi/predicted_path (nav_msgs/Path): Predicted trajectory
        /reference_path (nav_msgs/Path): Reference trajectory

    Publishes:
        /mppi/visualization (visualization_msgs/MarkerArray): RVIZ markers
    """

    def __init__(self):
        super().__init__('mppi_visualizer_node')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self.frame_id = self.get_parameter('frame_id').value
        self.show_samples = self.get_parameter('show_samples').value
        self.show_best = self.get_parameter('show_best').value
        self.show_reference = self.get_parameter('show_reference').value
        self.max_samples_viz = self.get_parameter('max_samples_viz').value

        # State
        self.sample_trajectories: Optional[np.ndarray] = None
        self.sample_weights: Optional[np.ndarray] = None
        self.predicted_path: Optional[Path] = None
        self.reference_path: Optional[Path] = None

        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/mppi/visualization',
            qos_profile
        )

        # Visualization timer
        viz_rate = self.get_parameter('visualization_rate').value
        self.viz_timer = self.create_timer(
            1.0 / viz_rate,
            self.visualization_callback
        )

        # Subscribers would go here if we had custom messages
        # For now, we'll simulate with timer

        self.get_logger().info('MPPI Visualizer Node initialized')

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('visualization_rate', 10.0)  # Hz
        self.declare_parameter('show_samples', True)
        self.declare_parameter('show_best', True)
        self.declare_parameter('show_reference', True)
        self.declare_parameter('max_samples_viz', 100)

    def visualization_callback(self):
        """Publish visualization markers"""
        markers = MarkerArray()

        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        marker_id = 0

        # 1. Sample trajectories (with transparency based on weight)
        if self.show_samples and self.sample_trajectories is not None:
            sample_markers = self._create_sample_trajectory_markers(
                self.sample_trajectories,
                self.sample_weights,
                marker_id
            )
            markers.markers.extend(sample_markers)
            marker_id += len(sample_markers)

        # 2. Best/Predicted trajectory
        if self.show_best and self.predicted_path is not None:
            best_marker = self._create_path_marker(
                self.predicted_path,
                marker_id,
                color=ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),  # Cyan
                scale=0.05,
                ns='predicted_path'
            )
            markers.markers.append(best_marker)
            marker_id += 1

        # 3. Reference trajectory
        if self.show_reference and self.reference_path is not None:
            ref_marker = self._create_path_marker(
                self.reference_path,
                marker_id,
                color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5),  # Red
                scale=0.03,
                ns='reference_path'
            )
            markers.markers.append(ref_marker)
            marker_id += 1

        self.marker_pub.publish(markers)

    def _create_sample_trajectory_markers(
        self,
        trajectories: np.ndarray,
        weights: Optional[np.ndarray],
        start_id: int
    ) -> list:
        """
        Create markers for sample trajectories

        Args:
            trajectories: (K, N+1, nx)
            weights: (K,) - normalized weights
            start_id: Starting marker ID

        Returns:
            List of Marker
        """
        markers = []
        K, N_plus_1, nx = trajectories.shape

        # Limit number of samples to visualize
        num_viz = min(K, self.max_samples_viz)
        if num_viz < K:
            # Sample uniformly
            indices = np.linspace(0, K-1, num_viz, dtype=int)
            trajectories = trajectories[indices]
            if weights is not None:
                weights = weights[indices]
                weights = weights / weights.sum()  # Renormalize

        for k in range(num_viz):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'sample_trajectories'
            marker.id = start_id + k
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Scale
            marker.scale.x = 0.01

            # Color with transparency based on weight
            if weights is not None:
                alpha = float(weights[k] * 5.0)  # Scale up for visibility
                alpha = min(1.0, alpha)
            else:
                alpha = 0.3

            marker.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=alpha)

            # Points
            for t in range(N_plus_1):
                point = Point()
                point.x = float(trajectories[k, t, 0])
                point.y = float(trajectories[k, t, 1])
                point.z = 0.0
                marker.points.append(point)

            markers.append(marker)

        return markers

    def _create_path_marker(
        self,
        path: Path,
        marker_id: int,
        color: ColorRGBA,
        scale: float,
        ns: str
    ) -> Marker:
        """Create marker for a path"""
        marker = Marker()
        marker.header = path.header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = scale
        marker.color = color

        for pose_stamped in path.poses:
            point = Point()
            point.x = pose_stamped.pose.position.x
            point.y = pose_stamped.pose.position.y
            point.z = pose_stamped.pose.position.z
            marker.points.append(point)

        return marker

    def update_sample_trajectories(
        self,
        trajectories: np.ndarray,
        weights: np.ndarray
    ):
        """Update sample trajectories (called from controller)"""
        self.sample_trajectories = trajectories
        self.sample_weights = weights

    def update_predicted_path(self, path: Path):
        """Update predicted path"""
        self.predicted_path = path

    def update_reference_path(self, path: Path):
        """Update reference path"""
        self.reference_path = path


def main(args=None):
    rclpy.init(args=args)
    node = MPPIVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
