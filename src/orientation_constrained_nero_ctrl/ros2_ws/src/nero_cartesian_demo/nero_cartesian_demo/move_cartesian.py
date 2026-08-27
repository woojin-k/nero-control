import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory


class NeroMove(Node):
    def __init__(self):
        super().__init__('nero_move')

        self.path_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path'
        )

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            '/execute_trajectory'
        )

    def run(self):
        target = Pose()
        target.position.x = 0.003     # -0.017 + 0.020
        target.position.y = 0.227
        target.position.z = 0.562

        target.orientation.x = 0.507
        target.orientation.y = -0.492
        target.orientation.z = 0.492
        target.orientation.w = 0.507

        request = GetCartesianPath.Request()

        request.header.frame_id = 'base_link'
        request.start_state.is_diff = True

        request.group_name = 'arm'
        request.link_name = 'tcp_link'

        request.waypoints = [target]

        # 5 mm 간격
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True

        self.get_logger().info('Planning...')

        self.path_client.wait_for_service()

        future = self.path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response is None:
            self.get_logger().error('Planning failed')
            return

        self.get_logger().info(
            f'Path completed: {response.fraction * 100:.1f}%'
        )

        if response.fraction < 0.999:
            self.get_logger().error('Complete path not found')
            return

        # 계산된 trajectory 실행
        self.execute_client.wait_for_server()

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution

        future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Execution rejected')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info('Done')


def main(args=None):
    rclpy.init(args=args)

    node = NeroMove()
    node.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()