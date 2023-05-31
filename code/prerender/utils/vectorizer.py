from abc import ABC, abstractmethod
import numpy as np
from .utils import filter_valid, get_filter_valid_roadnetwork_keys, get_filter_valid_anget_history


class Renderer(ABC):
    @abstractmethod
    def render(self, data):
        pass


class SegmentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _select_n_closest_segments(self, segments, types):
        """
        distances: Compute the Euclidean distance of each road segment from the origin (0, 0)
        n_closest_segments_ids: Find the indices of the n_closest_segments segments with the smallest distances from the origin
        """
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        n_closest_segments_ids = np.argpartition(
            distances, self._config["n_closest_segments"])[:self._config["n_closest_segments"]]
        return segments[n_closest_segments_ids], types[n_closest_segments_ids].flatten()

    def _select_segments_within_radius(self, segments, types):
        """
        Select segments within a range of a certain radius
        """
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        closest_segments_selector = distances < self._config["segments_filtering_radius"]
        return segments[closest_segments_selector], types[closest_segments_selector].flatten()

    def filter(self, segments, types):
        if self._config["policy"] == "n_closest_segments":
            return self._select_n_closest_segments(segments, types)
        if self._config["policy"] == "within_radius":
            return self._select_segments_within_radius(segments, types)
        raise Exception(f"Unknown segment filtering policy {self._config['policy']}")


class TargetAgentFilteringPolicy:
    """
    A filtering policy used to select agents from the data based on certain criteria. This is particularly useful when working with large datasets, as it allows you to focu
    """

    def __init__(self, config):
        self._config = config

    def _get_only_interesting_agents(self, data, i):
        """add an additional condition to select target agents that are pedestrians only (type == 2)"""
        return data["state/tracks_to_predict"][i] > 0 and data["state/type"][i] == 2

    def _get_only_fully_available_agents(self, data, i):
        """
        full_validity:
        The method first concatenates the validity arrays for past, current, and future states along the last axis.
        Each of these validity arrays is a binary array where 1 represents a valid state at that timestamp, and 0 represents an invalid state.
        By concatenating these arrays, we create a combined array representing the validity of the agent's states across all timestamps.

        n_timestamps:
        The total number of timestamps is calculated by getting the shape of the full_validity array along the last axis.
        This gives us the total number of time steps (past, current, and future) for which we have information.

        n_valid_timestamps:
        To count the number of valid timestamps for each agent, we sum the full_validity array along the last axis.
        This gives us an array where each element represents the number of valid timestamps for a particular agent.

        return n_valid_timestamps[i] == n_timestamps:
        Finally, the method checks if the number of valid timestamps for the agent at index i is equal to the total number of timestamps (n_timestamps).
        If they are equal, it means that the agent has valid information available for all timestamps, and the method returns True. Otherwise, it returns False.
        """

        """add an additional condition to select target agents that are pedestrians only (type == 2)"""
        full_validity = np.concatenate([
            data["state/past/valid"], data["state/current/valid"], data["state/future/valid"]],
            axis=-1)
        n_timestamps = full_validity.shape[-1]
        n_valid_timestamps = full_validity.sum(axis=-1)
        return n_valid_timestamps[i] == n_timestamps and data["state/type"][i] == 2

    def _get_interesting_and_fully_available_agents(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return interesting and fully_valid

    def _get_fully_available_agents_without_interesting(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return fully_valid and not interesting

    def allow(self, data, i):
        if self._config["policy"] == "interesting":
            return self._get_only_interesting_agents(data, i)
        if self._config["policy"] == "fully_available":
            return self._get_only_fully_available_agents(data, i)
        if self._config["policy"] == "interesting_and_fully_available":
            return self._get_interesting_and_fully_available_agents(data, i)
        if self._config["policy"] == "fully_available_agents_without_interesting":
            return self._get_fully_available_agents_without_interesting(data, i)
        raise Exception(f"Unknown agent filtering policy {self._config['policy']}")


class MultiPathPPRenderer(Renderer):
    def __init__(self, config):
        self._config = config
        self.n_segment_types = 20
        self._segment_filter = SegmentFilteringPolicy(self._config["segment_filtering"])
        self._target_agent_filter = TargetAgentFilteringPolicy(self._config["agent_filtering"])

    def _select_agents_with_any_validity(self, data):
        """filter agents that have at least one valid timestamp in their state history (past, current, or future)"""
        return data["state/current/valid"].sum(axis=-1) + \
            data["state/future/valid"].sum(axis=-1) + data["state/past/valid"].sum(axis=-1)

    def _preprocess_data(self, data):
        """ This function preprocesses the data by filtering out invalid road network samples and agents without any valid state"""
        valid_roadnetwork_selector = data["roadgraph_samples/valid"]  # shape [N, 1], value is either 1 or 0

        for key in get_filter_valid_roadnetwork_keys():
            data[key] = filter_valid(data[key], valid_roadnetwork_selector)

        agents_with_any_validity_selector = self._select_agents_with_any_validity(data)
        for key in get_filter_valid_anget_history():
            data[key] = filter_valid(data[key], agents_with_any_validity_selector)

    def _prepare_roadnetwork_info(self, data):
        """
        This function prepares the road network information by extracting node coordinates, node IDs, and node types.
        It also computes the start and end coordinates of each road segment and their corresponding types.

        """
        # Returns np.array of shape [N, 2, 2]
        # 0 dim: N - number of segments
        # 1 dim: the start and the end of a segment
        # 2 dim: (x, y)
        # and
        # ndarray of segment types
        node_xyz = data["roadgraph_samples/xyz"][:, :2]
        node_id = data["roadgraph_samples/id"].flatten()
        node_type = data["roadgraph_samples/type"]

        # Initialises two empty lists
        # `result` for storing the start and end coordinates of road segments
        # `segment_types` for storing the corresponding road segment types
        result = []
        segment_types = []

        for polyline_id in np.unique(node_id):
            polyline_nodes = node_xyz[node_id == polyline_id]
            polyline_type = node_type[node_id == polyline_id][0]

            # If the polyline consists of only one node, we duplicate it to create a segment
            if len(polyline_nodes) == 1:
                polyline_nodes = np.array([polyline_nodes[0], polyline_nodes[0]])

            # If the configuration contains a "drop_segments" setting, it selects nodes from the polyline at regular intervals as specified by the "drop_segments" value.
            # It ensures that the last node of the polyline is always included.
            if "drop_segments" in self._config:
                selector = np.arange(len(polyline_nodes), step=self._config["drop_segments"])
                if len(polyline_nodes) <= self._config["drop_segments"]:
                    selector = np.array([0, len(polyline_nodes) - 1])
                selector[-1] = len(polyline_nodes) - 1
                polyline_nodes = polyline_nodes[selector]

            # We create a list of start and end coordinates of road segments
            polyline_start_end = np.array(
                [polyline_nodes[:-1], polyline_nodes[1:]]).transpose(1, 0, 2)
            result.append(polyline_start_end)

            # We create a list of road segment types
            segment_types.extend([polyline_type] * len(polyline_start_end))

        result = np.concatenate(result, axis=0)
        assert len(segment_types) == len(result), \
            f"Number of segments {len(result)} doen't match the number of types {len(segment_types)}"
        return {
            "segments": result,
            "segment_types": np.array(segment_types)}

    def _split_past_and_future(self, data, key):
        """splits the data associated with the given key into two separate parts: past (including current) and future."""
        history = np.concatenate(  # adds an extra dimension to the history array using [..., None]
            [data[f"state/past/{key}"], data[f"state/current/{key}"]], axis=1)[..., None]
        future = data[f"state/future/{key}"][..., None]
        return history, future

    def _prepare_agent_history(self, data):
        # (n_agents, 11, 2)
        preprocessed_data = {}
        preprocessed_data["history/xy"] = np.array([
            np.concatenate([data["state/past/x"], data["state/current/x"]], axis=1),
            np.concatenate([data["state/past/y"], data["state/current/y"]], axis=1)
        ]).transpose(1, 2, 0)

        # (n_agents, 80, 2)
        preprocessed_data["future/xy"] = np.array(
            [data["state/future/x"], data["state/future/y"]]).transpose(1, 2, 0)

        # (n_agents, 11, 1)
        for key in ["speed", "bbox_yaw", "valid"]:
            preprocessed_data[f"history/{key}"], preprocessed_data[f"future/{key}"] = \
                self._split_past_and_future(data, key)

        for key in ["state/id", "state/is_sdc", "state/type", "state/current/width",
                    "state/current/length"]:
            # key.split('/')[-1] = id, is_sdc, type, width, length
            preprocessed_data[key.split('/')[-1]] = data[key]
        preprocessed_data["scenario_id"] = data["scenario/id"]
        return preprocessed_data

    def _transfrom_to_agent_coordinate_system(self, coordinates, shift, yaw):
        """
        Transforms the input coordinates from the global coordinate system to the local coordinate system of an agent
        This transformation is useful for rendering the scene from the perspective of a specific agent, taking into account the agent's position and orientation.
        """
        # coordinates
        # dim 0: number of agents / number of segments for road network
        # dim 1: number of history points / (start_point, end_point) for segments
        # dim 2: x, y

        # Negates the yaw angle since the rotation will be performed in the opposite direction (from global to local coordinate system).
        yaw = -yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s), (s, c))).reshape(2, 2)
        transformed = np.matmul((coordinates - shift), R.T)
        return transformed

    def _filter_closest_segments(self, segments, types):
        # This method works only with road segments in agent-related coordinate system
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        assert len(segments) == len(types), \
            f"n_segments={len(segments)} must match len_types={len(types)}"
        return self._segment_filter.filter(segments, types)

    def _compute_closest_point_of_segment(self, segments):
        # This method works only with road segments in agent-related coordinate system
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2

        # The start points A and end points B of the segments are extracted from the input array.
        A, B = segments[:, 0, :], segments[:, 1, :]

        # The vector M representing the direction and length of each segment is computed as the difference between B and A.
        M = B - A

        # The scalar t is calculated as the projection of the negative start point -A onto the segment direction M,
        # normalized by the length of the segment.
        # The addition of a small constant 1e-6 in the denominator is to avoid division by zero.
        t = (-A * M).sum(axis=-1) / ((M * M).sum(axis=-1) + 1e-6)

        # The scalar t is clipped between 0 and 1 to ensure that the closest point lies within the segment, not beyond its start or end points
        clipped_t = np.clip(t, 0, 1)[:, None]
        closest_points = A + clipped_t * M
        return closest_points

    def _generate_segment_embeddings(self, segments, types):
        """
        Computes an embedding for each road segment based on various geometric properties and the segment type
        """
        # This method works only with road segments in agent-related coordinate system
        # previously filtered
        closest_points = self._compute_closest_point_of_segment(segments)

        # Compute the norm (length) of the vector connecting the origin to the closest points
        r_norm = np.linalg.norm(closest_points, axis=-1, keepdims=True)

        # Calculate the unit vector
        r_unit_vector = closest_points / (r_norm + 1e-6)

        # Compute the unit vector of the segment direction
        segment_end_minus_start = segments[:, 1, :] - segments[:, 0, :]
        segment_end_minus_start_norm = np.linalg.norm(
            segment_end_minus_start, axis=-1, keepdims=True)
        segment_unit_vector = segment_end_minus_start / (segment_end_minus_start_norm + 1e-6)

        # Compute the norm of the vector connecting the closest points to the end points of the segments
        segment_end_minus_r_norm = np.linalg.norm(
            segments[:, 1, :] - closest_points, axis=-1, keepdims=True)

        # TODO: I made a modification here, but I am not sure if this is correct
        # segment_type_ohe = np.eye(self.n_segment_types)[types]
        segment_type_ohe = np.eye(self.n_segment_types)[types-1]


        # Concatenate all the computed features to form the final embedding
        resulting_embeddings = np.concatenate([
            r_norm, r_unit_vector, segment_unit_vector, segment_end_minus_start_norm,
            segment_end_minus_r_norm, segment_type_ohe], axis=-1)
        return resulting_embeddings[:, None, :]

    def _split_target_agent_and_other_agents(self, tensor, i, key):
        target_data = tensor[i][None,]
        other_data = np.delete(tensor, i, axis=0)
        return target_data, other_data

    def _get_trajectory_class(self, data):
        """
        Takes the data object and classifies the trajectory of the target agent based on its motion characteristics.
        It does this by analyzing the validity, position, yaw, and speed of the target agent in the past and future time steps

        First, it concatenates the valid state, x and y positions, yaw, and speed of the target agent for the last history step and all future steps.
        It then defines several constants representing thresholds for different motion types:

        kMaxSpeedForStationary: maximum speed (m/s) for an agent to be considered stationary
        kMaxDisplacementForStationary: maximum displacement (m) for an agent to be considered stationary
        kMaxLateralDisplacementForStraight: maximum lateral displacement (m) for an agent to be considered moving straight
        kMinLongitudinalDisplacementForUTurn: minimum longitudinal displacement (m) for an agent to be considered making a U-turn
        kMaxAbsHeadingDiffForStraight: maximum absolute heading difference (rad) for an agent to be considered moving straight

        The function then iterates through the valid states and finds the first and last valid indices.
        If the first state is not valid or the last valid index is not found, the function returns None.
        Next, it calculates the change in position (xy_delta), final displacement, change in heading (heading_delta), and maximum speed between the first and last valid indices.
        Finally, the function classifies the motion based on the calculated values and the defined thresholds, returning one of the following labels:

        "stationary"
        "straight"
        "straight_right"
        "straight_left"
        "right_u_turn"
        "right_turn"
        "left_u_turn"
        "left_turn"

        This classification helps to understand the movement pattern of the target agent in the given scenario.
        """
        valid = np.concatenate(
            [data["target/history/valid"][0, -1:, 0], data["target/future/valid"][0, :, 0]])
        future_xy = np.concatenate(
            [data["target/history/xy"][0, -1:, :], data["target/future/xy"][0, :, :]])
        future_yaw = np.concatenate(
            [data["target/history/yaw"][0, -1:, 0], data["target/future/yaw"][0, :, 0]])
        future_speed = np.concatenate(
            [data["target/history/speed"][0, -1:, 0], data["target/future/speed"][0, :, 0]])

        kMaxSpeedForStationary = 2.0  # (m/s)
        kMaxDisplacementForStationary = 5.0  # (m)
        kMaxLateralDisplacementForStraight = 5.0  # (m)
        kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
        kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

        first_valid_index, last_valid_index = 0, None
        """
        It initializes a variable last_valid_index as None.
        It iterates through the valid array, starting from index 1 (skipping the first element) up to the end of the array.
        If it finds an element with a value of 1 (indicating that the corresponding data point is valid), it updates the last_valid_index to the current index.
        After iterating through the entire array, it checks whether the first element of the valid array is 0 (indicating that the first data point is not valid) or if last_valid_index is still None (indicating that no valid data points were found).
        If either of these conditions is true, the function returns None, indicating that it's not possible to determine the trajectory class for this agent, as there are no valid data points to analyze.
        """
        for i in range(1, len(valid)):
            if valid[i] == 1:
                last_valid_index = i
        if valid[first_valid_index] == 0 or last_valid_index is None:
            return None

        xy_delta = future_xy[last_valid_index] - future_xy[first_valid_index]
        final_displacement = np.linalg.norm(xy_delta)
        heading_delta = future_yaw[last_valid_index] - future_yaw[first_valid_index]
        max_speed = max(future_speed[last_valid_index], future_speed[first_valid_index])

        if max_speed < kMaxSpeedForStationary and \
                final_displacement < kMaxDisplacementForStationary:
            return "stationary"
        if np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight:
            if np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight:
                return "straight"
            return "straight_right" if xy_delta[1] < 0 else "straight_left"
        if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]:
            return "right_u_turn" if xy_delta[0] < kMinLongitudinalDisplacementForUTurn \
                else "right_turn"
        if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
            return "left_u_turn"
        return "left_turn"

    def render(self, data):
        array_of_scene_data_dicts = []
        self._preprocess_data(data)
        road_network_info = self._prepare_roadnetwork_info(data)
        agent_history_info = self._prepare_agent_history(data)
        for i in range(agent_history_info["history/xy"].shape[0]):
            if not self._target_agent_filter.allow(data, i):
                continue
            current_agent_scene_shift = agent_history_info["history/xy"][i][-1]
            current_agent_scene_yaw = agent_history_info["history/bbox_yaw"][i][-1]

            current_scene_road_network_coordinates = self._transfrom_to_agent_coordinate_system(
                road_network_info["segments"],
                current_agent_scene_shift,
                current_agent_scene_yaw
            )
            current_scene_road_network_coordinates, current_scene_road_network_types = self._filter_closest_segments(
                    current_scene_road_network_coordinates,
                    road_network_info["segment_types"]
            )

            road_segments_embeddings = self._generate_segment_embeddings(
                current_scene_road_network_coordinates,
                current_scene_road_network_types
            )

            current_scene_agents_coordinates_history = self._transfrom_to_agent_coordinate_system(
                agent_history_info["history/xy"],
                current_agent_scene_shift,
                current_agent_scene_yaw
            )
            current_scene_agents_coordinates_future = self._transfrom_to_agent_coordinate_system(
                agent_history_info["future/xy"],
                current_agent_scene_shift,
                current_agent_scene_yaw
            )

            current_scene_agents_yaws_history = agent_history_info["history/bbox_yaw"] - current_agent_scene_yaw
            current_scene_agents_yaws_future = agent_history_info["future/bbox_yaw"] - current_agent_scene_yaw

            (current_scene_target_agent_coordinates_history,
             current_scene_other_agents_coordinates_history) = self._split_target_agent_and_other_agents(
                current_scene_agents_coordinates_history, i, "xy"
            )
            (current_scene_target_agent_yaws_history,
             current_scene_other_agents_yaws_history) = self._split_target_agent_and_other_agents(
                current_scene_agents_yaws_history, i, "yaw"
            )
            (current_scene_target_agent_speed_history,
             current_scene_other_agents_speed_history) = self._split_target_agent_and_other_agents(
                agent_history_info["history/speed"], i, "speed"
            )

            scene_data = {
                "shift": current_agent_scene_shift[None,],
                "yaw": current_agent_scene_yaw,
                "scenario_id": agent_history_info["scenario_id"].item().decode("utf-8"),
                "agent_id": int(agent_history_info["id"][i]),
                "target/agent_type": np.array([int(agent_history_info["type"][i])]).reshape(1),
                "other/agent_type": np.delete(agent_history_info["type"], i, axis=0).astype(int),

                # is_sdc stands for "is self-driving car", it is 1 if the agent is a self-driving car and 0 otherwise
                "target/is_sdc": np.array(int(agent_history_info["is_sdc"][i])).reshape(1),
                "other/is_sdc": np.delete(agent_history_info["is_sdc"], i, axis=0).astype(int),

                "target/width": agent_history_info["width"][i].item(),
                "target/length": agent_history_info["length"][i].item(),
                "other/width": np.delete(agent_history_info["width"], i),
                "other/length": np.delete(agent_history_info["length"], i),

                "target/future/xy": current_scene_agents_coordinates_future[i][None,],
                "target/future/yaw": current_scene_agents_yaws_future[i][None,],
                "target/future/speed": agent_history_info["future/speed"][i][None,],
                "target/future/valid": agent_history_info["future/valid"][i][None,],
                "target/history/xy": current_scene_target_agent_coordinates_history,
                "target/history/yaw": current_scene_target_agent_yaws_history,
                "target/history/speed": current_scene_target_agent_speed_history,
                "target/history/valid": agent_history_info["history/valid"][i][None,],

                "other/future/xy": np.delete(current_scene_agents_coordinates_future, i, axis=0),
                "other/future/yaw": np.delete(current_scene_agents_yaws_future, i, axis=0),
                "other/future/speed": np.delete(agent_history_info["future/speed"], i, axis=0),
                "other/future/valid": np.delete(agent_history_info["future/valid"], i, axis=0),
                "other/history/xy": current_scene_other_agents_coordinates_history,
                "other/history/yaw": current_scene_other_agents_yaws_history,
                "other/history/speed": current_scene_other_agents_speed_history,
                "other/history/valid": np.delete(agent_history_info["history/valid"], i, axis=0),

                "road_network_embeddings": road_segments_embeddings,
                "road_network_segments": current_scene_road_network_coordinates
            }
            scene_data["trajectory_bucket"] = self._get_trajectory_class(scene_data)
            array_of_scene_data_dicts.append(scene_data)
        return array_of_scene_data_dicts
