from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Set

from pyquaternion import Quaternion
import numpy as np
from numpy import linalg
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap


TOTAL_TIMESTEPS_LIMIT = 39


class WaymoAgentType(IntEnum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


@dataclass
class AgentRecord:
    x: float = -1
    y: float = -1
    yaw: float = -1
    length: float = -1
    width: float = -1
    speed: float = -1
    is_sdc: bool = False
    category: str = ''
    valid: bool = False
    attributes: Set[str] = field(default_factory=lambda: set())

    def category_to_type(self):
        if self.category == '':
            return WaymoAgentType.UNSET
        if self.category.startswith('human.pedestrian'):
            return WaymoAgentType.PEDESTRIAN
        if self.category == 'vehicle.bicycle' and 'cycle.with_rider' in self.attributes:
            return WaymoAgentType.CYCLIST
        if self.category.startswith('vehicle') and self.category != 'vehicle.bicycle':
            return WaymoAgentType.VEHICLE
        return WaymoAgentType.OTHER

    def get_core_tuple(self):
        return self.x, self.y, self.yaw, self.speed, self.valid


def get_annotation_tokens_by_sample(nuscenes, scene):
    curr_sample = nuscenes.get('sample', scene['first_sample_token'])
    samples_with_annotations = [curr_sample['anns']]
    while curr_sample['next'] != '':
        curr_sample = nuscenes.get('sample', curr_sample['next'])
        samples_with_annotations.append(curr_sample['anns'])

    return samples_with_annotations


def get_agents_data(nuscenes, annotation_tokens):
    agent_id_to_data = {}
    for sample_annotation_token in annotation_tokens:
        sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)

        x, y = sample_annotation['translation'][:2]
        rotation_quaternion = Quaternion(sample_annotation['rotation'])
        width, length = sample_annotation['size'][:2]
        speed = linalg.norm(nuscenes.box_velocity(sample_annotation_token))

        attributes = {nuscenes.get('attribute', attribute_token)['name'] for attribute_token in
                      sample_annotation['attribute_tokens']}

        agent_record = AgentRecord(
            x=x,
            y=y,
            yaw=quaternion_yaw(rotation_quaternion),
            length=length,
            width=width,
            speed=speed,
            category=sample_annotation['category_name'],
            valid=True,
            attributes=attributes,
        )

        agent_id_to_data[sample_annotation['instance_token']] = agent_record

    return agent_id_to_data


def get_scene_samples_data(nuscenes, scene):
    scene_samples_data = []
    for sample_annotation_tokens in get_annotation_tokens_by_sample(nuscenes, scene):
        sample_agents_data = get_agents_data(nuscenes, sample_annotation_tokens)
        scene_samples_data.append(sample_agents_data)
    return scene_samples_data


def get_scenes_data(nuscenes):
    scenes_data = []
    for scene in nuscenes.scene:
        scene_samples_data = get_scene_samples_data(nuscenes, scene)
        scenes_data.append(scene_samples_data)
    return scenes_data


@dataclass
class SceneBoundingBox:
    x_min: float = -1
    y_min: float = -1
    x_max: float = -1
    y_max: float = -1

    def extend_by_radius(self, r):
        self.x_min -= r
        self.y_min -= r
        self.x_max += r
        self.y_max += r

    def as_patch(self):
        return self.x_min, self.y_min, self.x_max, self.y_max


def scene_data_to_agents_timesteps_dict(scene_id, scene_samples_data, current_timestep_idx):
    num_timesteps_total = min(TOTAL_TIMESTEPS_LIMIT, len(scene_samples_data))
    scene_samples_data = scene_samples_data[:num_timesteps_total]

    agent_to_timestep_to_data = defaultdict(lambda: [AgentRecord()] * num_timesteps_total)

    for timestep, agents_data in enumerate(scene_samples_data):
        for agent_id, agent_record in agents_data.items():
            agent_to_timestep_to_data[agent_id][timestep] = agent_record

    num_agents = len(agent_to_timestep_to_data)

    num_timesteps_history = current_timestep_idx
    num_timesteps_future = num_timesteps_total - num_timesteps_history - 1

    core_data_array = np.empty((num_agents, num_timesteps_total, 5))
    for agent_idx, (agent_id, agent_records) in enumerate(agent_to_timestep_to_data.items()):
        agent_records_core = [agent_record.get_core_tuple() for agent_record in agent_records]
        core_data_array[agent_idx] = np.array(agent_records_core)



    result = {
        'scenario/id': np.array(str(scene_id).encode('utf-8')),
        'state/id': np.empty(num_agents),
        'state/is_sdc': np.empty(num_agents),
        'state/type': np.empty(num_agents),
        'state/tracks_to_predict': np.empty(num_agents),

        'state/current/length': np.empty((num_agents, 1)),
        'state/current/width': np.empty((num_agents, 1)),

        'state/past/x': np.empty((num_agents, num_timesteps_history)),
        'state/past/y': np.empty((num_agents, num_timesteps_history)),
        'state/past/bbox_yaw': np.empty((num_agents, num_timesteps_history)),
        'state/past/speed': np.empty((num_agents, num_timesteps_history)),
        'state/past/valid': np.empty((num_agents, num_timesteps_history)),

        'state/current/x': np.empty((num_agents, 1)),
        'state/current/y': np.empty((num_agents, 1)),
        'state/current/bbox_yaw': np.empty((num_agents, 1)),
        'state/current/speed': np.empty((num_agents, 1)),
        'state/current/valid': np.empty((num_agents, 1)),

        'state/future/x': np.empty((num_agents, num_timesteps_future)),
        'state/future/y': np.empty((num_agents, num_timesteps_future)),
        'state/future/bbox_yaw': np.empty((num_agents, num_timesteps_future)),
        'state/future/speed': np.empty((num_agents, num_timesteps_future)),
        'state/future/valid': np.empty((num_agents, num_timesteps_future)),
    }

    for agent_idx, (agent_id, agent_records) in enumerate(agent_to_timestep_to_data.items()):
        result['state/id'][agent_idx] = agent_idx

        valid_record = next((record for record in agent_records if record.valid))

        result['state/is_sdc'][agent_idx] = valid_record.is_sdc
        result['state/type'][agent_idx] = valid_record.category_to_type()
        result['state/tracks_to_predict'][agent_idx] = agent_idx

        result['state/current/length'][agent_idx] = valid_record.length
        result['state/current/width'][agent_idx] = valid_record.width

        result['state/past/x'][agent_idx] = core_data_array[agent_idx, :current_timestep_idx, 0]
        result['state/past/y'][agent_idx] = core_data_array[agent_idx, :current_timestep_idx, 1]
        result['state/past/bbox_yaw'][agent_idx] = core_data_array[agent_idx, :current_timestep_idx, 2]
        result['state/past/speed'][agent_idx] = core_data_array[agent_idx, :current_timestep_idx, 3]
        result['state/past/valid'][agent_idx] = core_data_array[agent_idx, :current_timestep_idx, 4]

        result['state/current/x'][agent_idx] = core_data_array[agent_idx, current_timestep_idx, 0]
        result['state/current/y'][agent_idx] = core_data_array[agent_idx, current_timestep_idx, 1]
        result['state/current/bbox_yaw'][agent_idx] = core_data_array[agent_idx, current_timestep_idx, 2]
        result['state/current/speed'][agent_idx] = core_data_array[agent_idx, current_timestep_idx, 3]
        result['state/current/valid'][agent_idx] = core_data_array[agent_idx, current_timestep_idx, 4]

        result['state/future/x'][agent_idx] = core_data_array[agent_idx, current_timestep_idx+1:, 0]
        result['state/future/y'][agent_idx] = core_data_array[agent_idx, current_timestep_idx+1:, 1]
        result['state/future/bbox_yaw'][agent_idx] = core_data_array[agent_idx, current_timestep_idx+1:, 2]
        result['state/future/speed'][agent_idx] = core_data_array[agent_idx, current_timestep_idx+1:, 3]
        result['state/future/valid'][agent_idx] = core_data_array[agent_idx, current_timestep_idx+1:, 4]

    x_min, y_min = core_data_array[:, :, :2].min(axis=(0, 1))
    x_max, y_max = core_data_array[:, :, :2].max(axis=(0, 1))

    scene_bounding_box = SceneBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )

    return result, scene_bounding_box


class RoadgraphLayerType(IntEnum):
    ROAD_SEGMENT = 0
    ROAD_BLOCK = 1
    LANE = 2
    PED_CROSSING = 3
    WALKWAY = 4
    STOP_LINE = 5
    CARPARK_AREA = 6
    ROAD_DIVIDER = 7
    LANE_DIVIDER = 8


def roadgraph_layer_string_to_enum(layer):
    if layer == 'road_segment':
        return RoadgraphLayerType.ROAD_SEGMENT
    if layer == 'road_block':
        return RoadgraphLayerType.ROAD_BLOCK
    if layer == 'lane':
        return RoadgraphLayerType.LANE
    if layer == 'ped_crossing':
        return RoadgraphLayerType.PED_CROSSING
    if layer == 'walkway':
        return RoadgraphLayerType.WALKWAY
    if layer == 'stop_line':
        return RoadgraphLayerType.STOP_LINE
    if layer == 'carpark_area':
        return RoadgraphLayerType.CARPARK_AREA
    if layer == 'road_divider':
        return RoadgraphLayerType.ROAD_DIVIDER
    if layer == 'lane_divider':
        return RoadgraphLayerType.LANE_DIVIDER
    raise RuntimeError(f'Unknown layer: {layer}')


def get_scene_map(nuscenes, scene):
    scene_location = nuscenes.get('log', scene['log_token'])['location']
    return NuScenesMap(dataroot=nuscenes.dataroot, map_name=scene_location)


def get_scene_roadgraph(nusc_map, scene_bbox, r, layers_of_interest):
    scene_bbox.extend_by_radius(r)

    node_coordinates = []
    node_object_ids = []
    node_types = []

    for layer_name, layer_object_tokens in nusc_map.get_records_in_patch(scene_bbox.as_patch(), layer_names=layers_of_interest).items():
        for object_token in layer_object_tokens:
            layer_objects = nusc_map.get(layer_name, object_token)

            node_tokens = layer_objects.get('exterior_node_tokens', []) + layer_objects.get('node_tokens', [])
            for node_token in node_tokens:
                node_data = nusc_map.get('node', node_token)
                x, y = node_data['x'], node_data['y']

                node_coordinates.append((x, y))
                node_object_ids.append(object_token)
                node_types.append(int(roadgraph_layer_string_to_enum(layer_name)))

    return {
        'roadgraph_samples/xyz': np.array(node_coordinates),
        'roadgraph_samples/id': np.array(node_object_ids),
        'roadgraph_samples/type': np.array(node_types),
        'roadgraph_samples/valid': np.ones(len(node_coordinates)),
    }


def get_full_scene_data(nuscenes, config, scene_id):
    scene = nuscenes.scene[scene_id]

    scene_samples_data = get_scene_samples_data(nuscenes, scene)
    agents_dict, scene_bbox = scene_data_to_agents_timesteps_dict(scene_id, scene_samples_data, config["current_timestep_idx"])

    scene_map = get_scene_map(nuscenes, scene)
    roadgraph_dict = get_scene_roadgraph(scene_map, scene_bbox, config["map_expansion_radius"],
                                         config["layers_of_interest"])

    agents_dict.update(roadgraph_dict)

    return agents_dict