import json
import os

from carla_env.envs.base_env import CarlaEnv
from carla_env.utils import config_utils


class CoRL2017Env(CarlaEnv):
    def __init__(self, carla_map, host, port, obs_configs, terminal_configs, reward_configs,
                 weather_group, route_description, task_type, carla_fps, tm_port, seed):
        all_tasks = self.build_all_tasks(carla_map, weather_group, route_description, task_type)
        super().__init__(carla_map, host, port, obs_configs, terminal_configs, reward_configs, all_tasks, carla_fps, tm_port, seed)

    @staticmethod
    def build_all_tasks(carla_map, weather_group, route_description, task_type):
        assert carla_map in ['Town01', 'Town02']
        assert weather_group in ['new', 'train']
        assert route_description in ['cexp', 'lbc', 'driving-benchmarks']
        assert task_type in ['straight', 'one_curve', 'navigation', 'navigation_dynamic']

        # weather
        if weather_group == 'new':
            weathers = ['SoftRainSunset', 'WetSunset']
        elif weather_group == 'train':
            weathers = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']
        # task_type setup
        RLFOLD_ROOT = os.getenv('RLFOLD_ROOT')
        description_path = f'{RLFOLD_ROOT}/carla_env/envs/scenario_descriptions/CoRL2017/{route_description}'
        if task_type == 'straight':
            description_folder = f'{description_path}/Straight/{carla_map}'
            num_zombie_vehicles = 0
            num_zombie_walkers = 0
        elif task_type == 'one_curve':
            description_folder = f'{description_path}/OneCurve/{carla_map}'
            num_zombie_vehicles = 0
            num_zombie_walkers = 0
        elif task_type == 'navigation':
            description_folder = f'{description_path}/Navigation/{carla_map}'
            num_zombie_vehicles = 0
            num_zombie_walkers = 0
        elif task_type == 'navigation_dynamic':
            if carla_map == 'Town01':
                description_folder = f'{description_path}/Navigation/{carla_map}'
                num_zombie_vehicles = 20
                num_zombie_walkers = 50
            elif carla_map == 'Town02':
                num_zombie_vehicles = 15
                num_zombie_walkers = 50

        actor_configs_dict = json.load(open(f'{description_folder}/actors.json'))
        route_descriptions_dict = config_utils.parse_routes_file(f'{description_folder}/routes.xml')

        all_tasks = []
        for weather in weathers:
            for route_id, route_description in route_descriptions_dict.items():
                task = {
                    'weather': weather,
                    'description_folder': description_folder,
                    'route_id': route_id,
                    'num_zombie_vehicles': num_zombie_vehicles,
                    'num_zombie_walkers': num_zombie_walkers,
                    'ego_vehicles': {
                        'routes': route_description['ego_vehicles'],
                        'actors': actor_configs_dict['ego_vehicles'],
                    },
                    'scenario_actors': {
                        'routes': route_description['scenario_actors'],
                        'actors': actor_configs_dict['scenario_actors']
                    } if 'scenario_actors' in actor_configs_dict else {}
                }
                all_tasks.append(task)

        return all_tasks