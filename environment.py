import carla
import random
import pygame
import queue
import numpy as np
import torch
import math
import time
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
import ctypes


class CarlaScene:
    def __init__(self, town='Town10HD', weather=carla.WeatherParameters.ClearNoon, host_ip_adress=None, steer_image_path=None):
        """
        host_ip_address: IP address of the host system that runs the CARLA server if running the client from a different port, such as WSL
        """
        self.client = carla.Client(host_ip_adress if host_ip_adress else 'localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.world.set_weather(weather)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (1.0 / 30)
        self.world.apply_settings(settings)

        self.display = None
        self.actors = []
        self.blueprint_library = self.world.get_blueprint_library()

        self._game_camera = None
        self._clock = pygame.time.Clock()

        self._steer_image = pygame.image.load('assets/wheel.png') 

        self.spawn_points = self.world.get_map().get_spawn_points()

        # collision info
        self.collision_history = []

        self.lookahead_steps = 20

    def set_spectator_position(self, height: float):
        self.spectator = self.world.get_spectator()
        # initialize variables to compute the average x, y, and z of the spawn points
        total_x, total_y, total_z = 0, 0, 0

        # sum all spawn points' locations
        for spawn_point in self.spawn_points:
            total_x += spawn_point.location.x
            total_y += spawn_point.location.y
            total_z += spawn_point.location.z
        
        # compute average to get the approximate center of the map
        num_points = len(self.spawn_points)
        center_x = total_x / num_points
        center_y = total_y / num_points
        center_z = total_z / num_points

        # The center of the town
        town_center = carla.Location(center_x, center_y, center_z)

        # height = height # height of spectator view
        birdseye_position = carla.Location(town_center.x, town_center.y, town_center.z + height) # position of spectator
        birdseye_rotation = carla.Rotation(pitch=-90, yaw=0, roll=0) # rotation to look straight down (-90 degree pitch)

        self.spectator.set_transform(carla.Transform(birdseye_position, birdseye_rotation))


    def add_car(self, blueprint_name='vehicle.ford.crown', spawn_point=None):
        if spawn_point is None:
            # spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(self.spawn_points)
            # print(f"Vehicle spawn point: {spawn_point.location}")
            # self.spawn_points = spawn_points

        blueprint = self.blueprint_library.find(blueprint_name)
        vehicle = self.world.spawn_actor(blueprint, spawn_point)
        print(f"vehicle name: {vehicle.type_id}")

        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

        self.actors.append(vehicle)

        return CarlaVehicle(vehicle, spawn_point)

    def add_game_camera(self, camera):
        self.actors.append(camera.camera)
        self._game_camera = camera

    def add_camera(self, camera):
        self.actors.append(camera.camera)

    def open_window(self, w=800, h=600):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.w, self.h = w, h
        ctypes.windll.user32.SetForegroundWindow(pygame.display.get_wm_info()['window'])

    def get_window_size(self):
        return self.w, self.h

    def render_steer(self, steer, x=25, y=25, scale=0.5):
        steer_image = pygame.transform.rotozoom(self._steer_image, -steer * (900 / 2), scale)
        rect = steer_image.get_rect(center=(x, y))
        self.display.blit(steer_image, rect)

    def render_text(self, text, x=25, y=25, color=(255, 255, 255), size=24, padding=10, opacity=128, anchor='topleft'):
        font = pygame.font.Font(None, size)
        total_height = sum(font.size(f"{key}: {value}")[1] + padding for key, value in text.items()) + padding
        max_text_width = max(font.size(f"{key}: {value}")[0] for key, value in text.items())

        background_surface = pygame.Surface((max_text_width + padding * 2, total_height), pygame.SRCALPHA)
        background_surface.fill((0, 0, 0, opacity))

        current_y = padding
        for key, value in text.items():
            text_surface = font.render(f'{key}: {value}', True, color)
            background_surface.blit(text_surface, (padding, current_y))
            current_y += text_surface.get_height() + padding

        # Use the anchor parameter in blit
        rect = background_surface.get_rect(**{anchor: (x, y)})
        self.display.blit(background_surface, rect)

    def run(self):
        self.frames = self.world.tick()

        if self._game_camera is not None:
            game_image = self._game_camera.get_image()
            if game_image is not None:
                surface = pygame.surfarray.make_surface(game_image.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))

            font = pygame.font.Font(None, 24)
            self.display.blit(font.render('% 5d FPS (real)' % self._clock.get_fps(), True, (255, 255, 255)), (8, 10))
            self._clock.tick(30)

    def update_display(self):
        pygame.display.flip()

    def cleanup(self):
        for actor in self.actors:
            if actor.is_alive:
                actor.destroy()
        pygame.quit()

    def add_traffic(self, num_cars=35):
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.set_synchronous_mode(True)

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for n, transform in enumerate(spawn_points):
            if n >= num_cars:
                break

            blueprint = random.choice(self.blueprint_library.filter('vehicle.*'))
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.spawn_actor(blueprint, transform)
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            vehicle.set_autopilot(True)

            self.actors.append(vehicle)

        return traffic_manager
    
    def add_collision_sensor(self, vehicle):
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=vehicle
        )
        collision_sensor.listen(lambda event: self._on_collision(event))

        self.actors.append(collision_sensor)

        return collision_sensor

    def _on_collision(self, event):
        self.collision_history.append(event)
        print(f"Collision detected: {event}")
    
    def get_collision_history(self):
        return self.collision_history
    

class CarlaCamera:
    def __init__(self, vehicle, x=1.1, y=0.0, z=1.4, w=200, h=88, fov=80, rot=None, tick=None, semantic=False):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.width = w
        self.height = h

        self.semantic = semantic
        if semantic:
            bp = 'sensor.camera.semantic_segmentation'
        else:
            bp = 'sensor.camera.rgb'

        self.camera_bp = self.blueprint_library.find(bp)
        self.camera_bp.set_attribute('image_size_x', f'{w}')
        self.camera_bp.set_attribute('image_size_y', f'{h}')
        self.camera_bp.set_attribute('fov', f'{fov}')

        if tick is not None:
            self.camera_bp.set_attribute('sensor_tick', f'{tick}')

        self.camera_transform = carla.Transform(carla.Location(x=x, y=y, z=z), rot or carla.Rotation())
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)

        self.queue = queue.LifoQueue()
        self.camera.listen(self.queue.put)

    def process_image(self, data):
        if self.semantic:
            data.convert(carla.ColorConverter.CityScapesPalette)
        else:
            data.convert(carla.ColorConverter.Raw)
        raw = np.array(data.raw_data).astype('uint8')
        bgra = np.reshape(raw, (self.height, self.width, 4))
        bgr = bgra[:, :, :3]
        rgb = bgr[:, :, ::-1]

        return rgb

    def get_image(self):
        while not self.queue.empty():
            self.queue.get_nowait()

        return self.process_image(self.queue.get())

    def get_image_float(self):
        image = self.process_image(self.queue.get())

        while not self.queue.empty():
            self.queue.get_nowait()

        return image.astype('float32') / 255.0


class CarlaVehicle:
    def __init__(self, vehicle, spawn_point=None):
        self.object = vehicle

        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._spawn_point = spawn_point

        self.stuck_steps = 0
        self.angle_deviation_steps = 0
    
    def set_num_waypoints(self, num_waypoints):
        self.num_waypoints = num_waypoints
    
    def draw_vehicle_location(self, scene, lifetime):
        location = self.object.get_transform().location
        location.z += 10

        scene.world.debug.draw_string(location, "HERE", color=carla.Color(0,0,0), life_time=0)

    def get_spawn_point(self):
        return self._spawn_point

    def reset(self, scene):
        self.stuck_steps = 0
        spawn_point = random.choice(scene.world.get_map().get_spawn_points())
        self.object.set_target_velocity(carla.Vector3D())
        self.object.set_transform(spawn_point)

    def apply_control(self, control):
        if control.brake < 0.01:
            control.brake = 0.0

        self.object.apply_control(control)

    def get_velocity(self):
        velocity = self.object.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6 # convert to km/h

    def get_velocity_mps(self):
        velocity_kmh = self.get_velocity()
        return velocity_kmh / 3.6

    def get_velocity_norm(self):
        return self.get_velocity() / 120.0

    def get_autopilot_control(self, model, scalars, image, command, device=torch.device('cpu'), for_training=False):
        if model:
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float().unsqueeze(0).to(device)
            scalars = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(device)
            command = torch.tensor([command], dtype=torch.uint8).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image, scalars, command)
                steer, throttle, brake = tuple(output.squeeze().tolist())

            steer = self._last_steer + (steer - self._last_steer) * 0.10
            throttle = self._last_throttle + (throttle - self._last_throttle) * 0.25

            self._last_steer = steer
            self._last_throttle = throttle

            if for_training:
                throttle = max(min(throttle, 0.8), 0.2)  # Limit throttle between 20% and 80%

            return steer, throttle, brake

    def compute_theta(self, vehicle_yaw_rad, curr_waypoint, num_waypoints_ahead=5):
        """
        Computes theta for the reward function, considering the road direction over multiple waypoints ahead.
        """
        # Get multiple waypoints ahead to account for curvature
        accumulated_direction = np.array([0.0, 0.0])
        current_wp = curr_waypoint

        for _ in range(num_waypoints_ahead):
            next_waypoints = current_wp.next(2.0)  # Get the next waypoint 2m ahead
            if len(next_waypoints) > 0:
                next_waypoint = next_waypoints[0]
                # Compute the direction from the current waypoint to the next
                waypoint_direction = np.array([next_waypoint.transform.location.x - current_wp.transform.location.x,
                                            next_waypoint.transform.location.y - current_wp.transform.location.y])
                waypoint_direction /= np.linalg.norm(waypoint_direction)  # Normalize the direction vector
                accumulated_direction += waypoint_direction  # Accumulate direction over multiple waypoints
                current_wp = next_waypoint  # Move to the next waypoint for the next iteration
            else:
                break

        if np.linalg.norm(accumulated_direction) == 0:
            return None  # No direction available

        # Normalize the accumulated direction to get the road's overall heading
        road_direction = accumulated_direction / np.linalg.norm(accumulated_direction)

        # Compute the road direction in radians (yaw)
        road_yaw_rad = np.arctan2(road_direction[1], road_direction[0])

        # Compute theta as the difference between vehicle yaw and road direction
        theta = vehicle_yaw_rad - road_yaw_rad
        theta = (theta + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

        return theta

    def compute_curvature(self, curr_waypoint, num_waypoints_ahead=5):
        """
        Computes an estimate of the road curvature by measuring the angle difference
        between the current waypoint and multiple waypoints ahead.
        """
        curvature_sum = 0
        current_wp = curr_waypoint

        for _ in range(num_waypoints_ahead):
            next_waypoints = current_wp.next(2.0)  # Get the next waypoint 2 meters ahead
            if len(next_waypoints) > 0:
                next_waypoint = next_waypoints[0]
                # Compute direction between current and next waypoint
                current_direction = np.array([next_waypoint.transform.location.x - current_wp.transform.location.x,
                                            next_waypoint.transform.location.y - current_wp.transform.location.y])
                current_direction /= np.linalg.norm(current_direction)

                # Compute direction of the next segment if possible
                future_waypoints = next_waypoint.next(2.0)
                if len(future_waypoints) > 0:
                    future_direction = np.array([future_waypoints[0].transform.location.x - next_waypoint.transform.location.x,
                                                future_waypoints[0].transform.location.y - next_waypoint.transform.location.y])
                    future_direction /= np.linalg.norm(future_direction)

                    # Calculate angle between current and future direction vectors (curvature)
                    angle_diff = np.arccos(np.clip(np.dot(current_direction, future_direction), -1.0, 1.0))
                    curvature_sum += angle_diff
                current_wp = next_waypoint
            else:
                break

        # Return average curvature
        return curvature_sum / num_waypoints_ahead if num_waypoints_ahead > 0 else 0

    def get_reward(self, scene, collision_history):
        done = False

        velocity = self.get_velocity_mps()
        vehicle_transform = self.object.get_transform()
        vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        closest_waypoint = scene.world.get_map().get_waypoint(vehicle_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoint_yaw_rad = math.radians(closest_waypoint.transform.rotation.yaw)

        theta = vehicle_yaw_rad - waypoint_yaw_rad
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        speed_along_lane = velocity * math.cos(theta)
        speed_lateral = velocity * math.sin(theta)

        vehicle_position = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        waypoint_position = np.array([closest_waypoint.transform.location.x, closest_waypoint.transform.location.y])

        # Use the waypoint heading (not vehicle yaw) for correct direction
        waypoint_yaw_rad = math.radians(closest_waypoint.transform.rotation.yaw)
        waypoint_direction = np.array([np.cos(waypoint_yaw_rad), np.sin(waypoint_yaw_rad)])

        vehicle_to_waypoint = vehicle_position - waypoint_position
        lane_deviation = np.abs(np.cross(waypoint_direction, vehicle_to_waypoint))

        alpha, beta, eta = 0.33, 0.33, 0.33
        # restricting to 15 m/s = 33.3 mph
        r_speed = velocity if velocity < 13.0 else 13.0-velocity
        # r_speed = velocity / 15
        r_center = speed_along_lane - abs(speed_lateral) - lane_deviation - (speed_along_lane * lane_deviation)
        r_out = 0 if lane_deviation < 5.0 else 25 - lane_deviation

        reward = alpha * r_speed + beta * r_center + eta * r_out

        if velocity < 0.5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps >= 200:
            done = True
            print("Episode finished: stuck")
        if lane_deviation > 20.0:
            done = True
            print(f"Episode finished: lane deviation {lane_deviation} meters")
        if len(collision_history) > 0:
            done = True
            print("Episode finished: collision")

        return reward, done, lane_deviation, velocity, theta
    



