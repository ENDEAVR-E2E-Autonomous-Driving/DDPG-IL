import carla
import random
import pygame
import queue
import numpy as np
import torch
import math


class CarlaScene:
    def __init__(self, town='Town10HD', weather=carla.WeatherParameters.ClearNoon):
        self.client = carla.Client('127.0.0.1', 2000)
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

        # for waypoints
        self.spectator = self.world.get_spectator()
        self.unique_waypoints = []


    def add_car(self, blueprint_name='vehicle.ford.crown', spawn_point=None):
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

        blueprint = self.blueprint_library.find(blueprint_name)
        vehicle = self.world.spawn_actor(blueprint, spawn_point)

        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

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
    
    def generate_unique_waypoints(self, draw_waypoints=True):
        print("Generating unique waypoints...")
        spectator_pos = carla.Transform(carla.Location(x=0, y=30, z=200), carla.Rotation(pitch=-90, yaw=-90))
        self.spectator.set_transform(spectator_pos)
        all_waypoints = self.town_map.generate_waypoints(0.3)
        self.unique_waypoints = []
        i = 1
        num_wp = len(all_waypoints)
        for wp in all_waypoints:
            print(f"{i}/{num_wp} waypoints")
            if not self.unique_waypoints or not any(wp.transform.location.distance(uwp.transform.location) < 0.1 for uwp in self.unique_waypoints):
                self.unique_waypoints.append(wp)
                if draw_waypoints:
                    self.world.debug.draw_string(wp.transform.location + carla.Location(z=1), 'O', color=carla.Color(r=255, g=0, b=0), life_time=120.0)

            i += 1

        print("Unique waypoints are generated and drawn.")


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
        self.collision_history = []
        self._setup_collision_sensor()

    def get_spawn_point(self):
        return self._spawn_point

    def reset(self):
        if self._spawn_point:
            self.object.set_target_velocity(carla.Vector3D())
            self.object.set_transform(self._spawn_point)

    def apply_control(self, control):
        if control.brake < 0.01:
            control.brake = 0.0

        self.object.apply_control(control)

    def get_velocity(self):
        velocity = self.object.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

    def get_velocity_norm(self):
        return self.get_velocity() / 120.0

    def get_autopilot_control(self, model, scalars, image, command):
        if model:
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float().unsqueeze(0)
            scalars = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0)
            command = torch.tensor([command], dtype=torch.uint8).unsqueeze(0)

            with torch.no_grad():
                output = model(image, scalars, command)
                steer, throttle, brake = tuple(output.squeeze().tolist())

            steer = self._last_steer + (steer - self._last_steer) * 0.10
            throttle = self._last_throttle + (throttle - self._last_throttle) * 0.25

            self._last_steer = steer
            self._last_throttle = throttle

            if self.get_velocity() >= 20.0:
                throttle = 0.0

            return steer, throttle, brake
        

    def _setup_collision_sensor(self):
        blueprint = self.object.get_world().get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.object.get_world().spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=self.object
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision_history.append(event)
        
    """
    Vehicle reward function: R_t = V_x * cos(theta) - V_x * sin(theta) - V_x * lanePostion - P

    V_x = velocity of the vehicle, theta = angle between vehicle and road center, 
    lanePosition = off-center position of vehicle, P = additional penalty

    V_x * cos(theta) = vehicle's velocity along direction of the road
    V_x * sin(theta) = vehicle's velocity perpendicular to the road
    V_x * lanePosition = penalty for the vehicle being off the center of the lane
    """
    def update_reward(self, closest_waypoint):
        velocity = self.get_velocity()

        # get vehicle heading and waypoint heading in radians
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        waypoint_yaw_rad = math.radians(closest_waypoint.transform.rotation.yaw)

        # compute the angle between the vehicle's heading and the waypoint's direction
        theta = vehicle_yaw_rad - waypoint_yaw_rad
        theta = (theta + math.pi) % (2 * math.pi) - math.pi  # normalize angle to [-pi, pi]

        # compute lanePosition (deviation from the current waypoint)
        lanePosition = closest_waypoint.transform.location.distance(vehicle_transform.location)

        # base reward without additional penalty
        reward = (velocity * math.cos(theta)) - (velocity * math.sin(theta)) - velocity * lanePosition

        # additional penalties
        additional_penalty = 0

        if len(self.collision_history) > 0: # collision penalty
            additional_penalty += 50
        if lanePosition > 2.0: # off-road penalty
            additional_penalty += 50
        
        angle_threshold = math.radians(30) # 30 degrees
        if abs(theta) > angle_threshold: # high angle deviation penalty
            additional_penalty += 10
        
        if velocity < 0.1: # stuck penalty
            self.stuck_steps += 1
            if self.stuck_steps >= 5: # num consecutive low-speed stepts to be considered stuck
                additional_penalty += 50
        else:
            self.stuck_steps = 0 # reset if not stuck anymore

        # final reward
        reward -= additional_penalty

        return reward


