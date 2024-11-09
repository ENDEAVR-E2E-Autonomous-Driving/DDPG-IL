import os
from torchrl.data import LazyMemmapStorage, ReplayBuffer
import carla
import numpy as np
import torch
import random
from model import *
from dotenv import load_dotenv
import time
from environment import *
load_dotenv()
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from tensordict import TensorDict
import json
from utils import *
from input import *

"""TRAINING HELPERS"""
# ------------------------------------------------------------------------------------------- 

# helper function for getting updated command and command cooldown counter
def get_command_and_cooldown(wp_direction, command_cooldown_counter):
    """
    Adjusts the command and command cooldown counter based on the current waypoint direction.
    Returns an updated command and command cooldown counter.
    """
    command = 0
    if wp_direction == RoadOption.LEFT:
        command = 0
        command_cooldown_counter = 180
    elif wp_direction == RoadOption.RIGHT:
        command = 2
        command_cooldown_counter = 180
    else:
        if command_cooldown_counter > 0:
            command_cooldown_counter -= 1
        else:
            command = 1 

    return command, command_cooldown_counter


# helper function to make a directory
def make_directory(dir_name):
    if not os.path.exists:
        os.makedirs(dir_name, exist_ok=True)

# helper function to get state info
# def get_state_info(vehicle: CarlaVehicle, camera: CarlaCamera):
#     speed_limit = vehicle.object.get_speed_limit()
#     gear = vehicle.object.get_control().gear
#     scalars = [vehicle.get_velocity_norm(), speed_limit / 120.0, gear / 8.0]
#     forward_image = camera.get_image_float()

#     return scalars, forward_image

# method to convert tuple to tensordict for storing in replay buffer
def tuple_to_tensordict(transition_tuple):
    state = transition_tuple[0]
    action = transition_tuple[1]
    reward = transition_tuple[2]
    next_state = transition_tuple[3]
    done = transition_tuple[4]

    transition = TensorDict({
        "state": torch.tensor(state[0], dtype=torch.float32),
        "scalars": torch.tensor(state[1], dtype=torch.float32),
        "command": torch.tensor([state[2]], dtype=torch.long),
        "action": torch.tensor(action, dtype=torch.float32),
        "reward": torch.tensor([reward], dtype=torch.float32),
        "next_state": torch.tensor(next_state[0], dtype=torch.float32),
        "next_scalars": torch.tensor(next_state[1], dtype=torch.float32),
        "next_command": torch.tensor([next_state[2]], dtype=torch.long),
        "done": torch.tensor([done], dtype=torch.float32)
    }, batch_size=[])

    return transition

# method to get x and z cooridinates of a vehicle to place a forward camera
def get_camera_placement_coordinates(vehicle: CarlaVehicle):
    # get bounding box and transform 
    vehicle_object = vehicle.object
    
    vehicle_bounding_box = vehicle_object.bounding_box
    vehicle_transform = vehicle_object.get_transform()

    # get x and z values for camera placement
    # x: place camera slightly in front of vehicle's bounding box (windshield area)
    # z: place camera at the top of the bus
    x_offset = vehicle_bounding_box.extent.x + 0.5 # 0.5 adjusts for small hood (can be adjusted)
    z_offset = vehicle_bounding_box.extent.z # top of the vehicle

    return x_offset, z_offset


def draw_route(local_planner: LocalPlanner, scene: CarlaScene, lifetime: float):
    for (wp_location, direction) in local_planner._waypoints_queue:
        wp_location = wp_location.transform.location
        wp_location.z += 15.0

        color = carla.Color(255, 0, 0) if direction == RoadOption.LEFT else carla.Color(0, 255, 0) if direction == RoadOption.RIGHT else carla.Color(0, 0, 255)

        scene.world.debug.draw_string(wp_location, "X", color=color, life_time=lifetime)

def waypoints_to_json_serializable(waypoints):
    waypoints_data = []

    for waypoint in waypoints:
        waypoint_data = {
            'location': {
                'x': waypoint.transform.location.x,
                'y': waypoint.transform.location.y,
                'z': waypoint.transform.location.z,
            },
            'rotation': {
                'pitch': waypoint.transform.rotation.pitch,
                'yaw': waypoint.transform.rotation.yaw,
                'roll': waypoint.transform.rotation.roll,
            },
            'road_option': str(waypoint.road_option) if hasattr(waypoint, 'road_option') else None
        }
        waypoints_data.append(waypoint_data)

    return waypoints_data

def load_waypoints_from_json(filename, scene):
    with open(filename, 'r') as json_file:
        waypoints_data = json.load(json_file)

    waypoints = []
    for waypoint_data in waypoints_data:
        location = carla.Location(
            x=waypoint_data['location']['x'],
            y=waypoint_data['location']['y'],
            z=waypoint_data['location']['z']
        )
        rotation = carla.Rotation(
            pitch=waypoint_data['rotation']['pitch'],
            yaw=waypoint_data['rotation']['yaw'],
            roll=waypoint_data['rotation']['roll']
        )
        transform = carla.Transform(location, rotation)

        waypoint = scene.world.get_map().get_waypoint(location)  # Retrieve the closest valid waypoint
        waypoints.append(waypoint)

    return waypoints



# helper function for creating an expert replay buffer with IL
def get_expert_data_with_IL(device, il_model: IL_Model, buffer_size=100000, exper_dir = "expert_buffer"):
    """
    Runs a trained IL model for 10 rounds to collect and store its expert experiences.
    Similar to DDPG training loop without the noisy action selection.
    Returns an expert replay buffer and the maximum reward achieved.
    """

    # Initialize the Carla scene
    scene = CarlaScene(town="Town02")
    scene.set_spectator_position(200.0)

    # Add a car to the scene
    vehicle = scene.add_car()
    scene.add_collision_sensor(vehicle.object)

    forward_camera = CarlaCamera(vehicle.object)
    scene.add_camera(forward_camera)

    command = 1

    il_max_reward = float('-inf') # get max reward when running the IL rounds
    make_directory(exper_dir)
    storage = LazyMemmapStorage(max_size=buffer_size, scratch_dir=exper_dir)
    expert_buffer = ReplayBuffer(storage=storage)

    print_stats_interval = 50
    scene.set_spectator_position(height=250.0) # set birds-eye view position for spectator
    max_time_per_episode = 300 # in seconds
    max_steps_per_episode = 2000

    print("Starting IL Rounds")
    try:
        scene.run()
        state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)
        for i in range(10):
            print("-------------------------------------------------------------------------")
            print_formatted(f"ROUND {i}", GRAY)
            
            vehicle.reset(scene)
            vehicle.draw_vehicle_location(scene, 0)
            scene.collision_history.clear()

            done = False
            total_reward = 0
            num_steps = 0
            episode_start_time = time.time()
            
            while not done:
                # if change_weather:
                #     current_weather = scene.world.get_weather()

                #     weather = carla.WeatherParameters(
                #         sun_altitude_angle=(current_weather.sun_altitude_angle + 0.5) % 180)

                #     scene.world.set_weather(weather)

                # if for_demo:
                #     for event in pygame.event.get():
                #         if event.type == pygame.QUIT:
                #             quit = True
                #         elif event.type == pygame.KEYDOWN:
                #             if event.key == pygame.K_q:
                #                 quit = Tru

                steer, throttle, brake = vehicle.get_autopilot_control(il_model, scalars=state[1], image=state[0], command=state[-1])    
                action = np.array([steer, throttle, brake])

                vehicle.apply_control(carla.VehicleControl(throttle=action[1], steer=action[0], brake=action[2]))

                scene.run()

                # step environment and get new state and reward
                next_state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)

                reward, done, lane_position, velocity, theta = vehicle.get_reward(scene, scene.collision_history)
                # store experience in exploration buffer
                transition = tuple_to_tensordict((state, action, reward, next_state, done))
                expert_buffer.add(transition)

                # more done conditions
                current_time = time.time()

                if num_steps % 50 == 0:
                    print_formatted(f"Steps: {num_steps}, reward: {reward},", RED)
                    print_formatted(f"lane deviation: {lane_position}, velocity: {velocity}, theta: {theta}", RED)

                # more done conditions
                if num_steps >= 2000:
                    done = True
                    print("Episode finished: max steps")
                if current_time - episode_start_time >= max_time_per_episode:
                    done = True
                    print("Episode finished: max time")

                num_steps += 1
                
            # update state
            state = next_state
            total_reward += reward

            il_max_reward = max(il_max_reward, total_reward)
            
            print("Overall episode stats:")
            print_formatted(f"episode: {i}, reward: {total_reward}, steps: {num_steps}", YELLOW)

    finally:
        scene.cleanup()

    print("IL Rounds Complete.")
    print("-------------------")

    print(f"Expert buffer size: {len(expert_buffer)}")
    return expert_buffer, il_max_reward


# helper function to combine samples from the exploration and expert buffers
def combine_and_shuffle_samples(expert_batch: TensorDict, exploration_batch: TensorDict):
    """
    Mixes samples from the expert and exploration batches, concatenates them, and shuffles the result.
    Both expert_batch and exploration_batch are expected to be TensorDicts.
    """
    # Check if either batch is empty
    if expert_batch.batch_size == torch.Size([]) and exploration_batch.batch_size == torch.Size([]):
        raise ValueError("Both expert and exploration batches are empty.")

    if expert_batch.batch_size == torch.Size([]):
        # If expert_batch is empty, return only exploration_batch
        return exploration_batch
    elif exploration_batch.batch_size == torch.Size([]):
        # If exploration_batch is empty, return only expert_batch
        return expert_batch

    # Concatenate the TensorDicts along the batch dimension
    combined_batch = torch.cat([expert_batch, exploration_batch], dim=0)

    # Get the number of samples
    num_samples = combined_batch.batch_size[0]

    # Shuffle the indices
    indices = torch.randperm(num_samples)

    # Shuffle the entire TensorDict
    shuffled_batch = combined_batch[indices]

    # Return the shuffled batch
    return shuffled_batch


# helper function to update actor and critic
def learn(batch_samples, device, actor: Actor, target_actor: Actor, critic: Critic, target_critic: Critic, \
          actor_optimizer: torch.optim.Adam, critic_optimizer: torch.optim.Adam):
        
    # Extract the required fields from the TensorDict
    states_img = batch_samples["state"].permute(0, 3, 1, 2).to(device).float()
    states_scalar = batch_samples["scalars"].to(device).float()
    states_cmd = batch_samples["command"].to(device).long()

    next_states_img = batch_samples["next_state"].permute(0, 3, 1, 2).to(device).float()
    next_states_scalar = batch_samples["next_scalars"].to(device).float()
    next_states_cmd = batch_samples["next_command"].to(device).long()

    actions = batch_samples["action"].to(device).float()
    rewards = batch_samples["reward"].to(device).float()
    dones = batch_samples["done"].to(device).float()

    # Update critic
    gamma = 0.99
    with torch.no_grad():
        next_actions = target_actor.forward(next_states_img, next_states_scalar, next_states_cmd)
        next_q_values = target_critic.forward(next_states_img, next_states_scalar, next_states_cmd, next_actions)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)  # Ensure target_q_values has the same shape as q_values

    q_values = critic.forward(states_img, states_scalar, states_cmd, actions)
    # target_q_values = target_q_values.view_as(q_values)  # Reshape target_q_values to match q_values
    critic_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update actor
    predicted_actions = actor.forward(states_img, states_scalar, states_cmd)
    actor_loss = -critic.forward(states_img, states_scalar, states_cmd, predicted_actions).mean()  # Maximizing reward

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    tau = 0.005
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    return critic_loss, actor_loss

# ------------------------------------------------------------------------------------------- 
