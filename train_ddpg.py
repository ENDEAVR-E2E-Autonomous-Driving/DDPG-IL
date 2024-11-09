import pygame
import carla
import queue

from environment import CarlaScene, CarlaCamera
from data import DataManager
import torch
from torchrl.data import LazyMemmapStorage, ReplayBuffer
import os
import json
from model import *
from statistics import mean as avg
import argparse
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from utils import *
from input import *
from train_helpers import *
from dotenv import load_dotenv
from ou_noise import OUActionNoise

load_dotenv()

import logidrivepy

def train_ddpg(expert_buffer, device, il_model, il_max_reward, buffer_size=100000, std_dev=0.2, num_episodes=500, batch_size=128, for_demo=False):
    print_game_letterhead("DDPG-IL in CARLA")

    game_width = 1920
    game_height = 1080

    # Initialize the Carla scene
    scene = CarlaScene(town="Town02")
    if for_demo:
        scene.open_window(w=game_width, h=game_height)
    scene.set_spectator_position(200.0)

    # Add a car to the scene
    vehicle = scene.add_car()
    scene.add_collision_sensor(vehicle.object)

    # GRP and local planner for navigation ---------------------------------------------------------
    import random
    spawn_points = scene.world.get_map().get_spawn_points()

    grp = GlobalRoutePlanner(scene.world.get_map(), sampling_resolution=4.0)
    local_planner = LocalPlanner(vehicle.object, map_inst=scene.world.get_map())

    start_waypoint = scene.world.get_map().get_waypoint(vehicle.get_spawn_point().location)
    end_waypoint = scene.world.get_map().get_waypoint(random.choice(spawn_points).location)

    route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)

    local_planner.set_global_plan(route)

    # ---------------------------------------------------------------------------------------------

    # Add cameras to the scene
    if for_demo:
        window_width, window_height = scene.get_window_size()

    forward_camera = CarlaCamera(vehicle.object)
    scene.add_camera(forward_camera)
    
    if for_demo:
        game_camera = CarlaCamera(vehicle.object, x=2.3, w=window_width, h=window_height, fov=110)
        left_camera = CarlaCamera(vehicle.object, y=-0.25, rot=carla.Rotation(yaw=-5))
        right_camera = CarlaCamera(vehicle.object, y=0.25, rot=carla.Rotation(yaw=5))
        scene.add_game_camera(game_camera)
        scene.add_camera(left_camera)
        scene.add_camera(right_camera)    

    action_dim = 3
    # initialize DDPG networks
    actor = Actor(pretrained_model=il_model).to(device)
    target_actor = Actor(pretrained_model=il_model).to(device)
    critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)
    target_critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)

    # initialize optimizers and action selection noise
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    mean = np.zeros(action_dim)
    std_dev = std_dev * np.ones(action_dim) 
    ou_noise = OUActionNoise(mean=mean, std_dev=std_dev)

    # initialize DDPG replay buffer
    print_formatted("Initializing the exploration replay buffer", GREEN)
    exploration_dir = "exploration_buffer" # directory where its saved

    make_directory(exploration_dir)
    storage_exploration = LazyMemmapStorage(max_size=buffer_size, scratch_dir=exploration_dir)
    exploration_buffer = ReplayBuffer(storage=storage_exploration)
    print_formatted("Exploration buffer loaded.", GREEN)
    # variables for data collection for DDPG training
    running = True
    collecting = False
    autopilot = True
    carla_autopilot = False  # New variable for CARLA's autopilot
    command = 1
    distance_traveled = 0.0
    command_cooldown_counter = 0
    logitech_detected = True
    navigate = False
    change_weather = False
    train_ddpg = False
    next_state = None
    stat_filename = "stats.json" if not for_demo else "stats_demo.json"
    start_waypoint = None
    end_waypoint = None
    route = None
    max_time_per_episode = 300 # in seconds
    saved_models_dir = "model_weights" # directory creation already exists in save_models method

    # initializations for statistics
    episodes_list = []
    reward_per_episode = []
    steps_per_episode = []
    actorLoss_per_episode = []
    criticLoss_per_episode = []

    buff_a = 0.8

    if for_demo:    
        try:
            logitech = logidrivepy.LogitechController()
            logitech.steering_initialize()
            print("Logitech initialized")
        except:
            print("Logitech not detected")
            logitech_detected = False

    # num_steps = 0
    quit = False
    try:
        scene.run()
        state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit() / 120.0, vehicle.object.get_control().gear / 8.0], command)
        while running:
            for i in range(num_episodes): 
                print("-------------------------------------------------------------------------")
                print_formatted(f"EPISODE {i}", GRAY)

                if quit:
                    running = False
                    break
                
                vehicle.reset(scene)
                vehicle.draw_vehicle_location(scene, 0.1)
                scene.collision_history.clear()

                done = False
                total_reward = 0
                num_steps = 0
                episode_start_time = time.time()

                if for_demo and navigate:
                    start_waypoint = scene.world.get_map().get_waypoint(vehicle.get_spawn_point().location)
                    end_waypoint = scene.world.get_map().get_waypoint(random.choice(scene.spawn_points).location)
                    route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
                    local_planner.set_global_plan(route)
                
                while not done:
                    if change_weather:
                        current_weather = scene.world.get_weather()

                        weather = carla.WeatherParameters(
                            sun_altitude_angle=(current_weather.sun_altitude_angle + 0.5) % 180)

                        scene.world.set_weather(weather)

                    if for_demo:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                quit = True
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_q:
                                    quit = True

                    if quit:
                        break

                    steer, throttle, brake = 0, 0, 0

                    speed_limit = vehicle.object.get_speed_limit()
                    gear = vehicle.object.get_control().gear
                    scalars = [vehicle.get_velocity_norm(), speed_limit / 120.0, gear / 8.0]

                    steer, throttle, brake = vehicle.get_autopilot_control(actor, scalars=state[1], image=state[0], command=state[-1], device=device, for_training=True)    
                    action = np.array([steer, throttle, brake])
                    noisy_action = ou_noise(action)

                    if for_demo:
                        logitech.LogiPlaySpringForce(0, int(steer * 100), 50, 80)
                        logitech.logi_update()

                    vehicle.apply_control(carla.VehicleControl(throttle=noisy_action[1], steer=noisy_action[0], brake=noisy_action[2]))

                    scene.run()

                    # step environment and get new state and reward
                    next_state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)

                    reward, done, lane_position, velocity, theta = vehicle.get_reward(scene, scene.collision_history)
                    # store experience in exploration buffer
                    transition = tuple_to_tensordict((state, action, reward, next_state, done))
                    exploration_buffer.add(transition)

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

                    if for_demo:
                        scene.render_steer(steer, x=50, y=75, scale=0.1)

                        text_to_render = {
                            "Speed": f"{vehicle.get_velocity():.1f}",
                            "Speed Limit": f"{int(speed_limit)}",
                            "Steer": f"{steer:.2f}",
                            "Throttle": f"{throttle:.2f}",
                            "Brake": f"{brake:.2f}",
                            "Gear": f"{gear}",
                            "Distance on Autopilot": f"{distance_traveled:.2f} km"
                        }

                        scene.render_text(text_to_render, x=0, y=game_height, anchor="bottomleft")

                        text_to_render = {
                            "Command": 'Left' if command == 0 else 'Center' if command == 1 else 'Right',
                            "Collecting": str(collecting),
                            "Autopilot": str(autopilot),
                            "Train DDPG": str(train_ddpg),
                            "Navigation": str(navigate),
                        }

                        scene.render_text(text_to_render, x=game_width // 2, y=game_height - 100, anchor="midbottom", size=36)

                        scene.update_display()

                # update buff_a based on DDPG rewards vs IL rewards
                if total_reward >= il_max_reward:
                    buff_a = 0 if buff_a < 0.1 else buff_a - 0.1

                # sample from the buffers using buff_a
                expert_samples = TensorDict({}, batch_size=0)
                experience_samples = TensorDict({}, batch_size=0)
                if buff_a == 0:
                    experience_samples = exploration_buffer.sample(batch_size)
                elif len(exploration_buffer) < (1 - buff_a) * batch_size:
                    expert_samples = expert_buffer.sample(batch_size=batch_size)
                else:
                    expert_samples = expert_buffer.sample(batch_size=int(buff_a*batch_size))
                    experience_samples = exploration_buffer.sample(batch_size=int((1-buff_a)*batch_size))
                
                print_formatted(f"buff_a: {buff_a}")
                print_formatted(f"exploration samples size: {experience_samples.batch_size}", RED)
                print_formatted(f"expert samples size: {expert_samples.batch_size}", RED)

                batch_samples = combine_and_shuffle_samples(expert_samples, experience_samples)

                # update the actor and critic models, obtain the loss, and save the models
                critic_loss, actor_loss = learn(batch_samples, device, actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer)
                print_formatted("Actor and critic updated.", GREEN)
                if (i + 1) % 10 == 0:
                    save_model(saved_models_dir, actor, "actor")
                    save_model(saved_models_dir, critic, "critic")
                    save_model(saved_models_dir, actor_optimizer, "actor_optimizer")
                    save_model(saved_models_dir, critic_optimizer, "critic_optimizer")
                
                # store stats for analysis
                episodes_list.append(i)
                reward_per_episode.append(total_reward)
                steps_per_episode.append(num_steps)
                criticLoss_per_episode.append(critic_loss.item())
                actorLoss_per_episode.append(actor_loss.item())

                stats_dict = {
                    "episodes": episodes_list,
                    "rewards": reward_per_episode,
                    "steps": steps_per_episode,
                    "critic_loss": criticLoss_per_episode,
                    "actor_loss": actorLoss_per_episode
                }

                with open("stats.json", "w") as outfile:
                    json.dump(stats_dict, outfile)
                    print_formatted("Stats saved to stats.json", GREEN)

                print("Overall episode stats:")
                print_formatted(f"episode: {i}, reward: {total_reward}, steps: {num_steps}, actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}", YELLOW)

                # print logging
                if (i + 1) % 10 == 0:
                    print("------------------------------------------------------------")
                    print_formatted(f"EPISODES MEAN STATISTCS: {i+1-10}-{i+1}", YELLOW)
                    start_index = i - 10 + 1
                    end_index = i + 1
                    if len(reward_per_episode) > 1:
                        print_formatted(f"Average reward per episode: {avg(reward_per_episode[start_index:end_index]) if (i+1) != num_episodes else avg(reward_per_episode[start_index:])}", YELLOW)
                    else:
                        print_formatted("No reward data to display", YELLOW)
                    if len(steps_per_episode) > 1:
                        print_formatted(f"Average steps per episode: {avg(steps_per_episode[start_index:end_index]) if (i+1) != num_episodes else avg(steps_per_episode[start_index:])}", YELLOW)
                    else:
                        print_formatted("No steps data to display", YELLOW)
                    if len(criticLoss_per_episode) > 1:
                        print_formatted(f"Average critic loss per episode: {avg(criticLoss_per_episode[start_index:end_index]) if (i+1) != num_episodes else avg(criticLoss_per_episode[start_index:])}", YELLOW)
                    else:
                        print_formatted("No critic loss data to display", YELLOW)
                    if len(actorLoss_per_episode) > 1:
                        print_formatted(f"Average actor loss per episode: {avg(actorLoss_per_episode[start_index:end_index]) if (i+1) != num_episodes else avg(actorLoss_per_episode[start_index:])}", YELLOW)
                    else:
                        print_formatted("No actor loss data to display", YELLOW)

                    print("Pausing for a few seconds...")
                    time.sleep(5)  # Pauses for 5 seconds

                if not for_demo and i == num_episodes - 1:
                    print_formatted(f"TRAINING COMPLETE.")
                    running = True
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print_formatted("Exiting...", RED)
        # save_queue.put((None, None, None, None))
        print_formatted("Save thread joined, exiting...", RED)
        scene.cleanup()
        if logitech_detected:
            logitech.steering_shutdown()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Command line arguments for some training configurations. Use -h to view the available arguments.")
    parser.add_argument('-d', action='store_true', help='Use this flag to enable demo mode which opens a pygame window and controls a logitech steering wheel if connected.')
    parser.add_argument('--buffer_size', type=int, help='Pass an integer representing the size of the expert and exploration replay buffer. The default is 100000.')
    parser.add_argument('--std_dev', type=float, help='Pass a float representing the standard deviation value that determines the level of variance in the noisy action selection for DDPG. The default is 0.2.')
    parser.add_argument('--episodes', type=int, help='Pass an integer representing the number of DDPG training episodes. The default is 500.')
    parser.add_argument('--batch_size', type=int, help='Pass an integer representing the batch size of data sampled during training to update the actor and critic. The default is 128.')

    args = parser.parse_args()

    for_demo = args.d if args.d else False
    buffer_size = args.buffer_size if args.buffer_size else 100000
    std_dev = args.std_dev if args.std_dev else 0.2
    num_episodes = args.episodes if args.episodes else 500
    batch_size = args.batch_size if args.batch_size else 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_formatted(f"Using device: {device}", GREEN)   

    # Load the model (or an empty model if it doesn't exist)
    il_model_path = os.environ.get('IL_MODEL')
    il_model = load_model(il_model_path, base_model=IL_Model())

    expert_buffer, il_max_reward = get_expert_data_with_IL(device, il_model)
    train_ddpg(expert_buffer, device, il_model, il_max_reward, buffer_size, std_dev, num_episodes, batch_size, for_demo)