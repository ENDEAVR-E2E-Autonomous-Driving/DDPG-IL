import torch
from torchrl.data import LazyMemmapStorage, ReplayBuffer
import carla
import os
from dotenv import load_dotenv
import time
from statistics import mean as avg
import logging
import json
from utils import *

load_dotenv()
from model import *
from environment import CarlaScene, CarlaCamera
from ou_noise import OUActionNoise
from train_helpers import *
from agents.navigation.global_route_planner import GlobalRoutePlanner
import pygame
from input import InputManager
import logidrivepy

"""MAIN TRAINING LOOP"""
def train_DDPG_IL(scene:CarlaScene, vehicle:CarlaVehicle, input_manager:InputManager, forward_camera = None, run_il_rounds=True, expert_buffer=None, il_model=None):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logitech_detected = True

    # Open a window if not already opened
    # if scene.display is None:
    #     scene.open_window(w=800, h=600)

    try:
        logger.info("Training DDPG-IL in CARLA")

        # use cuda or cpu for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_formatted(f"Using device: {device}", GREEN)

        """Initialize Some Environment Variables"""
        print_formatted("Initializing the environment and models...", GREEN)

        # host_ip_address = os.environ.get("os.environ.get")
        # env = CarlaScene(town='Town02')
        scene = CarlaScene('Town02') if scene is None else scene

        scene.set_spectator_position(height=200.0) # set birds-eye view position for spectator

        """Load and initialize models and replay buffers"""
        # loading the IL model
        if not il_model:    
            il_model_path = os.environ.get('IL_MODEL')
            il_model = load_model(il_model_path, base_model=IL_Model())
        # il_model.to(device)

        # loading the actor model for testing
        # actor_model_path = os.environ.get('ACTOR_MODEL')
        # actor_model = load_model(actor_model_path, base_model=Actor(il_model))

        # defining dimensions of state and action
        state_dim = (3, 88, 200)
        scalar_dim = 1
        action_dim = 3
        buffer_size = 100000

        # initialize DDPG networks
        actor = Actor(pretrained_model=il_model).to(device)
        target_actor = Actor(pretrained_model=il_model).to(device)
        critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)
        target_critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)

        # initialize optimizers and action selection noise
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        mean = np.zeros(action_dim)
        std_dev = 0.02 * np.ones(action_dim) 
        ou_noise = OUActionNoise(mean=mean, std_dev=std_dev)

        # initialize DDPG replay buffer
        print_formatted("Initializing the exploration replay buffer", GREEN)
        exploration_dir = "exploration_buffer" # directory where its saved
        previous_exploration_data_exists = False # flag denotes whether to initialize with first transition or not

        make_directory(exploration_dir)
        storage_exploration = LazyMemmapStorage(max_size=buffer_size, scratch_dir=exploration_dir)
        exploration_buffer = ReplayBuffer(storage=storage_exploration)
        print_formatted("Exploration buffer loaded.", GREEN)

        # run 10 IL rounds to get expert buffer and IL's most reward
        print_formatted("Initializing the expert replay buffer", GREEN)

        if run_il_rounds:
            expert_buffer, il_max_reward = get_expert_data_with_IL(scene, device, buffer_size, il_model)
            with open("il_max_reward.json", "w") as outfile:    
                il_max_reward_dict = {'il_max_reward': il_max_reward}
                json.dump(il_max_reward_dict, outfile)
            print_formatted("Expert buffer directory and IL max reward JSON created", GREEN)
        else:
            print_formatted("Using provided expert buffer", GREEN)

        # initialize buff_a (sampling distribution of expert and exploration buffers)
        # buff_a = 1 # can be adjusted with experimentation
        buff_a = 0.8

        # initialize hyperparameters
        num_episodes = 500
        max_steps_per_episode = 2000
        max_time_per_episode = 300 # in seconds
        batch_size = 64

        # initialize directories for saving progress
        saved_models_dir = "model_weights" # directory creation already exists in save_models method

        # initializations for statistics
        log_and_save_interval = 10
        episodes_list = []
        reward_per_episode = []
        steps_per_episode = []
        actorLoss_per_episode = []
        criticLoss_per_episode = []

        print_stats_interval = 50
        max_speed = 14 # max speed in m/s normalized

        print_formatted("Checking if previous training data is stored...", GREEN)
        # if os.path.exists("stats.json"):
        #     with open("stats.json") as file:
        #         stats = json.load(file)
        #         start_episode = stats['episodes'][-1]
        # else:
        #     start_episode = 0

        print_formatted("Beginning training...", YELLOW)

        # Initialize the vehicle once

        try:
            logitech = logidrivepy.LogitechController()
            logitech.steering_initialize()
            print("Logitech initialized")
        except:
            print("Logitech not detected")
            logitech_detected = False

        quit = False

        # Reset vehicle state for each episode
        # vehicle = scene.add_car('vehicle.mercedes.sprinter')
        vehicle.reset(scene)
        vehicle.draw_vehicle_location(scene, 6.0)
        # scene.add_collision_sensor(vehicle.object)

        # # Add cameras to the scene
        # # x, z = get_camera_placement_coordinates(vehicle=vehicle)
        # forward_camera = CarlaCamera(vehicle.object, x=2.3)
        # scene.add_camera(forward_camera)    
        # game_camera = CarlaCamera(vehicle.object, x=2.3, w=1920, h=1080, fov=110)
        # scene.add_game_camera(game_camera)  

        scene.open_window(w=1920, h=1080)

        for episode in range(num_episodes):
            print("--------------------------------------------------")
            print_formatted(f"Episode {episode}", GRAY)
            # scene.update_display()
            # Handle Pygame events
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        quit = True
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == input_manager.joystick_config["quit"]:
                        quit = True
            if quit:
                break

            
            # Initialize other variables
            done = False
            total_reward = 0
            num_steps = 0
            command = 1
            command_cooldown_counter = 0

            episode_start_time = time.time()

            scene.run()
            state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)

            while not done:     
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quit = True
                    # elif event.type == pygame.JOYBUTTONDOWN:
                    #     if event.button == input_manager.joystick_config["quit"]:
                    #         quit = True


                # generate action using actor model and add noise for exploration
                steer, throttle, brake = vehicle.get_autopilot_control(actor, scalars=state[1], image=state[0], command=state[-1], device=device, for_training=True)
                action = np.array([steer, throttle, brake])
                noisy_action = ou_noise(action)
 
                # apply noisy action to the vehicle
                vehicle.apply_control(carla.VehicleControl(throttle=noisy_action[1], steer=noisy_action[0], brake=noisy_action[2]))

                # execute local planner and env step after applying control to get new waypoint and command, and sync actors to env
                scene.run()
                # local_planner.run_step()

                vehicle.draw_vehicle_location(scene, 0.1)

                # step environment and get new state and reward
                next_state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)
                # reward, done, lane_position, velocity, curvature, reward_details, theta, axial_deviation, lateral_deviation = vehicle.get_reward(next_waypoint, local_planner, scene.collision_history, noisy_action[1], num_steps)

                reward, done, lane_position, velocity, theta = vehicle.get_reward(scene, scene.collision_history)
                # store experience in exploration buffer
                transition = tuple_to_tensordict((state, action, reward, next_state, done))
                exploration_buffer.add(transition)

                # more done conditions
                current_time = time.time()

                if num_steps % print_stats_interval == 0:
                    print_formatted(f"Steps: {num_steps}, episode time: {time.time() - episode_start_time}, reward: {reward},", RED)
                    print_formatted(f"lane deviation: {lane_position}, velocity: {velocity}, theta: {theta}", RED)
  
                # more done conditions
                if num_steps >= max_steps_per_episode:
                    done = True
                    print("Episode finished: max steps")
                if current_time - episode_start_time >= max_time_per_episode:
                    done = True
                    print("Episode finished: max time")

                # update state
                state = next_state
                total_reward += reward
                num_steps += 1

                if logitech_detected:
                    logitech.LogiPlaySpringForce(0, int(steer * 100), 50, 80)
                    logitech.logi_update()

                # Update the display
                scene.update_display()

            ou_noise.decay_noise() # decay the noise after each episode to reduce exploration

            vehicle.reset(scene)
            scene.collision_history.clear()
            # update buff_a based on DDPG rewards vs IL rewards
            # if total_reward >= il_max_reward:
            #     buff_a = 0 if buff_a < 0.1 else buff_a - 0.1

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
            
            print_formatted(f"exploration samples size: {experience_samples.batch_size}", RED)
            print_formatted(f"expert samples size: {expert_samples.batch_size}", RED)

            batch_samples = combine_and_shuffle_samples(expert_samples, experience_samples)

            # update the actor and critic models, obtain the loss, and save the models
            critic_loss, actor_loss = learn(batch_samples, device, actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer)
            print_formatted("Actor and critic updated.", GREEN)
            if (episode + 1) % log_and_save_interval == 0:
                save_model(saved_models_dir, actor, "actor")
                save_model(saved_models_dir, critic, "critic")
                save_model(saved_models_dir, actor_optimizer, "actor_optimizer")
                save_model(saved_models_dir, critic_optimizer, "critic_optimizer")

            # store stats for analysis
            episodes_list.append(episode)
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

            # print logging
            if (episode + 1) % log_and_save_interval == 0:
                print("------------------------------------------------------------")
                print_formatted(f"EPISODES MEAN STATISTCS: {episode+1-log_and_save_interval}-{episode+1}", YELLOW)
                start_index = episode - log_and_save_interval + 1
                end_index = episode + 1
                if len(reward_per_episode) > 1:
                    print_formatted(f"Average reward per episode: {avg(reward_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(reward_per_episode[start_index:])}", YELLOW)
                else:
                    print_formatted("No reward data to display", YELLOW)
                if len(steps_per_episode) > 1:
                    print_formatted(f"Average steps per episode: {avg(steps_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(steps_per_episode[start_index:])}", YELLOW)
                else:
                    print_formatted("No steps data to display", YELLOW)
                if len(criticLoss_per_episode) > 1:
                    print_formatted(f"Average critic loss per episode: {avg(criticLoss_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(criticLoss_per_episode[start_index:])}", YELLOW)
                else:
                    print_formatted("No critic loss data to display", YELLOW)
                if len(actorLoss_per_episode) > 1:
                    print_formatted(f"Average actor loss per episode: {avg(actorLoss_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(actorLoss_per_episode[start_index:])}", YELLOW)
                else:
                    print_formatted("No actor loss data to display", YELLOW)

                print("Pausing for a few seconds...")
                time.sleep(5)  # Pauses for 5 seconds
            else:
                print("Overall episode stats:")
                print_formatted(f"episode: {episode}, reward: {total_reward}, steps: {num_steps}, actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}", YELLOW)


        # scene.cleanup()

    finally:
        # logger.error(f"An error occurred: {e}", exc_info=True)
        print_formatted("Training complete.", GREEN)
        print_formatted("cleaning up...", RED)
        scene.cleanup()
        print_formatted("all cleaned up", RED)
        if logitech_detected:
            logitech.steering_shutdown()

# if __name__ == '__main__':
#     main()
