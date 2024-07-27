import torch
from torchrl.data import LazyMemmapStorage, ReplayBuffer
import carla
import os
from dotenv import load_dotenv
import time
from statistics import mean as avg
import logging
import json

load_dotenv()
from model import *
from environment import *
from ou_noise import *
from train_helpers import *
from agents.navigation.global_route_planner import GlobalRoutePlanner


"""MAIN TRAINING LOOP"""

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info("Training DDPG-IL in CARLA")

        # use cuda or cpu for training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        """Initialize Some Environment Variables"""
        logger.info("Initializing the environment and models...")

        env = CarlaScene(town='Town02')
        grp = GlobalRoutePlanner(env.world.get_map(), sampling_resolution=4.0) # global planner for routes in environment

        """Load and initialize models and replay buffers"""
        # loading the IL model
        il_model_path = os.environ.get('IL_MODEL')
        il_model = load_model(il_model_path, device, base_model=IL_Model())
        il_model.to(device)

        # defining dimensions of state and action
        state_dim = (3, 88, 200)
        scalar_dim = 1
        action_dim = 3
        buffer_size = 10000

        # initialize DDPG networks
        actor = Actor(pretrained_model=il_model).to(device)
        target_actor = Actor(pretrained_model=il_model).to(device)
        critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)
        target_critic = Critic(action_dim=action_dim, pretrained_model=il_model).to(device)

        # initialize optimizers and action selection noise
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        mean = np.zeros(action_dim)
        std_dev = 0.2 * np.ones(action_dim)
        ou_noise = OUActionNoise(mean=mean, std_dev=std_dev)

        # initialize DDPG replay buffer
        exploration_dir = "exploration_buffer" # directory where its saved
        make_directiory(exploration_dir)
        storage = LazyMemmapStorage(max_size=buffer_size, scratch_dir=exploration_dir)
        exploration_buffer = ReplayBuffer(storage=storage)

        # run 10 IL rounds to get expert buffer and IL's most reward
        expert_buffer, il_max_reward = get_expert_data_with_IL(env, grp, il_model)

        # initialize buff_a (sampling distribution of expert and exploration buffers)
        buff_a = 1 # can be adjusted with experimentation

        # initialize hyperparameters
        num_episodes = 500
        max_steps_per_episode = 1500
        max_time_per_episode = 300 # in seconds
        batch_size = 32

        # initialize directories for saving progress
        saved_models_dir = "trained_models" # directory creation already exists in save_models method

        # initializations for statistics
        log_and_save_interval = 10
        episodes_list = []
        reward_per_episode = []
        steps_per_episode = []
        actorLoss_per_episode = []
        criticLoss_per_episode = []

        logger.info("Beginning training...")
        for episode in range(num_episodes):
            vehicle, forward_camera, local_planner, start_waypoint, end_waypoint = env.reset()

            route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
            local_planner.set_global_plan(route)

            # execute local planner step to get command for the first step
            local_planner.run_step()
            next_waypoint, wp_direction = local_planner.get_incoming_waypoint_and_direction(steps=4)

            # initialize other variables
            ou_noise.reset()
            done = False
            num_steps = 0
            total_reward = 0

            episode_start_time = time.time()

            # get initial observation
            image_state = forward_camera.get_image_float()
            scalars = [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear]
            command, command_cooldown_counter = get_command_and_cooldown(wp_direction, 0)
            state = (image_state, scalars, command)

            while not done:     
                # generate action using actor model and add noise for exploration
                steer, throttle, brake = vehicle.get_autopilot_control(actor, scalars=state[1], image=state[0],
                                                                       command=state[2], for_training=True) # passes state info through model
                action = np.array([steer, throttle, brake])
                noisy_action = ou_noise(action)

                # apply noisy action to the vehicle
                vehicle.apply_control(carla.VehicleControl(throttle=noisy_action[1], steer=noisy_action[0], brake=noisy_action[2]))

                # execute local planner step after applying control to get new waypoint and command
                local_planner.run_step()
                next_waypoint, wp_direction = local_planner.get_incoming_waypoint_and_direction(steps=4)
                command, command_cooldown_counter = get_command_and_cooldown(wp_direction, command_cooldown_counter)

                # step environment and get new state and reward
                next_state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)
                reward, done = vehicle.get_reward(closest_waypoint=next_waypoint, collision_history=env.get_collision_history())
                total_reward += reward
                num_steps += 1

                # more done conditions
                if num_steps > max_steps_per_episode or (time.time() - episode_start_time) > max_time_per_episode or local_planner.done():
                    done = True 

                # store experience in exploration buffer
                exploration_buffer.add((state, noisy_action, reward, next_state, done))

                # update state
                state = next_state

            # update buff_a based on DDPG rewards vs IL rewards
            if total_reward >= il_max_reward:
                buff_a = 0 if buff_a < 0.1 else buff_a - 0.1

            # sample from the buffers using buff_a
            expert_samples = None
            experience_samples = None
            if buff_a == 0:
                experience_samples = exploration_buffer.sample(batch_size)
            elif len(exploration_buffer) < (1 - buff_a) * batch_size:
                expert_samples = expert_buffer.sample(batch_size=batch_size)
            else:
                expert_samples = expert_buffer.sample(batch_size=buff_a*batch_size)
                experience_samples = exploration_buffer.sample(batch_size=(1-buff_a)*batch_size)

            batch_samples = combine_and_shuffle_samples(expert_samples, experience_samples)

            # update the actor and critic models, obtain the loss, and save the models
            critic_loss, actor_loss = learn(batch_samples, device, actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer)
            logger.info("Actor and critic updated.")
            if (episode + 1) % log_and_save_interval == 0:
                save_model(saved_models_dir, actor, "actor")
                save_model(saved_models_dir, critic, "critic")
                save_model(saved_models_dir, actor_optimizer, "actor_optimizer")
                save_model(saved_models_dir, critic_optimizer, "critic_optimizer")

            # store stats for analysis
            episodes_list.append(episode+1)
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
                logger.info("Stats saved to stats.json")

            # print logging
            if (episode + 1) % log_and_save_interval == 0:
                logger.info("-------------------------------------")
                logger.info(f"EPISODES: {episode+1-log_and_save_interval}-{episode+1}")
                start_index = episode - log_and_save_interval + 1
                end_index = episode + 1
                logger.info(f"Average reward per episode: {avg(reward_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(reward_per_episode[start_index:])}")
                logger.info(f"Average steps per episode: {avg(steps_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(steps_per_episode[start_index:])}")
                logger.info(f"Average critic loss per episode: {avg(criticLoss_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(criticLoss_per_episode[start_index:])}")
                logger.info(f"Average actor loss per episode: {avg(actorLoss_per_episode[start_index:end_index]) if (episode+1) != num_episodes else avg(actorLoss_per_episode[start_index:])}")
            else:
                logger.info(f"episode: {episode}, reward: {total_reward}, steps: {num_steps}, actor loss: {actor_loss.item()}, critic loss: {critic_loss.item()}")

        env.cleanup()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
