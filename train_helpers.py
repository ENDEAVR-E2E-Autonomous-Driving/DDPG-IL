import os
from torchrl.data import LazyMemmapStorage, ReplayBuffer
import carla
import numpy as np
import torch
import random
from model import *
from dotenv import load_dotenv
load_dotenv()
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


"""TRAINING HELPERS"""
# ------------------------------------------------------------------------------------------- 

# helper function for getting updated command and command cooldown counter
def get_command_and_cooldown(wp_direction, command_cooldown_counter):
    """
    Adjusts the command and command cooldown counter based on the current waypoint direction.
    Returns an updated command and command cooldown counter.
    """
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
def make_directiory(dir_name):
    if not os.path.exists:
        os.makedirs(dir_name, exist_ok=True)


# helper function for creating an expert replay buffer with IL
def get_expert_data_with_IL(env, grp: GlobalRoutePlanner, il_model: IL_Model):
    """
    Runs a trained IL model for 10 rounds to collect and store its expert experiences.
    Returns an expert replay buffer and the maximum reward achieved.
    """
    il_max_reward = float('-inf') # get max reward when running the IL rounds
    exper_dir = "expert_buffer" # directory where its saved
    make_directiory(exper_dir)
    storage = LazyMemmapStorage(max_size=10000, scratch_dir=exper_dir)
    expert_buffer = ReplayBuffer(storage=storage)
    for _ in range(10):
        # reset environment and vehicle
        vehicle, forward_camera, local_planner, start_waypoint, end_waypoint = env.reset()
        
        route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        local_planner.set_global_plan(route)

        # execute local planner step to get command for the first step
        local_planner.run_step()
        next_waypoint, wp_direction = local_planner.get_incoming_waypoint_and_direction(steps=4)

        done = False
        total_reward = 0

        command, command_cooldown_counter = get_command_and_cooldown(wp_direction, 0)
        state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)

        while not done:
            steer, throttle, brake = vehicle.get_autopilot_control(il_model, scalars=state[1], image=state[0], command=state[2])
            action = np.array([steer, throttle, brake])
            vehicle.apply_control(carla.VehicleControl(throttle=action[1], steer=action[0], brake=action[2]))

            local_planner.run_step()
            next_waypoint, wp_direction = local_planner.get_incoming_waypoint_and_direction(steps=4)
            command, command_cooldown_counter = get_command_and_cooldown(wp_direction, command_cooldown_counter)

            next_state = (forward_camera.get_image_float(), [vehicle.get_velocity_norm(), vehicle.object.get_speed_limit(), vehicle.object.get_control().gear], command)
            reward, done = vehicle.get_reward(next_waypoint)
            total_reward += reward

            if local_planner.done():
                done = True

            expert_buffer.add((state, action, reward, next_state, done))
            state = next_state

        il_max_reward = max(il_max_reward, total_reward)
        print(f"IL Round {_} Complete")

    return expert_buffer, il_max_reward


# helper function to combine samples from the exploration and expert buffers
def combine_and_shuffle_samples(expert_batch, exploration_batch):
    """
    Mixes samples from the expert and exploration batches 
    """
    # concatenate them first
    state1, action1, reward1, next_state1, done1 = expert_batch if expert_batch is not None else [], [], [], [], []
    state2, action2, reward2, next_state2, done2 = exploration_batch if exploration_batch is not None else [], [], [], [], []

    states = torch.cat((state1, state2), dim=0)
    actions = torch.cat((action1, action2), dim=0)
    rewards = torch.cat((reward1, reward2), dim=0)
    next_states = torch.cat((next_state1, next_state2), dim=0)
    dones = torch.cat((done1, done2), dim=0)

    # stack experiences into single list of tuples
    experiences = list(zip(states, actions, rewards, next_states, dones))

    # shuffle tuples
    random.shuffle(experiences)

    # unzip the list of tuples back to separate components
    states, actions, rewards, next_states, dones = zip(*experiences)

    return states, actions, rewards, next_states, dones


# helper function to update actor and critic
def learn(batch_samples, device, actor: Actor, target_actor: Actor, critic: Critic, target_critic: Critic, \
          actor_optimizer: torch.optim.Adam, critic_optimizer: torch.optim.Adam):
    
    states, actions, rewards, next_states, dones = batch_samples
    
    # convert to tensors
    states_img = torch.stack([s[0] for s in states]).to(device)
    states_scalar = torch.stack([torch.tensor(s[1]) for s in states]).to(device)
    states_cmd = torch.tensor([s[2] for s in states], dtype=torch.int64).to(device)

    next_states_img = torch.stack([s[0] for s in next_states]).to(device)
    next_states_scalar = torch.stack([torch.tensor(s[1]) for s in next_states]).to(device)
    next_states_cmd = torch.tensor([s[2] for s in next_states], dtype=torch.int64).to(device)

    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # update critic
    gamma = 0.99
    with torch.no_grad():
        next_actions = target_actor.forward(next_states_img, next_states_scalar, next_states_cmd)
        next_q_values = target_critic.forward(next_states_img, next_states_scalar, next_states_cmd, next_actions)
        target_q_values = rewards + gamma * next_q_values
    
    q_values = critic.forward(states_img, states_scalar, states_cmd, actions)
    critic_loss = torch.nn.functional.mse_loss(q_values, target_q_values)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # update actor
    predicted_actions = actor.forward(states_img, states_scalar, states_cmd)

    # negate the objective to convert the maximizaiton problem into a minimization problem 
    actor_loss = -critic.forward(states_img, states_scalar, states_cmd, predicted_actions).mean() # maximizing reward

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # soft update target networks
    tau = 0.005
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    return critic_loss, actor_loss

# ------------------------------------------------------------------------------------------- 
