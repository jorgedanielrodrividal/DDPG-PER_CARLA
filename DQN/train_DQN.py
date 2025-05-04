import json 
import pickle

import os
import sys
import random
import time
import numpy as np
import cv2
import math
from datetime import date
import glob

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm


import carla_config as settings
from agent_model import DQNAgent
from carla_env import CarEnv

def find_latest_checkpoint(checkpoint_dir):
    model_paths = glob.glob(os.path.join(checkpoint_dir, "model_ep_*.model"))
    if not model_paths:
        return None
    episodes = [int(p.split("_ep_")[1].split(".")[0]) for p in model_paths]
    return max(episodes)
# Own Tensorboard class

if __name__ == '__main__':
    distance_acum = []
    epsilon = settings.epsilon
    FPS = 60
    # For stats
    ep_rewards = [-200]
    # tf.config.optimizer.set_jit(True)
    # For more repetitive results
    # random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    gpu_options = tf.GPUOptions(allow_growth=True)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    latest_ep = find_latest_checkpoint(checkpoints_dir)
    resume = latest_ep is not None
    # Create models folder
    # if not os.path.isdir('models'):
    #     os.makedirs('models')
    # print("Antes de create agent")
    # Create agent and environment
    agent = DQNAgent()


    # === Resume Training Support ===
    if resume:
        print(f"[INFO] Resuming training: latest saved checkpoint episode {latest_ep}")
        epsilon_path = os.path.join(checkpoints_dir, f"meta_ep_{latest_ep}.json")
        buffer_path = os.path.join(checkpoints_dir, f"buffer_ep_{latest_ep}.pkl")
        model_path = os.path.join(checkpoints_dir, f"model_ep_{latest_ep}.model")

        if os.path.exists(model_path):
            agent.model.load_weights(model_path)
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                agent.replay_memory = pickle.load(f)
        if os.path.exists(epsilon_path):
            with open(epsilon_path, 'r') as f:
                metadata = json.load(f)
                start_episode = metadata.get("episode", latest_ep)
                # print(f"Starting episode is: {start_episode + 1}")
                epsilon = metadata.get("epsilon", epsilon)
        else:
            start_episode = latest_ep + 1  # fallback
    else:
        print("[INFO] Starting training from scratch")
        start_episode = 0


    # # === INITIALIZE ROUTE TYPE BEFORE TRAINING STARTS ===
    # if settings.TRAIN_MODE == "RANDOM_TURN":
    #     settings.TRAIN_MODE = random.choice(["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"])
    #     print(f"[TRAIN_MODE_INITIALIZATION] First selected maneuver: {settings.TRAIN_MODE}")


    env = CarEnv()
    # print("Despues de agente y environment")
    date_title = date.today()
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        # print("Esperando inicializacion de agente")
        time.sleep(0.01)
    # print("Antes de get_qs")

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # agent.get_qs(np.ones((env.im_height, env.im_width, IM_LAYERS)))

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        agent.get_qs(np.ones(settings.state_dim, ))
    else:
        agent.get_qs(np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS)))


    # Iterate over episodes
    # for episode in tqdm(range(1, settings.EPISODES + 1), ascii=True, unit='episodes'):
    for episode in tqdm(range(start_episode + 1, settings.EPISODES + 1),
                        initial=start_episode + 1,
                        total=settings.EPISODES, 
                        ascii=True, 
                        unit='episodes'):

        # === RANDOMIZE TRAIN_MODE BEFORE RESET ===
        if settings.TRAIN_MODE_OPTIONS[7] == "RANDOM_TURN":
            settings.TRAIN_MODE = random.choice(["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"])
            print(f"[TRAIN_MODE] Episode {episode}: {settings.TRAIN_MODE}")

        
        # # Randomize maneuver at the beginning of the episode
        # if settings.TRAIN_MODE == "RANDOM_TURN":
        #     settings.TRAIN_MODE = random.choice(["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"])
        #     print(f"[TRAIN_MODE] Randomly selected for next episode: {settings.TRAIN_MODE}")

        
        # try:
        env.collision_hist = []
        env.crossline_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            # We are only interested in waypoints, not images
            _, state_train = env.reset()
        else:
            # We are only interested in images, not waypoints
            state_train, _ = env.reset()
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()


        done = False
        episode_start = time.time()
        # Play for given number of seconds only

        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action_vector = agent.get_qs(state_train)
                action = np.argmax(action_vector)
                #print(settings.ACTIONS_NAMES[action])
                # print(settings.ACTIONS_NAMES[action])

            else:
                # Get random action
                # print("Accion random")
                action = np.random.randint(0, settings.N_actions)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0]:
            #     [_, new_state_train], reward, done, _ = env.step(action)
            # elif settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            #     new_image, reward, done, info = env.step(action)
            #     new_state_train = env.Calcular_estado(new_image)

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                # print("We are training only with waypoint info.")
                [_, new_state_train], reward, done, _ = env.step(action)
            else:
                [new_state_train, _], reward, done, _ = env.step(action)
                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()

            #print('Action: ', ACTIONS_NAMES[action], ' Reward: ', reward)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((state_train, action, reward, new_state_train, done))


            state_train = new_state_train

            # NOT HERE: it'd run EVERY STEP WITHIN THE SAME EPISODE!
            # Restore the flag at the end of the episode
            # settings.TRAIN_MODE = "RANDOM_TURN"
            
            step += 1
            if done:
                break


        # === RESTORE RANDOM_TURN FOR NEXT EPISODE ===
        if settings.TRAIN_MODE_OPTIONS[7] == "RANDOM_TURN":
            settings.TRAIN_MODE = "RANDOM_TURN"
        
        #print(agent.model.get_weights())
        #json_wei = agent.model.to_json()
        #print(json_wei)


        # End of episode - destroy agents
        # for actor in env.actor_list:
        #     actor.destroy()
        for actor in env.actor_list:
            if actor is not None:
                try:
                    actor.destroy()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"[WARNING] Failed to destroy actor {actor.id}: {e}")



        # Decay epsilon
        if epsilon > settings.MIN_EPSILON:
            epsilon *= settings.EPSILON_DECAY
            epsilon = max(settings.MIN_EPSILON, epsilon)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if (episode > 1) and ((episode % settings.AGGREGATE_STATS_EVERY) == 0) or (episode == 2):
            average_reward = sum(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-settings.AGGREGATE_STATS_EVERY:])
            average_dist = sum(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:]) / len(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           efshowpsilon=epsilon, avegare_dist=average_dist)



        # === SAVE CHECKPOINT AND BUFFER ===
        if episode % settings.env_params['save_freq'] == 0:
            model_save_path = os.path.join(checkpoints_dir, f"model_ep_{episode}.model")
            buffer_save_path = os.path.join(checkpoints_dir, f"buffer_ep_{episode}.pkl")
            meta_save_path = os.path.join(checkpoints_dir, f"meta_ep_{episode}.json")
            agent.model.save(model_save_path)
            with open(buffer_save_path, 'wb') as f:
                pickle.dump(agent.replay_memory, f)
                print(f"Replay buffer checkpoint saved.")
            with open(meta_save_path, 'w') as f:
                json.dump({"episode": episode, "epsilon": epsilon}, f)
                print(f"Epsilon checkpoint saved.")

        


        # Ensure folder exists
        save_dir = settings.AGENT_PATH
        os.makedirs(save_dir, exist_ok=True)
        
        #Guardar datos del entrenamiento en ficheros
        if episode % 3 == 0:
            agent.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_model.model"))
            # agent.model.save(save_dir + str(settings.TRAIN_MODE) + "_model.model")
        if episode % settings.N_save_stats == 0:
            agent.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{episode}_model.model"))
            # agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_" + str(episode) + "_model.model")
        if (episode > 10) and (episode_reward > np.max(ep_rewards[:-1])):
            agent.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_best_reward_ep_{episode}_model.model"))
            # agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_best_reward_model.model")
        # Save model if the agent reaches the target
        if env.summary.get('Target', 0) > 0:
            agent.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_reached_goal_ep_{episode}_model.model"))
            print(f"[SAVED] Model saved for reaching goal at episode {episode}")
            # Optional: reset counter so we only save *once* per episode
            env.summary['Target'] = 0

        acum = 0

        # # Randomize maneuver at the beginning of the episode
        # if settings.TRAIN_MODE == "RANDOM_TURN":
        #     settings.TRAIN_MODE = random.choice(["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"])
        #     print(f"[TRAIN_MODE] Randomly selected for next episode: {settings.TRAIN_MODE}")


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE) + "_last_model.model")