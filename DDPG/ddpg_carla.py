import os
import glob
import json
import pickle
from tqdm import tqdm

import argparse
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow as tf
from carla_env import CarEnv
from actor import ActorNetwork
from critic import CriticNetwork
from actor_CNN import ActorNetwork_CNN
from critic_CNN import CriticNetwork_CNN
# from DDPG.carla_env import CarEnv
# from DDPG.actor import ActorNetwork
# from DDPG.critic import CriticNetwork
# from DDPG.actor_CNN import ActorNetwork_CNN
# from DDPG.critic_CNN import CriticNetwork_CNN

from keras.callbacks import TensorBoard
# from util.noise import OrnsteinUhlenbeckActionNoise
# from replay_buffer_PER import PrioritizedReplayBuffer
from replay_buffer import ReplayBuffer
# from DDPG.replay_buffer import ReplayBuffer
import cv2
import carla_config as settings
from keras.models import load_model

time_buff = []

AGGREGATE_STATS_EVERY = 10

# class OrnsteinUhlenbeckNoise:
#     def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
#         self.size = size
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.reset()

#     def reset(self):
#         self.state = np.ones(self.size) * self.mu

#     def sample(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
#         self.state = self.state + dx
#         return self.state


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# def find_latest_checkpoint(checkpoint_dir, train_mode):
#     pattern = os.path.join(checkpoint_dir, f"{train_mode}_*_actor.h5")
#     model_paths = glob.glob(pattern)
#     if not model_paths:
#         return None
#     episodes = [int(p.split(f"{train_mode}_")[1].split("_")[0]) for p in model_paths]
#     return max(episodes)

def find_latest_checkpoint(checkpoint_dir, train_mode):
    pattern = os.path.join(checkpoint_dir, f"{train_mode}_*_actor.model")
    model_paths = glob.glob(pattern)
    if not model_paths:
        return None
    episodes = []
    for p in model_paths:
        filename = os.path.basename(p)
        parts = filename.replace(".model", "").split(f"{train_mode}_")
        if len(parts) > 1:
            episode_part = parts[1].split("_")[0]
            if episode_part.isdigit():  # Only accept real episode numbers
                episodes.append(int(episode_part))
    if not episodes:
        return None
    return max(episodes)




def play(train_indicator):

    # ou_sigma = 0.3
    tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_{settings.WORKING_MODE}/{settings.TRAIN_MODE}-{int(time.time())}")
    step = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.Session(config=config)

    keras_backend.set_session(tf_session)

    # === Checkpoint loading for resume training ===
    checkpoints_dir = settings.AGENT_PATH
    os.makedirs(checkpoints_dir, exist_ok=True)

    latest_ep = find_latest_checkpoint(checkpoints_dir, settings.TRAIN_MODE)
    resume = latest_ep is not None

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
            or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
        actor = ActorNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=2,
                             tau=settings.tau, lr=settings.lra)
        critic = CriticNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=2,
                               tau=settings.tau, lr=settings.lrc)
    else:
        actor = ActorNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lra)
        critic = CriticNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lrc)

    buffer = ReplayBuffer(settings.buffer_size)
    # buffer = PrioritizedReplayBuffer(settings.buffer_size)

    if resume:
        print(f"[INFO] Resuming training: latest saved checkpoint episode {latest_ep}")
        episode_path = os.path.join(checkpoints_dir, f"meta_ep_{latest_ep}.json")
        actor_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{latest_ep}_actor.model")
        critic_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{latest_ep}_critic.model")
        buffer_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{latest_ep}_buffer.pkl")

        # Load model weights and episode
        if os.path.exists(actor_path):
            actor.model.load_weights(actor_path)
            actor.target_model.load_weights(actor_path)
        if os.path.exists(critic_path):
            critic.model.load_weights(critic_path)
            critic.target_model.load_weights(critic_path)
        if os.path.exists(episode_path):
            with open(episode_path, 'r') as f:
                metadata = json.load(f)
                start_episode = metadata.get("episode", latest_ep)
                ep_rewards = metadata.get("ep_rewards", [])
                print(f"Loaded episode rewards: {ep_rewards}")

        # Load replay buffer
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                buffer = pickle.load(f)

        # start_episode = latest_ep + 1
    else:
        print("[INFO] Starting training from scratch")
        start_episode = 0
        ep_rewards = []

    env = CarEnv()

    # Create Ornstein-Uhlenbeck noise for exploration
    # ou_noise = OrnsteinUhlenbeckNoise(size=2, mu=0.0, theta=0.15, sigma=0.2)

    # noise function for exploration
    # ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))

    # Torcs environment - throttle and gear change controlled by client


    # try:
    #     # print(settings.critic_weights_file)
    #     actor.model.load_weights(settings.actor_weights_file)
    #     critic.model.load_weights(settings.critic_weights_file)
    #     actor.target_model.load_weights(settings.actor_weights_file)
    #     critic.target_model.load_weights(settings.critic_weights_file)
    #     print("Weights loaded successfully")
    # except:
    #     print("Cannot load weights")

    # for i in range(settings.episodes_num):
    for i in tqdm(range(start_episode + 1, settings.episodes_num + 1),
                  initial=start_episode + 1,
                  total=settings.episodes_num,
                  ascii=True,
                  unit='episodes'):

        tensorboard.step = i
        all_td_errors = []
        print("Episode : %s Replay Buffer %s" % (i, len(buffer)))

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
                or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            _, state = env.reset()
            # ou_noise.reset()

        else:
            current_state, _ = env.reset()
            # ou_noise.reset()
            if settings.IM_LAYERS == 1:
                current_state = np.expand_dims(current_state, -1)
            current_state = current_state / 255
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                current_state = current_state.flatten()

        if settings.GUARDAR_DATOS == 1:
        #Guardar los waypoints
            np.savetxt('Waypoints/DDPG_wp_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(i) + '.txt', env.waypoints_txt, delimiter=';')

        total_reward = 0.0
        for j in range(settings.max_steps):
            tm1 = time.time()

            loss = 0
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] \
                    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                action_predicted = actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
                # action_predicted = action_predicted[0] + ou_noise.sample()
                # action_predicted = np.clip(action_predicted, -1, 1)  # clip to [-1, 1]
                [new_image, new_state], reward, done, info = env.step(action_predicted[0])
                buffer.add((state, action_predicted[0], reward, new_state, done))  # add replay buffer
                # print(new_state)

            else:
                input_to_actor = np.transpose(current_state, (1, 0, 2))  # Swap H and W
                action_predicted = actor.model.predict(input_to_actor.reshape(1, *input_to_actor.shape))
                # action_predicted = actor.model.predict(np.array(current_state).reshape(-1, *current_state.shape))  # + ou()  # predict and add noise
                # action_predicted = action_predicted[0] + ou_noise.sample()
                # action_predicted = np.clip(action_predicted, -1, 1)  # clip to [-1, 1]
                [new_current_state, _], reward, done, info = env.step(action_predicted[0])
                if settings.IM_LAYERS == 1:
                    new_current_state = np.expand_dims(new_current_state, -1)
                new_current_state = new_current_state / 255
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_current_state = new_current_state.flatten()
                buffer.add((current_state, action_predicted[0], reward, new_current_state, done))  # add replay buffer


            # batch update
            batch = buffer.get_batch(settings.batch_size)
            # batch, indices = buffer.get_batch(settings.batch_size)

            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.zeros((len(batch), 1))
            #try:
            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[3]:
            #     # Transpose all image states
            #     new_states = np.transpose(new_states, (0, 2, 1, 3))  # assuming shape = (batch, H, W, C)
            #     # Then run predictions
            #     target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            # else:
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])


            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + settings.gamma * target_q_values[k]
            
            # # Compute TD errors
            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[3]:
            #     states = np.transpose(states, (0, 2, 1, 3))
            #     current_q_values = critic.model.predict([states, actions])
            # else:
            #     current_q_values = critic.model.predict([states, actions])
            # td_errors = y_t.squeeze() - current_q_values.squeeze()
            # # .extend() adds each element individually into the list, not as a sublist.
            # all_td_errors.extend(np.abs(td_errors))  # accumulate td_errors batch by batch

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.get_gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target_model()
                critic.train_target_model()

            # Update priorities
            # buffer.update_priorities(indices, td_errors)

            total_reward += reward

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
                    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                state = new_state
            # elif settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            #     state = state1
            else:
                current_state = new_current_state

            # Imprimir estadísticas cada step
            print(f"Predicted action in step {step} is: {action_predicted[0]}")
            # print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))

            step += 1
            if done:
                print(env.summary)
                # Imprimir estadisticas cada episode
                # print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))
                break


        # Compute mean TD-error over the last episode
        # mean_td_error = np.mean(all_td_errors)
        
        #Guardar datos en tensorboard
        ep_rewards.append(total_reward)
        if (i > 0) and ((i % AGGREGATE_STATS_EVERY == 0) or (i ==1)):
            average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_dist = np.mean(env.distance_acum[-AGGREGATE_STATS_EVERY:])
            tensorboard.update_stats(average_reward=average_reward, 
                                     min_reward=min_reward, 
                                     max_reward=max_reward,
                                     distance=average_dist, 
                                     loss=loss) # mean_td_error=mean_td_error

        # Ensure folder exists
        save_dir = settings.AGENT_PATH
        os.makedirs(save_dir, exist_ok=True)
        
        #Guardar datos del entrenamiento en ficheros
        if i % 3 == 0 and train_indicator:
            actor.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_actor.model"))
            critic.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_critic.model"))
            # actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
            # critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)
        if i % settings.N_save_stats == 0 and train_indicator:
            actor.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_actor.model"))
            critic.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_critic.model"))
            # actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_actor.h5", overwrite=True)
            # critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_critic.h5", overwrite=True)
        if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
            actor.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_best_reward_actor.model"))
            critic.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_best_reward_critic.model"))
            # actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_actor.h5", overwrite=True)
            # critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_critic.h5", overwrite=True)
        # --- Save model if agent reaches target ---
        if env.summary.get('Target', 0) > 0:
            actor.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_reached_goal_actor.model"))
            critic.model.save(os.path.join(save_dir, f"{settings.TRAIN_MODE}_{i}_reached_goal_critic.model"))
            print(f"[SAVED] Model reached goal at episode {i}")
            env.summary['Target'] = 0  # Reset counter

        # --- SAVE CHECKPOINT AND BUFFER ---
        if i % settings.N_save_stats == 0 and train_indicator:
            actor_ckpt_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{i}_actor.model")
            critic_ckpt_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{i}_critic.model")
            buffer_ckpt_path = os.path.join(checkpoints_dir, f"{settings.TRAIN_MODE}_{i}_buffer.pkl")
            meta_ckpt_path = os.path.join(checkpoints_dir, f"meta_ep_{i}.json")

            actor.model.save(actor_ckpt_path, overwrite=True)
            critic.model.save(critic_ckpt_path, overwrite=True)
            with open(buffer_ckpt_path, 'wb') as f:
                pickle.dump(buffer, f)
                print(f"[Checkpoint] Saved replay buffer for episode {i}")
            with open(meta_ckpt_path, 'w') as f:
                json.dump({"episode": i, "ep_rewards" : ep_rewards}, f)
                print(f"[Checkpoint] Saved metadata for episode {i}")
        
        if settings.GUARDAR_DATOS == 1:
        #Guardar la trayectoria recorrida
            np.savetxt('Trayectorias/DDPG_trayectoria_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(i) + '.txt', env.position_array, delimiter=';')
            env.position_array = []

        time_buff.append((time.time() - tm1))
        #print(np.mean(np.array(time_buff)))
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        # dif_time = "Step time %s" % (time.time() - tm1)
        print(episode_stat)

        # Guardar estadísticas de cada episode
        # with open(train_stat_file, "a") as outfile:
        #     outfile.write(episode_stat + "\n")
        for actor_world in env.actor_list:
            actor_world.destroy()

    # actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "last_actor.h5", overwrite=True)
    # critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "last_critic.h5", overwrite=True)
    actor.model.save(os.path.join(settings.AGENT_PATH, f"{settings.TRAIN_MODE}_last_actor.model"), overwrite=True)
    critic.model.save(os.path.join(settings.AGENT_PATH, f"{settings.TRAIN_MODE}_last_critic.model"), overwrite=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
    args = parser.parse_args()
    play(args.train)
