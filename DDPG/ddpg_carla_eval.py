import os

from collections import deque
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import keras
from carla_env import CarEnv
import carla_config as settings

if __name__ == '__main__':
    # Memory fraction
    gpu_options = tf.GPUOptions(allow_growth=True)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    # Load the model
    actor_model = keras.models.load_model(settings.ACTOR_MODEL_PATH)
    print(f"Model selected for evaluation: Actor: {settings.ACTOR_MODEL_PATH}\n")
    # Create environment
    env = CarEnv()

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # The Critic network is never used during evaluation.

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]: #or \
                #settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        aux = np.ones(settings.state_dim, )
        actor_model.predict(np.array(aux).reshape(-1, *aux.shape))
    else:
        aux = np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS))
        aux = aux / 255
        aux = np.transpose(aux, (1, 0, 2))  # CNN setups require transposition
        actor_model.predict(np.array(aux).reshape(-1, *aux.shape))[0]

    # Loop over episodes
    episode = 0
    while True:

        print('Restarting episode')

        # Select random maneuver BEFORE reset if using RANDOM_TURN mode
        # if settings.TRAIN_MODE == "RANDOM_TURN":
        #     settings.TRAIN_MODE = random.choice(["STRAIGHT", "TURN_LEFT", "TURN_RIGHT"])
        #     print(f"[EVAL_MODE] Episode {episode}: {settings.TRAIN_MODE}")

        # Reset environment and get initial state

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]: #or \
                #settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            _, state_train = env.reset()
        else:
            state_train, _ = env.reset()
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            state_train = state_train / 255
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[3]:
                state_train = np.transpose(state_train, (1, 0, 2))

        env.collision_hist = []

        done = False
        if settings.SAVE_DATA == 1:
            wp_dir = 'Waypoints/waypoints_' + str(settings.WORKING_MODE)
            os.makedirs(wp_dir, exist_ok=True)
            np.savetxt(
                wp_dir + '/' + str(settings.TRAIN_MODE) + '_waypoints' + str(episode) + '.txt', 
                env.waypoints_txt, delimiter=';'
            )
        # Loop over steps
        while True:

            # Predict action using the actor model
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]: # or \
            # settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                action_pred = actor_model.predict(np.array(state_train).reshape(-1, *state_train.shape))[0]
            else:
                action_pred = actor_model.predict(np.array(state_train).reshape(-1, *state_train.shape))[0]

            print(f"Predicted action: {action_pred}")

            # Step environment
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]: #or \
            #settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                [_, new_state_train], reward, done, _ = env.step(action_pred)
            else:
                [new_state_train, _], reward, done, _ = env.step(action_pred)
                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()
                new_state_train = new_state_train / 255.0
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[3]:
                    new_state_train = np.transpose(new_state_train, (1, 0, 2))
            
            # Update state
            state_train = new_state_train 

            if done:
                break

        # Save trajectory if required
        if settings.SAVE_DATA == 1:
            traj_dir = 'Trayectorias_DDPG/trayectoria_' + str(settings.WORKING_MODE)
            os.makedirs(traj_dir, exist_ok=True)
            np.savetxt(
                os.path.join(traj_dir, f'{settings.TRAIN_MODE}_trayectoria_{episode}.txt'),
                env.position_array, delimiter=';'
            )
            env.position_array = []
        
        # # Restore RANDOM_TURN for next episode if needed
        # if settings.TRAIN_MODE_OPTIONS[7] == "RANDOM_TURN":
        #     settings.TRAIN_MODE = "RANDOM_TURN"
        
        # Destroy all actors spawned during the episode
        for actor_world in env.actor_list:
            if actor_world is not None:
                try:
                    actor_world.destroy()
                except Exception as e:
                    print(f"[WARNING] Failed to destroy actor: {e}")
        
        # Increment episode counter 
        episode += 1
        
