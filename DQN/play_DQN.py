import os

from collections import deque
import numpy as np
import time
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
    model = keras.models.load_model(settings.MODEL_PATH)
    print(f"Model selected for evaluation: {settings.MODEL_PATH}")
    # Instantiate environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps

    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        aux = np.ones(settings.state_dim, )
        model.predict(np.array(aux).reshape(-1, *aux.shape))
    else:
        aux = np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS))
        model.predict(np.array(aux).reshape(-1, *aux.shape) / 255)[0]

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
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            _, state_train = env.reset()
        else:
            state_train, _ = env.reset()
            data_RNN = np.array([env.trackpos_rw, env.angle_rw])
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()


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

            # For FPS counter
            step_start = time.time()

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
                    settings.WORKING_MODE_OPTIONS[1] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                qs = model.predict(np.array(state_train).reshape(-1, *state_train.shape))
            else:
                qs = model.predict(np.array(state_train).reshape(-1, *state_train.shape) / 255)[0]
    
            action = np.argmax(qs)
            print(settings.ACTIONS_NAMES[action])

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                [_, new_state_train], reward, done, _ = env.step(action)
            else:
                [new_state_train, _], reward, done, _ = env.step(action)

                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()

            # Set current step for next loop iteration
            state_train = new_state_train
            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            
        if settings.SAVE_DATA == 1:
            tray_dir = 'Trayectorias/trayectoria_' + str(settings.WORKING_MODE)
            os.makedirs(tray_dir, exist_ok=True)
            np.savetxt(
                tray_dir + '/' + str(settings.TRAIN_MODE) + '_trayectoria_' + str(episode) + '.txt', 
                env.position_array, delimiter=';'
            )
            env.position_array = []
        
        # # === RESTORE RANDOM_TURN FOR NEXT EPISODE ===
        # if settings.TRAIN_MODE_OPTIONS[7] == "RANDOM_TURN":
        #     settings.TRAIN_MODE = "RANDOM_TURN"
        
        # Destroy an actor at end of episode
        episode += 1
        for actor in env.actor_list:
            actor.destroy()


