import os
import carla_config as settings
import time

print('### Reseting Carla Map ###')
# Explicit path to the CARLA egg file
carla_egg_path = '/home/jorge-daniel/Desktop/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg'
# Full path to the config script
config_path = settings.path2CARLA + 'PythonAPI/util/config.py'
# Launch config.py with explicit PYTHONPATH for CARLA
os.system(f'PYTHONPATH={carla_egg_path} python3 {config_path} -m {settings.CARLA_MAP}')
time.sleep(5)

#Clear Carla Environment
# print('### Reseting Carla Map ###')
# os.system('python3 ' + settings.path2CARLA + 'PythonAPI/util/config.py -m ' + str(settings.CARLA_MAP))
# time.sleep(5)

print('####### RUNNING DQN', settings.WORKING_MODE, ' IN ', settings.TRAIN_PLAY[settings.TRAIN_PLAY_MODE], ' MODE #######')

if settings.TRAIN_PLAY_MODE == 1:
    os.system('python3 ddpg_carla.py')
elif settings.TRAIN_PLAY_MODE == 0:
    os.system('python3 ddpg_carla_eval.py')

# print('####### RUNNING DDPG ', settings.WORKING_MODE, ' IN ', settings.TRAIN_PLAY[settings.TRAIN_PLAY_MODE], ' MODE #######')
# os.system('python3 DDPG/ddpg_carla.py --train ' + str(settings.TRAIN_PLAY_MODE))