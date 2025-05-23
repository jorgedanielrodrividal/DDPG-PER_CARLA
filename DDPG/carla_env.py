import glob
import os
import sys

sys.path.append('/home/jorge-daniel/Desktop/carla/PythonAPI/carla')
sys.path.append('/home/jorge-daniel/Desktop/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg')

sys.path.append('/home/jorge-daniel/Desktop/carla/PythonAPI/carla/agents')

try:
    sys.path.append(glob.glob('../../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import random
import carla
from carla import ColorConverter
import time as time
import math
import numpy as np
import cv2
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import carla_config as settings

import tensorflow as tf
import keras.backend as keras_backend


red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


reward_mode = 2  #0: velocity reward | 1: velocity angle reward = 1 | 2: angle as a function of velocity reward

class CarEnv:
    im_width = settings.IM_WIDTH_VISUALIZATION
    im_height = settings.IM_HEIGHT_VISUALIZATION
    front_camera = None
    bev_camera = None
    angle_rw = 0
    trackpos_rw = 0
    cmd_vel = 0
    summary = {'Target': 0, 'Steps': 0}
    distance_acum = []
    #
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    tf_session2 = tf.Session(config=config2)

    keras_backend.set_session(tf_session2)

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.prev_d2goal = 10000
        self.Target = 0
        self.numero_tramo = 0
        self.error_lateral = []
        self.position_array = []
        self.prev_next = 0
        self.waypoints_txt = []
        # -------  Soft lane-invasion bookkeeping  -------
        # self._lane_violation_window = 0          # consecutive "bad" ticks
        # self._lane_violation_limit  = 3          # how many consecutive bad ticks we tolerate

        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[1]:
            self.pos_a = carla.Transform(carla.Location(x=92.109985, y=105.661537, z=1.320625),
                                         carla.Rotation(pitch=0.000000, yaw=-90.000298, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=283.021973, y=199.059723, z=1.320625),
                                         carla.Rotation(pitch=0.000000, yaw=-0.000092, roll=0.000000))


        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[2]:
            self.pos_a = carla.Transform(carla.Location(x=208.669876, y=195.149597, z=1.000000),
                                    carla.Rotation(pitch=360.000000, yaw=180.004654, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=88.415741, y=300.859680, z=1.000000),
                                    carla.Rotation(pitch=0.000000, yaw=89.991280, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[3]:
            self.pos_a = carla.Transform(carla.Location(x=208.669876, y=195.149597, z=1.000000),
                                    carla.Rotation(pitch=360.000000, yaw=180.004654, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=92.385292, y=100.597343, z=1.000000),
                                    carla.Rotation(pitch=360.000000, yaw=269.991272, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[4]:
            self.pos_a = carla.Transform(carla.Location(x=196.748154, y=55.487041, z=1.000000),
                                    carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                            carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))

        elif settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[5]:
            self.pos_a = carla.Transform(carla.Location(x=250.351135, y=59.474419, z=1.000000),
                                         carla.Rotation(pitch=0.000000, yaw=-0.006982, roll=0.000000))
            self.pos_b = carla.Transform(carla.Location(x=379.485901, y=2.017289, z=1.000000),
                                         carla.Rotation(pitch=360.000000, yaw=0.030457, roll=0.000000))
        else:
            self.pos_a = 0
            self.pos_b = 0

        self.ind = 1

        src = np.float32([[0, settings.IM_HEIGHT_VISUALIZATION], [1200, settings.IM_HEIGHT_VISUALIZATION], [0, 0], [settings.IM_WIDTH_VISUALIZATION, 0]])
        dst = np.float32([[569, settings.IM_HEIGHT_VISUALIZATION], [711, settings.IM_HEIGHT_VISUALIZATION], [0, 0], [settings.IM_WIDTH_VISUALIZATION, 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)



    def reset(self):
        self.tm = time.time()
        self.dif_tm = 0
        global acum
        global x_prev
        global y_prev
        acum = 0
        self.collision_hist = []
        self.actor_list = []
        self.crossline_hist = []
        self.coeficientes = np.zeros((51-1, 8))
        self.pos_array_wp = 0
        self.waypoints_current_plan = []
        # self.route_lane_markings = []  # New: Store lane marking types here
        self.dif_angle_routes = 0
        self.d2goal = 1
        self.map = self.world.get_map()
        self.dao = GlobalRoutePlannerDAO(self.map, 1.0)
        self.grp = GlobalRoutePlanner(self.dao, 2.0)


        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[6]:
            if self.ind == 0:
                self.pos_a = carla.Transform(carla.Location(x=196.748154, y=55.487041, z=1.000000),
                                             carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
                self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))
                self.ind = 1

            elif self.ind == 1:
                self.pos_a = carla.Transform(carla.Location(x=300.351135, y=59.474419, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-0.006982, roll=0.000000))
                self.pos_b = carla.Transform(carla.Location(x=379.485901, y=2.017289, z=1.000000),
                                             carla.Rotation(pitch=360.000000, yaw=0.030457, roll=0.000000))
                self.ind = 2

            elif self.ind == 2:
                self.pos_a = carla.Transform(carla.Location(x=196.748154, y=55.487041, z=1.000000),
                                             carla.Rotation(pitch=360.000000, yaw=179.993011, roll=0.000000))
                self.pos_b = carla.Transform(carla.Location(x=109.849731, y=-2.049278, z=1.000000),
                                             carla.Rotation(pitch=0.000000, yaw=-179.993881, roll=0.000000))
                self.ind = 1


        if settings.TRAIN_MODE == settings.TRAIN_MODE_OPTIONS[0]:
            spawn_points = self.map.get_spawn_points()
            self.waypoints_current_plan = []
            while self.d2goal < 2000 and self.dif_angle_routes == 0:
                self.pos_a = random.choice(spawn_points)
                self.pos_b = random.choice(spawn_points)
                angles_dif = abs(abs(self.pos_a.rotation.yaw) - abs(self.pos_b.rotation.yaw))
                if angles_dif > 80 and angles_dif < 100:
                    self.dif_angle_routes = 1

                a = self.pos_a.location
                b = self.pos_b.location
                self.current_plan = self.grp.trace_route(a, b)
                self.d2goal = self.total_distance(self.current_plan)

            self.d2goal = self.total_distance(self.current_plan)

            self.transform = self.pos_a

        else:
            self.current_plan = self.grp.trace_route(self.pos_a.location, self.pos_b.location)
            self.sssd2goal = self.total_distance(self.current_plan)

            self.transform = self.pos_a

        # print(self.transform)

        for i in range(len(self.current_plan)):
            w1 = self.current_plan[i][0]
            self.waypoints_current_plan.append([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z,
                 w1.transform.rotation.pitch, w1.transform.rotation.yaw, w1.transform.rotation.roll])
            # NEW: Capture lane marking types
            # left_marking = w1.left_lane_marking.type if w1.left_lane_marking else "None"
            # right_marking = w1.right_lane_marking.type if w1.right_lane_marking else "None"

            # self.route_lane_markings.append({
            #     "index": i,
            #     "left_marking": str(left_marking),
            #     "right_marking": str(right_marking),
            #     "lane_id": w1.lane_id
            # })

        # for entry in self.route_lane_markings:
        #         print(f"[Waypoint {entry['index']}] Lane ID: {entry['lane_id']} | Left: {entry['left_marking']} | Right: {entry['right_marking']}")

        self.waypoints_current_plan.append([0, 0, 0, 0, 0, 0])
        self.Target = w1.transform.location
        self.draw_path(self.world, self.current_plan, tl=settings.LINE_TIME)
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.actor_list.append(self.vehicle)
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")

        transform = carla.Transform(carla.Location(x=settings.CAM_X, z=settings.CAM_Z),
                                    carla.Rotation(pitch=settings.CAM_PITCH, yaw=settings.CAM_YAW, roll=settings.CAM_ROLL))

        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        x_linesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.x_linesensor = self.world.spawn_actor(x_linesensor, transform, attach_to=self.vehicle)
        gnss_sensor = self.blueprint_library.find("sensor.other.gnss")
        self.gnss_sensor = self.world.spawn_actor(gnss_sensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.x_linesensor.listen(lambda event2: self.crossline_data(event2))
        self.gnss_sensor.listen(lambda event3: self.gnss_data(event3))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        location_reset = self.vehicle.get_transform()
        x_prev = location_reset.location.x
        y_prev = location_reset.location.y

        #self.state_train = self.Calcular_estado(self.front_camera)

        # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
        #     self.state_train = self.Calcular_estado(self.front_camera)
        #     return self.front_camera, self.state_train
        # else:
        im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))

        wp_state, _ = self.transform2local(im)
        return im, wp_state


    def total_distance(self, current_plan):
        sum = 0
        for i in range(len(current_plan) - 1):
            sum = sum + self.distance_wp(current_plan[i + 1][0], current_plan[i][0])
        return sum

    def distance_wp(self, target, current):
        dx = target.transform.location.x - current.transform.location.x
        dy = target.transform.location.y - current.transform.location.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_target(self, target, current):
        dx = target.x - current.x
        dy = target.y - current.y
        return math.sqrt(dx * dx + dy * dy)

    def draw_path(self, world, current_plan, tl):
        for i in range(len(current_plan) - 1):
            w1 = current_plan[i][0]
            w2 = current_plan[i + 1][0]
            self.world.debug.draw_line(w1.transform.location, w2.transform.location, thickness=settings.LINE_WIDHT,
                                       color=green, life_time=tl)

    def draw_waypoint_info(self, world, w, lt=(settings.SECONDS_PER_EPISODE+ 5.0)):
        w_loc = w.transform.location
        world.debug.draw_point(w_loc, 0.5, red, lt)


    def crossline_data(self):
        self.crossline_hist.append(1)
    
    # -------------------------------------------------
    #  Softened Lane-Invasion callback
    # -------------------------------------------------
    
    # def crossline_data(self, event):
    #     """
    #     Called automatically by CARLA every time the LaneInvasion sensor
    #     detects that any wheel touches a lane marking.

    #     We *ignore* harmless markings (dashed / NONE) and require
    #     several consecutive 'solid' violations before terminating.
    #     """
    #     # Lane-marking types we consider "harmless"
    #     SAFE_MARKINGS = {
    #         carla.LaneMarkingType.NONE,
    #         carla.LaneMarkingType.Broken,   # dashed / broken white
    #         carla.LaneMarkingType.Other,
    #         carla.LaneMarkingType.BottsDots
    #     }

    #     # Which marking types were crossed in this event?
    #     crossed = {m.type for m in event.crossed_lane_markings}

    #     # If *all* crossed markings are safe --> reset the window
    #     if crossed.issubset(SAFE_MARKINGS):
    #         self._lane_violation_window = 0
    #         return                                  # no penalty

    #     # Otherwise we hit a SOLID / DOUBLE line  → count one “bad” tick
    #     self._lane_violation_window += 1

    #     # Only flag an episode-ending violation if it happens
    #     # `self._lanpossible.e_violation_limit` times *in a row*
    #     if self._lane_violation_window >= self._lane_violation_limit:
    #         self.crossline_hist.append(1)           # ← same flag used elsewhere
    #         # Reset so a *new* sequence is needed after reset()
    #         self._lane_violation_window = 0


    def gnss_data(self, event3):
        global latitude
        global longitude

        latitude = event3.latitude
        longitude = event3.longitude

    def collision_data(self, event):
        self.collision_hist.append(1)

    def process_img(self, image):
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] \
                or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
                # or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            i = np.array(image.raw_data)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            if settings.BEV_PRE_CNN == 1:
                self.front_camera = cv2.warpPerspective(i3, self.M,
                                                        (settings.IM_WIDTH_VISUALIZATION,
                                                         settings.IM_HEIGHT_VISUALIZATION))
            else:
                self.front_camera = i3
        else:
            if settings.THRESHOLD == 0:
                if settings.IM_TYPE == 1:
                    image.convert(ColorConverter.CityScapesPalette)
                i = np.array(image.raw_data)
                i2 = i.reshape((self.im_height, self.im_width, 4))
                i3 = i2[:, :, :3]
                gray = cv2.cvtColor(i3, cv2.COLOR_BGR2GRAY)

                if settings.IM_LAYERS == 1:
                    self.front_camera = gray
                elif settings.IM_LAYERS == 3:
                    if settings.BEV_PRE_CNN == 1:
                        self.front_camera = cv2.warpPerspective(i3, self.M,
                                                              (settings.IM_WIDTH_VISUALIZATION,
                                                               settings.IM_HEIGHT_VISUALIZATION))
                    else:
                        self.front_camera = i3

            else:
                i = np.array(image.raw_data)
                i2 = i.reshape((self.im_height, self.im_width, 4))
                i3 = i2[:, :, :3]
                kernel = np.ones((5, 5), np.uint8)
                # kernel = np.ones((6, 6), np.uint8)
                ang_deg = 0.0
                mask = cv2.inRange(i3, (0, 200, 0), (10, 256, 10))
                gray = cv2.dilate(mask, kernel, iterations=2)
                gray = cv2.erode(gray, kernel, iterations=2)
                self.front_camera = gray

        # if settings.BEV_PRE_CNN == 1:
        #     self.bev_camera = cv2.warpPerspective(i3, self.M, (settings.IM_WIDTH_VISUALIZATION, settings.IM_HEIGHT_VISUALIZATION))


    def step(self, action):
        global x_prev
        global y_prev
        global acum
        global acum_prev
        global d_i_prev

        #Action is applied like steerin while throttle is cte
        throttle2apply = 0.125*float(action[1]) + 0.475
        # throttle2apply = 0.15*float(action[1]) + 0.35
        steering2apply = float(action[0]/2)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle2apply, steer=steering2apply))

        location_rv = self.vehicle.get_transform()

        d_i = math.sqrt((x_prev - location_rv.location.x) ** 2 + (y_prev - location_rv.location.y) ** 2)

        acum += d_i
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        self.position_array.append([x_prev, y_prev, location_rv.location.z, location_rv.rotation.pitch, location_rv.rotation.yaw, location_rv.rotation.roll])
        x_prev = location_rv.location.x
        y_prev = location_rv.location.y
        d_i_prev = d_i

        reward, done, d2target = self.get_reward()

        if settings.SHOW_CAM == 1:
            cv2.namedWindow('Real', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Real', self.front_camera)
            cv2.waitKey(1)

        # # SALIDA UTILIZADA PARA EL PROGRAMA DE LOS WAYPOINTS OBTENIDOS POR TRATAMIENTO DE IMAGEN
        # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
        #     if done == True:
        #         self.distance_acum.append(acum)
        #     state = self.Calcular_estado(self.front_camera)
        #     return [self.front_camera, state], reward, done, None

        # SALIDA UTILIZADA EL RESTO DE PROGRAMAS
        # else:
        im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))
        next, exit_flag = self.transform2local(im)

            # im_crop = self.front_camera[412:, 354:788]
            # cv2.namedWindow('Crop', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Crop', im_crop)
            # cv2.waitKey(1)
            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            #     if np.sum(im_crop[:, :, 1] == 234) < 50 and np.sum(im_crop[:, :, 1] == 233) < 50 and np.sum(im_crop[:, :, 1] == 235) < 50:  # si no hay mil puntos blancos decimos que nos hemos salido
            #         print('Salida por pérdida de carril en crop image')
            #         done = True

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[2]:
            if np.count_nonzero(self.front_camera) < 1000:  # si no hay mil puntos blancos decimos que nos hemos salido
                exit_flag = 1

        # If the vehicle deviates from the waypoint route detects no green (route) and exits
        # ###########
        # # VERSION 3
        # ###########
        # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[4] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] \
        #         or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0]: # or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[3]
        #     if np.sum(self.front_camera[:, :, 1] >= 245) <= 100: # and np.sum(self.front_camera[:, :, 1] == 220) <= 260:  # si no hay mil puntos blancos decimos que nos hemos salido
        #         g = self.front_camera[:, :, 1]
        #         print(f"[DEBUG] Green channel - min: {g.min()}, max: {g.max()}, mean: {g.mean():.2f}")
        #         # print("\nThe road or lane markings are no longer clearly visible")
        #         # print(f"There are {np.sum(self.front_camera[:, :, 1] >= 245)} pixels with white values\n")
        #         exit_flag = 1
        if exit_flag == 1:
            print('Waypoints lost, distance to goal: ', d2target)
            done = True
            reward = -200


        if settings.SHOW_CAM_RESIZE == 1:
            cv2.namedWindow('Resize', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Resize', im)
            cv2.waitKey(1)
        if done == True:
            self.distance_acum.append(acum)

        return [im, next], reward, done, None

    def get_image(self):
        im = cv2.resize(self.front_camera, (settings.IM_WIDTH_CNN, settings.IM_HEIGHT_CNN))
        return im


    def get_reward(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        if kmh > 120:
            kmh = 120

        location = self.vehicle.get_location()
        progress = np.cos(self.angle_rw) - abs(np.sin(self.angle_rw)) - abs(self.trackpos_rw)
        salida = 0 

        d2target = self.distance_target(self.Target, location)

        # Collision exit
        if len(self.collision_hist) != 0: 
            done = True
            salida = 1
            reward = -200
            print('There has been a collision, distance to goal: ', d2target)
            self.summary['Steps'] += 1
        
        # if len(self.crossline_hist) != 0:
        #     # done = True
        #     # salida = 1
        #     reward = -100
        #     print('There has been a lane deviation, distance to goal: ', d2target)
        #     # self.summary['Steps'] += 1

        # Hard lane deviation exit condition
        # if len(self.crossline_hist) != 0:
        #     done = True
        #     salida = 1
        #     reward = -200
        #     print('There has been a lane deviation, distance to goal: ', d2target)
        #     self.summary['Steps'] += 1

        if salida == 0:  

            # Given reward depending on circumstances
            if reward_mode == 0:
                if kmh < 10:
                    done = False
                    reward = -1
                else:
                    done = False
                    reward = 1
            elif reward_mode == 1:
                reward = progress
                done = False
            else:
                reward = (kmh) * progress
                done = False

            # Setup reward and exit if the condition is reached
            if self.distance_target(self.Target, location) < 15:
                done = True
                reward = 100
                self.summary['Steps'] += 1
                self.summary['Target'] += 1
                print('Reached goal')

            # SSetup timeout and corresponding reward
            if self.episode_start + settings.SECONDS_PER_EPISODE < time.time():
                print('Timeout, distance to goal: ', d2target)
                done = True
                self.summary['Steps'] += 1
                if acum <= 50:
                    reward = -200
                elif (acum > 50) and (acum < 160):
                    reward = -100
                else:
                    reward = 100

        self.cmd_vel = kmh / 120  

        return reward, done, d2target


    # def Calcular_estado(self, img2):
    #     global center_old
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.inRange(img2, (0, 200, 0), (10, 256, 10))
    #     gray = cv2.dilate(mask, kernel, iterations=2)
    #     gray = cv2.erode(gray, kernel, iterations=2)

    #     edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    #     height = gray.shape[0]
    #     width = gray.shape[1]
    #     waypoint = np.zeros((15, ))
    #     waypoint_edges = np.zeros((15, 2))
    #     state = np.zeros((settings.state_dim, ))

    #     #CALCULO DEL PUNTO DE FUGA
    #     for i in range(0, 15):
    #         dato_y = int(height - 1 - 25 * i)

    #         for j in range(0, width):
    #             if gray[dato_y, j] == 255:
    #                 waypoint_edges[i][
    #                     0] = j  # me quedo con la coordenada x empezando por la izqueirda de la imagen, la y viene dada por el indice
    #                 break
    #         for j in range(0, width):
    #             if gray[dato_y, width - 1 - j] == 255:
    #                 waypoint_edges[i][
    #                     1] = width - 1 - j  # me quedo con la coordenada x empezando por la izqueirda de la imagen, la y viene dada por el indice
    #                 break
    #         waypoint[i] = int((waypoint_edges[i][0] + waypoint_edges[i][1]) / 2)
    #         if i < 6:
    #             if (waypoint_edges[i][0] == 0) and (waypoint_edges[i][1] < (width - 1)):
    #                 waypoint[i] = waypoint_edges[i][1] - (280 - 20 * i)
    #                 if waypoint[i] < 0:
    #                     waypoint[i] = 0
    #             elif (waypoint_edges[i][0] > 0) and (waypoint_edges[i][1] >= (width - 1)):
    #                 waypoint[i] = waypoint_edges[i][0] + (280 - 20 * i)
    #                 if waypoint[i] > (width - 1):
    #                     waypoint[i] = width - 1

    #         if i == 1:
    #             waypoint[0] = waypoint[1]
    #         # waypoint[i] = int((waypoint_edges[i][0] + waypoint_edges[i][1]) / 2)

    #     # PINTAR LOS PUNTOS DE LA CARRETERA
    #     for i in range(0, 15):
    #         dato_y = int(height - 1 - 25 * i)
    #         waypointcenter2 = (int(waypoint[i]), int(dato_y))
    #         cv2.circle(gray, waypointcenter2, 2, (0, 0, 0), 2)
    #         waypoint[i] = (waypoint[i] - width / 2) / (width / 2)
    #         if i > 1 and waypoint[i] == -1:
    #             if waypoint[i - 1] > 0.3:
    #                 waypoint[i] = 1
    #             elif waypoint[i - 1] < -0.3:
    #                 waypoint[i] = -1
    #             else:
    #                 waypoint[i] = 0

    #         state[0:(settings.state_dim-2)] = waypoint
    #         state[settings.state_dim-2] = self.angle_rw/math.pi
    #         state[settings.state_dim - 1] = self.cmd_vel
    #         self.trackpos_rw = waypoint[0]


    #     #CALCULAR EL ÁNGULO DE LA CARRETERA
    #     x_diff = waypoint[7] - waypoint[5]
    #     y_diff = (7*25 - 5*25)/(width/2)
    #     self.angle_rw = np.arctan2(x_diff, y_diff)
    #     print(self.angle_rw*180/np.pi)
    #     if settings.SHOW_WAYPOINTS == 1:
    #         cv2.namedWindow('Punto de fuga', cv2.WINDOW_AUTOSIZE)
    #         cv2.imshow('Punto de fuga', gray)
    #         cv2.waitKey(1)
    #     return state


    def transform2local(self, im):
        state = np.zeros((settings.dimension_vector_estado,))
        actual_pos = self.vehicle.get_transform()
        yaw_c = actual_pos.rotation.yaw*math.pi/180 - math.pi/2
        # print('yaw_C: ',yaw_c)
        Xc = actual_pos.location.x
        Yc = actual_pos.location.y
        Zc = actual_pos.location.z
        # print('Pc :', Xc, ' ', Yc, ' ', Zc)
        self.waypoints_current_plan[-1] = [actual_pos.location.x, actual_pos.location.y, actual_pos.location.z,
                                           actual_pos.rotation.pitch, actual_pos.rotation.yaw,
                                           actual_pos.rotation.roll]

        aux_waypoints = np.array(self.waypoints_current_plan)
        self.waypoints_txt = aux_waypoints

        aux_waypoints = aux_waypoints[0:-1, 0:4]
        aux_waypoints[:, 3] = 1
        #aux_waypoints[:, 0] = -aux_waypoints[:, 0]

        M = np.array(([np.cos(yaw_c), -np.sin(yaw_c), 0, Xc],
                      [np.sin(yaw_c), np.cos(yaw_c), 0, Yc],
                      [0, 0, 1, Zc],
                      [0, 0, 0, 1]))
        # print('WP1: ', aux_waypoints[0, :])
        # print('WP-1: ', aux_waypoints[-1, :])

        M_inv = np.linalg.inv(M)
        P_locales = np.zeros((len(aux_waypoints), 4))
        #plt.figure(1)
        for i in range(len(aux_waypoints)):
            P_locales[i] = np.dot(M_inv, aux_waypoints[i, :])
        P_locales[:, 0] = -P_locales[:, 0]

        P_locales_aux = P_locales[self.pos_array_wp:(self.pos_array_wp+30)]

        #Pintar el número de waypoints que se han pasado
        wp_out = np.where(P_locales_aux[:, 1] < 0)
        n_wp_out = len(wp_out[0])


        nextWP = P_locales_aux[n_wp_out:(n_wp_out+15)]


        self.pos_array_wp += n_wp_out
        self.dif_tm += (time.time() - self.tm)
        # print(self.dif_tm)
        if self.dif_tm > 5:
            print('pasa')
            self.draw_path(self.world, self.current_plan[self.pos_array_wp:(self.pos_array_wp+100)], tl=settings.LINE_TIME+1)
            self.dif_tm = 0

        self.tm = time.time()

        # next15_aux = nextWP[:, 0]
        # next15_aux_y = nextWP[:, 1]

        nextt15_aux = nextWP[:, 0:2]
        next15 = np.zeros((15, 2))
        tam_wp = len(nextt15_aux)

        if tam_wp < 15:
            if tam_wp == 0:
                exit_flag = 1
                next15 = self.prev_next
            else:
                exit_flag = 0
                next15[0:tam_wp] = nextt15_aux
                for k in range(15 - tam_wp):
                    next15[-1-k] = nextt15_aux[tam_wp-1]
        else:
            exit_flag = 0
            next15 = nextt15_aux

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            if settings.SHOW_WAYPOINTS == 1:
                img_negra = np.zeros((512, 512, 3), np.uint8)
                for i in range(len(next15)):

                    pto = (int(next15[i][0]*20+512/2), int(512-next15[i][1]*30))
                    cv2.circle(img_negra, pto, 3, (255, 0, 0), 2)

                cv2.namedWindow('Waypoints', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Waypoints', img_negra)
                cv2.waitKey(1)

        self.prev_next = next15


        x_diff = next15[4][1] - next15[2][1]
        y_diff = -(next15[4][0] - next15[2][0])

        self.angle_rw = np.arctan2(y_diff, x_diff)
        self.trackpos_rw = next15[0][0]


        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:

            waypoints_predicted = self.model_waypoints.predict(np.array(im).reshape(-1, settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, 3)/255, verbose=0)
            waypoints_predicted = waypoints_predicted.reshape(15, 2)

            if settings.SHOW_WAYPOINTS == 1:
                for i in range(len(waypoints_predicted)):
                    pto = (int(waypoints_predicted[i][0] * 20 + 512 / 2), int(512 - waypoints_predicted[i][1] * 30))
                    cv2.circle(img_negra, pto, 3, (0, 0, 255), 2)

                cv2.namedWindow('Waypoints', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Waypoints', img_negra)
                cv2.waitKey(1)

            if settings.WAYPOINTS == 'XY':
                state[0:(settings.dimension_vector_estado - 1)] = waypoints_predicted.flatten()
                state[settings.dimension_vector_estado - 1] = self.angle_rw/ np.pi
                return state, exit_flag

            elif settings.WAYPOINTS == 'X':
                state[0:(settings   .dimension_vector_estado - 1)] = waypoints_predicted[:, 0]/10
                state[settings.dimension_vector_estado - 1] = self.angle_rw / np.pi
                return state, exit_flag

        if settings.WAYPOINTS == 'XY':
            state[0:(settings.dimension_vector_estado - 1)] = next15.flatten()/10
            state[settings.dimension_vector_estado - 1] = self.angle_rw/ np.pi
            return state, exit_flag

        elif settings.WAYPOINTS == 'X':
            state[0:(settings.dimension_vector_estado - 1)] = next15[:, 0]/10
            state[settings.dimension_vector_estado - 1] = self.angle_rw/ np.pi

            return state, exit_flag
