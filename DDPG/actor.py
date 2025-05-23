import os

import keras.backend as keras_backend
import tensorflow as tf
from keras.initializers import RandomUniform # normal
from keras.layers import Dense, Input
from keras.models import Model
from carla_config import hidden_units, image_network


class ActorNetwork:
    def __init__(self, tf_session, state_size, action_size=2, tau=0.001, lr=0.0001):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr

        keras_backend.set_session(tf_session)

        self.model, self.model_states = self.generate_model()
        model_weights = self.model.trainable_weights

        self.target_model, _ = self.generate_model()

        # Generate tensors to hold the gradients for Policy Gradient update
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size])
        # Default implementation worked with action_size=1 and then overwrote it to action_size=1 in the training file. This setup implied it likely truncates to apply the gradient to only the first action dimension, e.g. only steering. You're only training the policy to optimize 1 of the 2 outputs.
        self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        self.gradients = zip(self.parameter_gradients, model_weights)

        # Optimizer
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        self.tf_session.run(tf.global_variables_initializer())

    def train(self, states, action_gradients):
        self.tf_session.run(
            self.optimize,
            feed_dict={
                self.model_states: states,
                self.action_gradients: action_gradients,
            },
        )

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        updated_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(updated_weights)
    
    def generate_model(self):
        input_layer = Input(shape=[self.state_size])
        h0 = Dense(hidden_units[0], activation="relu")(input_layer)
        h1 = Dense(hidden_units[1], activation="relu")(h0)
        output_layer = Dense(2, activation="tanh")(h1)
        model = Model(input=input_layer, output=output_layer)
        tf.keras.utils.plot_model(model,
                                  to_file=image_network + 'actor_model_WP_Carla.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')



        return model, input_layer

    ################
    # DDPG Paper
    ###############
    # def generate_model(self):
    #     input_layer = Input(shape=[self.state_size])
    #     # Hidden Layer 0 - 400 units, ReLU, Glorot init
    #     h0 = Dense(400, activation="relu")(input_layer)
    #     # Hidden Layer 1 - 300 units, ReLU, Glorot init
    #     h1 = Dense(300, activation="relu")(h0)
    #     # Output Layer - tanh, custom init [-3e-3, 3e-3]
    #     init = RandomUniform(minval=-3e-3, maxval=3e-3)
    #     # The number of output neurons in the actor must match the dimensionality of the continuous action space. It is 2 in this case: throttle and steering
    #     output_layer = Dense(self.action_size, activation="tanh", kernel_initializer=init)(h1)
    #     model = Model(input=input_layer, output=output_layer)
        
    #     os.makedirs('NETWORKS', exist_ok=True)
    #     tf.keras.utils.plot_model(model,
    #                               to_file=image_network + 'actor_model_WP_Carla.png',
    #                               show_shapes=True,
    #                               show_layer_names=True, rankdir='TB')



    #     return model, input_layer
