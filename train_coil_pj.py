import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf
import argparse
from utils import *

tf1.compat.v1.enable_eager_execution()
class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()

        # assert tf1.executing_eagerly()
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the CoIL network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT 1: An example of this was given to you in Homework 1's Problem 1 in svm_tf.py. Now you will implement a multi-layer version.
        # HINT 2: You should use either of the following for weight initialization:
        #           - tf1.contrib.layers.xavier_initializer (this is what we tried)
        #           - tf.keras.initializers.GlorotUniform (supposedly equivalent to the previous one)
        #           - tf.keras.initializers.GlorotNormal
        #           - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal

        weights_initializer = tf1.contrib.layers.xavier_initializer()
        # weights_initializer = tf.keras.initializers.GlorotUniform


        self.w1 = tf1.compat.v1.get_variable(name="w1", shape=(in_size, 8), initializer=weights_initializer)
        self.b1 = tf1.compat.v1.get_variable(name="b1", shape=8, initializer=weights_initializer)

        self.w2 = tf1.compat.v1.get_variable(name="w2", shape=(8, 16), initializer=weights_initializer)
        self.b2 = tf1.compat.v1.get_variable(name="b2", shape=16, initializer=weights_initializer)

        self.w3 = tf1.compat.v1.get_variable(name="w3", shape=(16, out_size), initializer=weights_initializer)
        self.b3 = tf1.compat.v1.get_variable(name="b3", shape=out_size, initializer=weights_initializer)

        self.w4 = tf1.compat.v1.get_variable(name="w4", shape=(8, 16), initializer=weights_initializer)
        self.b4 = tf1.compat.v1.get_variable(name="b4", shape=16, initializer=weights_initializer)

        self.w5 = tf1.compat.v1.get_variable(name="w5", shape=(16, out_size), initializer=weights_initializer)
        self.b5 = tf1.compat.v1.get_variable(name="b5", shape=out_size, initializer=weights_initializer)

        self.w6 = tf1.compat.v1.get_variable(name="w6", shape=(8, 16), initializer=weights_initializer)
        self.b6 = tf1.compat.v1.get_variable(name="b6", shape=16, initializer=weights_initializer)

        self.w7 = tf1.compat.v1.get_variable(name="w7", shape=(16, out_size), initializer=weights_initializer)
        self.b7 = tf1.compat.v1.get_variable(name="b7", shape=out_size, initializer=weights_initializer)




        ########## Your code ends here ##########

    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.int8)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for (x,u) where:
        # - x is a (? x |O|) tensor that keeps a batch of observations
        # - u is a (? x 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # FYI: For the intersection scenario, u=0 means the goal is to turn left, u=1 straight, and u=2 right. 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found tf.math.equal and tf.cast useful. This is not necessarily a requirement though.

        bach_size = len(x)

        y_1 = tf.matmul(x, self.w1) - self.b1
        y_1 = tf.math.tanh(y_1)


        mask_0 = tf.math.equal(u, 0)

        mask_0 = tf.reshape(mask_0, [bach_size])
        mask_1 = tf.math.equal(u, 1)
        mask_1 = tf.reshape(mask_1, [bach_size])
        mask_2 = tf.math.equal(u, 2)
        mask_2 = tf.reshape(mask_2, [bach_size])

        y_2 = tf.boolean_mask(y_1, mask_0)
        y_3 = tf.boolean_mask(y_1, mask_1)
        y_4 = tf.boolean_mask(y_1, mask_2)

        y_5 = tf.matmul(y_2, self.w2) - self.b2
        y_5 = tf.math.sigmoid(y_5)
        y_6 = tf.matmul(y_5, self.w3) - self.b3

        y_7 = tf.matmul(y_3, self.w4) - self.b4
        y_7 = tf.math.sigmoid(y_7)
        y_8 = tf.matmul(y_7, self.w5) - self.b5

        y_9 = tf.matmul(y_4, self.w6) - self.b6
        y_9 = tf.math.sigmoid(y_9)
        y_10 = tf.matmul(y_9, self.w7) - self.b7

        # y_est = tf.concat([y_6, y_8, y_10], 0)

        indices_zero = tf.cast(mask_0, dtype=tf.float32)
        diag_zero = tf1.linalg.tensor_diag(indices_zero)
        final_zero = tf.transpose(tf.boolean_mask(diag_zero, mask_0))
        final_zero = tf.cast(final_zero, tf.float32)

        indices_one = tf.cast(mask_1, dtype=tf.float32)
        diag_one = tf1.linalg.tensor_diag(indices_one)
        final_one = tf.transpose(tf.boolean_mask(diag_one, mask_1))
        final_one = tf.cast(final_one, tf.float32)

        indices_two = tf.cast(mask_2, dtype=tf.float32)
        diag_two = tf1.linalg.tensor_diag(indices_two)
        final_two = tf.transpose(tf.boolean_mask(diag_two, mask_2))
        final_two = tf.cast(final_two, tf.float32)

        y_est = tf.matmul(final_zero, y_6) + tf.matmul(final_one, y_8) + tf.matmul(final_two, y_10)

        # y_est = tf.matmul(final_zero, y_6) + tf.matmul(final_one, y_8)
        return y_est

        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally

    loss = tf.norm(y - y_est)

    # loss = tf.norm(y_est - y)
    return loss

    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]

    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y, u):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model (note both x and u are inputs now)
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        # HINT: You did the exact same thing in Homework 1! It is just the networks weights and biases that are different.

        with tf.GradientTape() as tape:
            y_est = nn_model.call(x, u)
            current_loss = loss(y_est, y)
        varss = [nn_model.w1, nn_model.b1, nn_model.w2, nn_model.b2, nn_model.w3, nn_model.b3, nn_model.w4, nn_model.b4,
                 nn_model.w5, nn_model.b5, nn_model.w6, nn_model.b6, nn_model.w7, nn_model.b7]
        grads = tape.gradient(current_loss, varss)
        optimizer.apply_gradients(zip(grads, varss))

        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y, u in train_data:
            train_step(x, y, u)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'], data['u_train'])).shuffle(
        100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_CoIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    args.goal = 'all'

    maybe_makedirs("./policies")

    data = load_data(args)

    nn(data, args)
