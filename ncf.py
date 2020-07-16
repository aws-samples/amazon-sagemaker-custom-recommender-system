"""

 Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 SPDX-License-Identifier: MIT-0
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this
 software and associated documentation files (the "Software"), to deal in the Software
 without restriction, including without limitation the rights to use, copy, modify,
 merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import tensorflow as tf
import argparse
import os
import numpy as np
import json


# for data processing
def _load_training_data(base_dir):
    """ load training data """
    df_train = np.load(os.path.join(base_dir, 'train.npy'))
    user_train, item_train, y_train = np.split(np.transpose(df_train).flatten(), 3)
    return user_train, item_train, y_train


def batch_generator(x, y, batch_size, n_batch, shuffle, user_dim, item_dim):
    """ batch generator to supply data for training and testing """

    user_df, item_df = x

    counter = 0
    training_index = np.arange(user_df.shape[0])

    if shuffle:
        np.random.shuffle(training_index)

    while True:
        batch_index = training_index[batch_size*counter:batch_size*(counter+1)]
        user_batch = tf.one_hot(user_df[batch_index], depth=user_dim)
        item_batch = tf.one_hot(item_df[batch_index], depth=item_dim)
        y_batch = y[batch_index]
        counter += 1
        yield [user_batch, item_batch], y_batch

        if counter == n_batch:
            if shuffle:
                np.random.shuffle(training_index)
            counter = 0


# network
def _get_user_embedding_layers(inputs, emb_dim):
    """ create user embeddings """
    user_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    user_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    return user_gmf_emb, user_mlp_emb


def _get_item_embedding_layers(inputs, emb_dim):
    """ create item embeddings """
    item_gmf_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    item_mlp_emb = tf.keras.layers.Dense(emb_dim, activation='relu')(inputs)

    return item_gmf_emb, item_mlp_emb


def _gmf(user_emb, item_emb):
    """ general matrix factorization branch """
    gmf_mat = tf.keras.layers.Multiply()([user_emb, item_emb])

    return gmf_mat


def _mlp(user_emb, item_emb, dropout_rate):
    """ multi-layer perceptron branch """
    def add_layer(dim, input_layer, dropout_rate):
        hidden_layer = tf.keras.layers.Dense(dim, activation='relu')(input_layer)

        if dropout_rate:
            dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
            return dropout_layer

        return hidden_layer

    concat_layer = tf.keras.layers.Concatenate()([user_emb, item_emb])

    dropout_l1 = tf.keras.layers.Dropout(dropout_rate)(concat_layer)

    dense_layer_1 = add_layer(64, dropout_l1, dropout_rate)

    dense_layer_2 = add_layer(32, dense_layer_1, dropout_rate)

    dense_layer_3 = add_layer(16, dense_layer_2, None)

    dense_layer_4 = add_layer(8, dense_layer_3, None)

    return dense_layer_4


def _neuCF(gmf, mlp, dropout_rate):
    concat_layer = tf.keras.layers.Concatenate()([gmf, mlp])

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

    return output_layer


def build_graph(user_dim, item_dim, dropout_rate=0.25):
    """ neural collaborative filtering model """

    user_input = tf.keras.Input(shape=(user_dim))
    item_input = tf.keras.Input(shape=(item_dim))

    # create embedding layers
    user_gmf_emb, user_mlp_emb = _get_user_embedding_layers(user_input, 32)
    item_gmf_emb, item_mlp_emb = _get_item_embedding_layers(item_input, 32)

    # general matrix factorization
    gmf = _gmf(user_gmf_emb, item_gmf_emb)

    # multi layer perceptron
    mlp = _mlp(user_mlp_emb, item_mlp_emb, dropout_rate)

    # output
    output = _neuCF(gmf, mlp, dropout_rate)

    # create the model
    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)

    return model


def model(x_train, y_train, n_user, n_item, num_epoch, batch_size):

    num_batch = np.ceil(x_train[0].shape[0]/batch_size)

    # build graph
    model = build_graph(n_user, n_item)

    # compile and train
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit_generator(
        generator=batch_generator(
            x=x_train, y=y_train,
            batch_size=batch_size, n_batch=num_batch,
            shuffle=True, user_dim=n_user, item_dim=n_item),
        epochs=num_epoch,
        steps_per_epoch=num_batch,
        verbose=2
    )

    return model


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_user', type=int)
    parser.add_argument('--n_item', type=int)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    # load data
    user_train, item_train, train_labels = _load_training_data(args.train)

    # build model
    ncf_model = model(
        x_train=[user_train, item_train],
        y_train=train_labels,
        n_user=args.n_user,
        n_item=args.n_item,
        num_epoch=args.epochs,
        batch_size=args.batch_size
    )

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        ncf_model.save(os.path.join(args.sm_model_dir, '000000001'), 'neural_collaborative_filtering.h5')
