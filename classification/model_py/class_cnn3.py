import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers
import tensorflow.keras.layers as klayer
import tensorflow.contrib.keras as K
from kgcn.default_model import DefaultModel

import model_py.model_utility as mu


class MultimodalNetwork(DefaultModel):
    def build_placeholders(self, info, config, batch_size, **kwargs):
        # input data types (placeholders) of this neural network
        return self.get_placeholders(info, config, batch_size,
                                     ['nodes', 'labels', 'mask_label', 'mask', 'dropout_rate',
                                      'enabled_node_nums', 'is_train', 'features',
                                      'sequences', 'sequences_len', 'embedded_layer'], **kwargs)

    def build_model(self, placeholders, info, config, batch_size, feed_embedded_layer=False, **kwargs):
        self.batch_size = batch_size
        self.input_dim = info.input_dim
        self.adj_channel_num = info.adj_channel_num
        self.sequence_symbol_num = info.sequence_symbol_num
        self.label_dim = info.label_dim
        self.embedding_dim = config["embedding_dim"]
        self.pos_weight = info.pos_weight
        self.feed_embedded_layer = feed_embedded_layer
        ## aliases
        adj_channel_num = info.adj_channel_num
        batch_size = self.batch_size
        # in_adjs = self.placeholders["adjs"]
        # features = self.placeholders["features"]
        sequences = self.placeholders["sequences"]
        sequences_len = self.placeholders["sequences_len"]
        # in_nodes = self.placeholders["nodes"]
        labels = self.placeholders["labels"]
        mask = self.placeholders["mask"]
        mask_label = self.placeholders["mask_label"]
        dropout_rate = self.placeholders["dropout_rate"]
        is_train = self.placeholders["is_train"]
        # enabled_node_nums = self.placeholders["enabled_node_nums"]
        embedded_layer = self.placeholders['embedded_layer']

        ## Sequence Part
        with tf.variable_scope("seq_nn") as scope_part:
            # Embedding
            self.embedding_layer = tf.keras.layers.Embedding(
                self.sequence_symbol_num, self.embedding_dim)(sequences)

            if self.feed_embedded_layer:
                layer = embedded_layer
            else:
                layer = self.embedding_layer
            print("sequence input layer:", layer.shape)
            # CNN + Pooling
            stride = 3
            layer = tf.keras.layers.Conv1D(
                200, stride, padding="same", activation='relu')(layer)
            layer = tf.keras.layers.MaxPooling1D(stride)(layer)

            stride = 2
            layer = tf.keras.layers.Conv1D(
                100, stride, padding="same", activation='relu')(layer)
            layer = tf.keras.layers.MaxPooling1D(stride)(layer)

            layer = tf.keras.layers.Conv1D(
                1, stride, padding="same", activation='tanh')(layer)
            layer = tf.squeeze(layer)

            if len(layer.shape) == 1:
                # When batch-size is 1, this shape doesn't have batch-size dimmension due to just previous 'tf.squeeze()'.
                layer = tf.expand_dims(layer, axis=0)
            seq_output_layer = layer
            seq_output_layer_dim = layer.shape[1]
            print("sequence output layer:", seq_output_layer.shape)

        ## predict
        input_dim = seq_output_layer_dim

        ###
        ### Shared part
        ###
        # 32dim (Graph part)+ 32 dim (Sequence part)
#        layer = tf.concat([seq_output_layer, graph_output_layer], axis=1)
        layer = tf.concat([seq_output_layer], axis=1)
#        input_dim = seq_output_layer_dim+graph_output_layer_dim
        input_dim = seq_output_layer_dim
        with tf.variable_scope("shared_nn") as scope_part:
            layer = K.layers.Dense(50)(layer)
            #layer=K.layers.BatchNormalization()(layer)
            layer = tf.nn.relu(layer)
            layer = K.layers.Dense(info.label_dim)(layer)

        prediction = tf.nn.softmax(layer)
        # computing cost and metrics
        cost = mask * \
            tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=layer)
        cost_opt = tf.reduce_mean(cost)

        metrics = {}
        cost_sum = tf.reduce_sum(cost)

        correct_count = mask * \
            tf.cast(tf.equal(tf.argmax(prediction, 1),
                             tf.argmax(labels, 1)), tf.float32)
        metrics["correct_count"] = tf.reduce_sum(correct_count)
        self.out = layer
        return self, prediction, cost_opt, cost_sum, metrics

    def embedding(self, sess, data):
        key = self.placeholders['sequences']
        feed_dict = {key: data}
        out = sess.run(self.embedding_layer, feed_dict)
        return out

