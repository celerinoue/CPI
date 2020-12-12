import tensorflow as tf
import numpy as np
import joblib
import kgcn.layers


class MultiModalNetwork(object):
    def __init__(self,
                 batch_size,
                 input_dim,
                 adj_channel_num,
                 sequence_symbol_num,
                 graph_node_num,
                 label_dim,
                 pos_weight,
                 embedding_dim: int=25,
                 feed_embedded_layer: bool=False,
    ):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.adj_channel_num = adj_channel_num
        self.sequence_symbol_num = sequence_symbol_num
        self.graph_node_num = graph_node_num
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.pos_weight = pos_weight

        self.feed_embedded_layer = feed_embedded_layer

    @classmethod
    def build_placeholders(cls, info, config, batch_size=4):
        raise NotImplementedError        

    def build_model(self):
        raise NotImplementedError
        
def multitask_logits(features,
                    num_tasks,
                    num_classes=2,
                    weight_init=None,
                    bias_init=None,
                    dropout_prob=None,
                    name=None):
    """Create a logit tensor for each classification task.

    Args:
        features: A 2D tensor with dimensions batch_size x num_features.
        num_tasks: Number of classification tasks.
        num_classes: Number of classes for each task.
        weight_init: Weight initializer.
        bias_init: Bias initializer.
        dropout_prob: Float giving dropout probability for weights (NOT keep
        probability).
        name: Name for this op. Defaults to 'multitask_logits'.

    Returns:
        A list of logit tensors; one for each classification task.
    """
    logits_list = []
    with tf.name_scope('multitask_logits'):
        for task_idx in range(num_tasks):# 12
            with tf.name_scope(name,
                                ('task' + str(task_idx).zfill(len(str(num_tasks)))),
                                [features]):
                logits_list.append(
                    logits(
                        features,# 出力
                        num_classes,# 2
                        weight_init=weight_init,# None
                        bias_init=bias_init,# None
                        dropout_prob=dropout_prob))
    return logits_list# shape：[12×50×2]

def logits(features,
           num_classes=2,
           weight_init=None,
           bias_init=None,
           dropout_prob=None,
           name=None):
  """Create a logits tensor for a single classification task.

  You almost certainly don't want dropout on there -- it's like randomly setting
  the (unscaled) probability of a target class to 0.5.

  Args:
    features: A 2D tensor with dimensions batch_size x num_features.
    num_classes: Number of classes for each task.
    weight_init: Weight initializer.
    bias_init: Bias initializer.
    dropout_prob: Float giving dropout probability for weights (NOT keep
      probability).
    name: Name for this op.

  Returns:
    A logits tensor with shape batch_size x num_classes.
  """
  with tf.name_scope(name, 'logits', [features]) as name:
    return fully_connected_layer(
            features,
            num_classes,# 2
            weight_init=weight_init,
            bias_init=bias_init,
            name=name)


def fully_connected_layer(tensor,
                          size=None,
                          weight_init=None,
                          bias_init=None,
                          name=None):
    """Fully connected layer.

    Parameters
    ----------
    tensor: tf.Tensor
        Input tensor.
    size: int
        Number of output nodes for this layer.
    weight_init: float
        Weight initializer.
    bias_init: float
        Bias initializer.
    name: str
        Name for this op. Defaults to 'fully_connected'.

    Returns
    -------
    tf.Tensor:
        A new tensor representing the output of the fully connected layer.

    Raises
    ------
    ValueError
        If input tensor is not 2D.
    """
    if weight_init is None:
        num_features = tensor.get_shape()[-1].value# 128
        weight_init = tf.truncated_normal([num_features, size], stddev=0.01)#size:2
        # weight_init : shape[128×2]
    if bias_init is None:
        bias_init = tf.zeros([size])# size：2

    with tf.name_scope(name, 'fully_connected', [tensor]):
        w = tf.Variable(weight_init, name='w', dtype=tf.float32)
        b = tf.Variable(bias_init, name='b', dtype=tf.float32)
    return tf.nn.xw_plus_b(tensor, w, b)#  matmul(x, weights) + biases.

def loss_fn(x, t, w):
    costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=t)
    w = tf.reshape(w,[t.shape[0].value,1])
    # 不均衡を加味し、欠損値を省く
    weighted_costs = tf.multiply(costs, w)
    return tf.reduce_sum(weighted_costs)

def add_training_loss(logits,label,pos_weight,batch_size,n_tasks,mask):
    """Computes loss using logits.
    return
    各タスクごとのloss
    shape:12
    """
    task_losses = []
    # label_placeholder of shape (batch_size, n_tasks). Split into n_tasks
    # tensors of shape (batch_size,)
    task_labels = tf.split(# ラベルを12分割　12×50×１
        axis=1, num_or_size_splits=n_tasks, value=label)
    task_weights = pos_weight#tf.split(# 重みを12分割　12×None×１
        # axis=1, num_or_size_splits=n_tasks, value=pos_weight)
    for task in range(n_tasks):
        task_label_vector = task_labels[task]
        task_mask = mask[:,task]
        task_weight = task_weights[task]
        # 各taskのweightの入った[dim(50)],欠損値の部分は0
        task_weight_vector = task_mask * task_weight
        # 正解が0の部分を１とするtensor
        task_zeros_one_hot = tf.subtract(tf.ones_like(task_label_vector), task_label_vector)
        # task_weight_vectorの正解値が0の部分を0とし、1を加える。task_weight_vectorの正解値が0の値を0とするため
        task_weight_vector = tf.reshape(task_weight_vector,[task_weight_vector.shape[0].value,1])
        task_weight_vector = tf.add(tf.multiply(task_weight_vector , task_label_vector) , task_zeros_one_hot)

        # Convert the labels into one-hot vector encodings.
        one_hot_labels = tf.to_float(
            tf.one_hot(tf.to_int32(tf.squeeze(task_label_vector)), 2))
        # Since we use tf.nn.softmax_cross_entropy_with_logits note that we pass in
        # un-softmaxed logits rather than softmax outputs.

        if len(one_hot_labels.shape) == 1:
            # When batch-size is 1, this shape doesn't have batch-size dimmension.
            one_hot_labels = tf.expand_dims(one_hot_labels, axis=0)
        
        task_loss = loss_fn(logits[task], one_hot_labels, task_weight_vector)
        task_losses.append(task_loss)
    # It's ok to divide by just the batch_size rather than the number of nonzero
    # examples (effect averages out)
    # total_loss = tf.add_n(task_losses)#全タスクのlossを合計
    # total_loss = tf.div(total_loss, batch_size)# バッチサイズで割る
    # return total_loss

    total_loss = tf.add_n(task_losses)#全タスクのlossをそれぞれ合計
    task_losses = tf.div(task_losses, batch_size)
    return task_losses#各タスクの損失バッチ平均

def add_softmax(outputs):
    """Replace logits with softmax outputs."""
    softmax = []
    with tf.name_scope('inference'):
        for i, logits in enumerate(outputs):
            softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
    return softmax


