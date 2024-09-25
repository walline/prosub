import tensorflow as tf
import math
import csv
import os

class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_steps, decay_factor=7/8, pretrain_steps=0):

        super(CosineDecay, self).__init__()

        # TODO: maybe rename attributes to decay_end and decay_start

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps # where to end decay
        self.decay_factor = decay_factor
        self.pretrain_steps = pretrain_steps # where to start decay

    def __call__(self, step):

        with tf.name_scope("CosineDecay"):

            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_factor = tf.cast(self.decay_factor, dtype)

            pretrain_steps = tf.cast(self.pretrain_steps, dtype)
            
            global_step_recomp = tf.cast(step, dtype)
            complete_fraction = tf.clip_by_value(
                (global_step_recomp-pretrain_steps)/(decay_steps-pretrain_steps),
                0.0,
                1.0)

            decay_factor = tf.cos(decay_factor*tf.constant(math.pi, dtype=dtype)*complete_fraction/2.0)

            return tf.multiply(initial_learning_rate, decay_factor)

class ConstantLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate,):
        super(ConstantLR, self).__init__()
        self.learning_rate = tf.cast(learning_rate, tf.float32)

    def __call__(self, step):
        return self.learning_rate
    

def log_partition(logits):
    return -tf.reduce_logsumexp(logits, axis=1)

def dataset_cardinality(dataset):
    # TODO: should implement something to avoid infinite loops here
    
    count = 0
    for item in dataset:
        count += 1

    return count
    
def auroc_rank_calculation(id_vals, ood_vals):

    pairwise_tests = tf.greater(id_vals[:, tf.newaxis], ood_vals[tf.newaxis, :])
    num_pairs = tf.size(id_vals)*tf.size(ood_vals)
    num_pairs = tf.cast(num_pairs, tf.float32)
    auroc = tf.divide(tf.reduce_sum(tf.cast(pairwise_tests, tf.float32)), num_pairs)
    return auroc

def print_model_summary(model, name):

    WIDTH=70

    def shorten_variable_name(name_string):
        splitted = name_string.split("/")
        if len(splitted)>1:
            return "{}/{}".format(splitted[-2], splitted[-1])
        else:
            return splitted[-1]

    format_string = "{:<35} {:<20} {}"

    print("="*WIDTH)
    print("Model: {}".format(name))
    print("-"*WIDTH)
    print(format_string.format("Variable", "Shape", "Nr. params"))
    print("-"*WIDTH)

    for var in model.trainable_variables:
        print(format_string.format(
            shorten_variable_name(var.name),
            str(var.shape.as_list()),
            tf.reduce_prod(var.shape).numpy()))

    print("-"*WIDTH)
    print("Nr trainable params: {:,}".format(
        sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])))
    print("Nr non-trainable params: {:,}".format(
        sum([tf.reduce_prod(var.shape) for var in model.non_trainable_variables])))
    print("="*WIDTH)
    
        
def save_stats_csv(data, labels, save_dir, step, floatdecimals=4):

    fname = os.path.join(save_dir, "stats_{}.csv".format(step.read_value()))
    assert len(data) == len(labels)

    def set_precision(list_, decimals=floatdecimals):
        return ["{:.{p}f}".format(x, p=decimals) for x in list_]

    data = [set_precision(x) if isinstance(x[0], float) else x for x in data]
    
    zipdata = list(zip(*data))

    with open(fname, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        writer.writerows(zipdata)
    
