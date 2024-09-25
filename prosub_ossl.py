import tensorflow as tf
from absl import flags, app
import math
from tqdm import trange, tqdm

from models import get_model, EmaPredictor
from utils import (CosineDecay, auroc_rank_calculation, dataset_cardinality,
                   log_partition, save_stats_csv,)
from base import BaseModel
import os
import estimator
from data import OpenSSLDataSets, PARSEDICT
from augment import weak_augmentation_pair, weak_augmentation, rand_augment_cutout_batch
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

FLAGS = flags.FLAGS

class ProSub(BaseModel):

    def setup_model(self, arch, nclass, wd, ws, wpl, wsub, lr, momentum, **kwargs):

        # metrics that are not updated at training time
        # these are placed outside of strategy scope
        self.metrics = {
            "test/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/accuracy_ema": tf.keras.metrics.SparseCategoricalAccuracy(),
            "test/xe_loss": tf.keras.metrics.Mean(),
            "valid/accuracy_ema": tf.keras.metrics.SparseCategoricalAccuracy(),
            "monitors/wd_loss": tf.keras.metrics.Mean(),
            "monitors/lr": tf.keras.metrics.Mean(),
        }            
        
        with self.strategy.scope():

            self.models["classifier"] = get_model(arch, nclass)

            self.models["classifier"].build(ema_decay=FLAGS.ema_decay,
                                            input_shape=[None] + self.datasets.shape)

            output_shape = self.models["classifier"].compute_output_shape(
                input_shape=[None] + self.datasets.shape)

            feature_dimension = output_shape[-1][-1]

            self.models["projection_head"] = tf.keras.layers.Dense(
                feature_dimension,
                kernel_initializer=tf.keras.initializers.GlorotNormal())
            self.models["projection_head"].build(input_shape=[None, feature_dimension])

            self.wd = tf.constant(wd, tf.float32)
            self.ws = tf.constant(ws, tf.float32)
            self.wpl = tf.constant(wpl, tf.float32)
            self.wsub = tf.constant(wsub, tf.float32)

            # initial values for estimation of Beta distributions
            # TODO: could make these input arguments
            self.alpha1_init = 10.0
            self.beta1_init = 2.0
            self.alpha2_init = 2.0
            self.beta2_init = 10.0

            self.alpha1 = tf.Variable(self.alpha1_init, dtype=tf.float32, trainable=False,
                                      aggregation=tf.VariableAggregation.MEAN)
            self.beta1 = tf.Variable(self.beta1_init, dtype=tf.float32, trainable=False,
                                     aggregation=tf.VariableAggregation.MEAN)
            self.alpha2 = tf.Variable(self.alpha2_init, dtype=tf.float32, trainable=False,
                                      aggregation=tf.VariableAggregation.MEAN)
            self.beta2 = tf.Variable(self.beta2_init, dtype=tf.float32, trainable=False,
                                     aggregation=tf.VariableAggregation.MEAN)

            self.pi = tf.Variable(FLAGS.pi, dtype=tf.float32, trainable=False,
                                  aggregation=tf.VariableAggregation.MEAN)
            
            self.class_prototypes = tf.Variable(
                tf.zeros([nclass, feature_dimension], dtype=tf.float32),
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )
            
            self.cosine_learning_rate = CosineDecay(lr,
                                                    decay_steps=FLAGS.trainsteps,
                                                    decay_factor=FLAGS.decayfactor,
                                                    pretrain_steps=FLAGS.pretrainsteps)
            
            self.optimizer = tf.keras.optimizers.SGD(self.cosine_learning_rate,
                                                     momentum=tf.Variable(momentum),
                                                     nesterov=True)

            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
            

            # need this line because of some tf bug when restoring checkpoints
            self.optimizer.decay=tf.Variable(0.0)            

            # metrics that are updated at training time (inside strategy scope)
            self.metrics.update({
                "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
                "train/xe_loss": tf.keras.metrics.Mean(),
                "train/us_loss": tf.keras.metrics.Mean(),
                "train/sub_loss": tf.keras.metrics.Mean(),
                "monitors/mask": tf.keras.metrics.Mean(),
                "monitors/wpl": tf.keras.metrics.Mean(),
                "monitors/ood_mask": tf.keras.metrics.Mean(),
            })

            # initialize checkpoint
            ckpt_dir = os.path.join(self.train_dir, "checkpoints")
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                            optimizer=self.optimizer,
                                            models=self.models,
                                            alpha1 = self.alpha1,
                                            beta1 = self.beta1,
                                            alpha2 = self.alpha2,
                                            beta2 = self.beta2,
                                            class_prototypes = self.class_prototypes,
                                            )
            
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=FLAGS.keepckpt)

            # restore from previous checkpoint if exists
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
                assert self.optimizer.iterations == self.ckpt.step
            else:
                print("Initializing from scratch")

                
        with tf.device('CPU:0'):
            self.log_weak_augmentation = tf.Variable(
                initial_value=tf.zeros(self.datasets.shape, dtype=tf.float32),
                trainable=False)
            
            self.log_strong_augmentation = tf.Variable(
                initial_value=tf.zeros(self.datasets.shape, dtype=tf.float32),
                trainable=False)

            self.testset_id_size = dataset_cardinality(self.datasets.test)
            self.testset_ood_size = dataset_cardinality(self.datasets.test_ood)
            self.testset_unseen_size = dataset_cardinality(self.datasets.test_unseen)
            
            self.labelset_size = dataset_cardinality(self.datasets.train_labeled)
            self.ulset_size = (dataset_cardinality(self.datasets.train_unlabeled_id)
                               + dataset_cardinality(self.datasets.train_unlabeled_ood))

            self.scores = ["energy", "conf", "maxlogit", "subspacescore"]

            self.scores_testset_id = {}
            self.scores_testset_ood = {}
            self.scores_testset_unseen = {}
            self.scores_ulset = {}
            self.scores_ulset_ema = {}

            for key in self.scores:
                self.scores_testset_id[key] = tf.Variable(tf.zeros(self.testset_id_size, dtype=tf.float32),
                                                          trainable=False)
                self.scores_testset_ood[key] = tf.Variable(tf.zeros(self.testset_ood_size, dtype=tf.float32),
                                                           trainable=False)
                self.scores_testset_unseen[key] = tf.Variable(tf.zeros(self.testset_unseen_size, dtype=tf.float32),
                                                           trainable=False)
                self.scores_ulset[key] = tf.Variable(tf.zeros(self.ulset_size, dtype=tf.float32),
                                                     trainable=False)
                self.scores_ulset_ema[key] = tf.Variable(tf.zeros(self.ulset_size, dtype=tf.float32),
                                                         trainable=False)

            self.ulset_labels = tf.Variable(tf.zeros(self.ulset_size, dtype=tf.int64),
                                            trainable=False)
            
    @tf.function
    def p_id(self, x):

        pdf_id = estimator.beta_pdf(x, self.alpha1, self.beta1)
        pdf_ood = estimator.beta_pdf(x, self.alpha2, self.beta2)
        
        pdf_joint = self.pi*pdf_id + (1 - self.pi)*pdf_ood
        p_id = self.pi*pdf_id/(pdf_joint + 1e-1)
        return p_id

    @tf.function
    def train_step(self, labeled_inputs, unlabeled_inputs):


        labeled_images = labeled_inputs["image"]
        labels = labeled_inputs["label"]

        unlabeled_images = unlabeled_inputs["image"]

        local_batch = int(FLAGS.batch / self.strategy.num_replicas_in_sync)
        uratio = int(FLAGS.uratio)

        step_fn = tf.maximum(0.0, tf.cast(tf.sign(self.ckpt.step-FLAGS.pretrainsteps), tf.float32))
        wpl = step_fn*self.wpl
        wsub = step_fn*self.wsub
        
        with tf.GradientTape() as tape:

            x = tf.concat([labeled_images, unlabeled_images[:,0], unlabeled_images[:,1]], 0)
            
            logits, embeds = self.models["classifier"](x, training=True)
            logits = tf.cast(logits, tf.float32)
            embeds = tf.cast(embeds, tf.float32)            

            embeds_strong = embeds[-local_batch*uratio:]
            embeds_dm = self.models["projection_head"](embeds_strong)
            embeds_dm = tf.cast(embeds_dm, tf.float32)
            
            embeds_weak = embeds[local_batch:-local_batch*uratio]
            
            logits_labeled = logits[:local_batch]
            logits_weak, logits_strong = tf.split(logits[local_batch:], 2)
            
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                logits_labeled,
                                                                from_logits=True))

            pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))

            xep_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(pseudo_labels, axis=1),
                                                                       logits_strong,
                                                                       from_logits=True)


            q, r = tf.linalg.qr(tf.transpose(self.class_prototypes))
            qt = tf.transpose(q)

            projs = tf.stop_gradient(tf.matmul(tf.matmul(embeds_weak, q), qt))
            norm_projs = tf.linalg.l2_normalize(projs, axis=-1)
            norm_embeds = tf.linalg.l2_normalize(embeds_weak, axis=-1)
            norm_dm = tf.linalg.l2_normalize(embeds_dm, axis=-1)

            sims = tf.reduce_sum(
                tf.multiply(norm_projs, norm_embeds),
                axis=-1)

            p_id = self.p_id(tf.stop_gradient(sims))

            rand = tf.random.uniform([local_batch*uratio])
            id_mask = rand <= p_id
            fixmatch_mask = tf.reduce_max(pseudo_labels, axis=1) >= FLAGS.tau

            pseudo_mask = tf.logical_and(id_mask, fixmatch_mask)
            ood_mask = tf.logical_not(id_mask)

            pseudo_mask = tf.cast(pseudo_mask, tf.float32)
            ood_mask = tf.cast(ood_mask, tf.float32)
            id_mask = tf.cast(id_mask, tf.float32)
                        
            xep_loss = tf.reduce_mean(xep_loss * pseudo_mask)

            # unsupervised representation loss
            doublematch_similarity = tf.reduce_sum(
                tf.multiply(norm_dm, tf.stop_gradient(norm_embeds)),
                axis=-1)
            us_loss = tf.reduce_mean(-doublematch_similarity+1)

            sub_loss = tf.reduce_mean((ood_mask-id_mask)*sims)
            
            variables = self.models["classifier"].trainable_variables + \
                self.models["projection_head"].trainable_variables
            
            l2_loss = sum(tf.nn.l2_loss(v) for v in variables
                          if "kernel" in v.name)

            full_loss =  (xe_loss + wpl*xep_loss + self.ws*us_loss + self.wd*l2_loss
                          + wsub*sub_loss)

            # scale loss for the current strategy b.c. apply_gradients sums over all replicas
            full_loss = full_loss / self.strategy.num_replicas_in_sync
            full_loss = self.optimizer.get_scaled_loss(full_loss)


        ctx = tf.distribute.get_replica_context()


        embeds_labeled = embeds[:local_batch]

        embeds_labeled, labels_gathered, sims = ctx.all_gather(
            (embeds_labeled, labels, sims), axis=0)
        
        mean_class_prototypes = tf.math.unsorted_segment_mean(
            embeds_labeled,
            labels_gathered,
            self.datasets.nclass
        )

        label_counts = tf.math.bincount(tf.cast(labels_gathered, tf.int32),
                                        minlength=self.datasets.nclass)
        
        mask = tf.reshape(tf.cast(label_counts == 0, tf.float32),
                          (self.datasets.nclass,1))

        mean_class_prototypes += mask*self.class_prototypes

        embeds_labeled_projs = tf.matmul(tf.matmul(embeds_labeled, q), qt)
        embeds_labeled_norm = tf.linalg.l2_normalize(embeds_labeled, axis=-1)
        embeds_labeled_proj_norm = tf.linalg.l2_normalize(embeds_labeled_projs, axis=-1)
        labeled_sims = tf.reduce_sum(
                tf.multiply(embeds_labeled_norm, embeds_labeled_proj_norm),
                axis=-1)

        p1 = estimator.beta_pdf(sims, self.alpha1, self.beta1)
        p2 = estimator.beta_pdf(sims, self.alpha2, self.beta2)
                
        w1 = self.pi * p1
        w2 = (1 - self.pi) * p2
        p = w1 / (w1 + w2 + 1e-8)

        id_weights = tf.concat([tf.ones([FLAGS.batch], tf.float32), p], axis=0)
        ood_weights = 1.0 - p
        id_sims = tf.concat([labeled_sims, sims], axis=0)

        alpha1, beta1 = estimator.estimator_beta_tf(id_sims, id_weights)
        alpha2, beta2 = estimator.estimator_beta_tf(sims, ood_weights)
        
        self.class_prototypes.assign_sub(
            (1. - FLAGS.ema_decay)*(self.class_prototypes - mean_class_prototypes))

        for var, x in zip((self.alpha1, self.beta1, self.alpha2, self.beta2),
                          (alpha1, beta1, alpha2, beta2)):
            var.assign_sub((1. - FLAGS.ema_decay)*(var - x))

        grads = tape.gradient(full_loss, variables)
        grads = self.optimizer.get_unscaled_gradients(grads)
        
        self.optimizer.apply_gradients(zip(grads, variables))
        self.models["classifier"].ema.apply(self.models["classifier"].trainable_variables)

        self.metrics["train/us_loss"].update_state(us_loss)
        self.metrics["train/xe_loss"].update_state(xe_loss)
        self.metrics["train/sub_loss"].update_state(sub_loss)
        self.metrics["train/accuracy"].update_state(labels, tf.nn.softmax(logits_labeled))
        self.metrics["monitors/mask"].update_state(tf.reduce_mean(pseudo_mask))
        self.metrics["monitors/ood_mask"].update_state(tf.reduce_mean(ood_mask))
        self.metrics["monitors/wpl"].update_state(wpl)
        
    @tf.function
    def test_step(self, inputs):

        images = inputs["image"]
        labels = inputs["label"]
        
        logits, embeds = self.models["classifier"](images, training=False)
        logits = tf.cast(logits, tf.float32)
        embeds = tf.cast(embeds, tf.float32)

        energies = log_partition(logits)
        maxlogit = tf.reduce_max(logits, axis=-1)
        conf = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)

        q, r = tf.linalg.qr(tf.transpose(self.class_prototypes))
        qt = tf.transpose(q)

        embed_projections = tf.matmul(tf.matmul(embeds, q), qt)

        proj_sims = tf.divide(
            tf.math.reduce_euclidean_norm(embed_projections, axis=-1),
            tf.math.reduce_euclidean_norm(embeds, axis=-1))        

        l2_loss = sum(tf.nn.l2_loss(v) for v in self.models["classifier"].trainable_variables
                          if "kernel" in v.name)

        return {
            "logits": logits,
            "energy": -1.0*energies,
            "conf": conf,
            "maxlogit": maxlogit,
            "subspacescore": proj_sims,
            "labels": labels,
            "l2_loss": l2_loss,
            }
    

    def save_stats(self, data_train_unlabeled_raw):

        batch = FLAGS.batch

        
        # make predictions on unlabeled training set
        for b, inputs in enumerate(data_train_unlabeled_raw):
            results = self.strategy.run(self.test_step,
                                        args=(inputs,))

            for key in self.scores:
                scores = self.strategy.gather(results[key], axis=0)
                self.scores_ulset[key][b*batch:(b+1)*batch].assign(scores)
            
            labels = self.strategy.gather(results["labels"], axis=0)

            self.ulset_labels[b*batch:(b+1)*batch].assign(labels)

        # make predictions on unlabeled trainig set using ema-model
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(data_train_unlabeled_raw):
                results = self.strategy.run(self.test_step,
                                            args=(inputs,))

                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_ulset_ema[key][b*batch:(b+1)*batch].assign(scores)

        savedata = []
        savelabels = []

        for key in self.scores:
            savedata.append(self.scores_ulset[key].numpy().tolist())
            savedata.append(self.scores_ulset_ema[key].numpy().tolist())
            savelabels.append(key)
            savelabels.append("{}_ema".format(key))

        savedata.append(self.ulset_labels.numpy().tolist())
        savelabels.append("labels")
        
        save_stats_csv(savedata,
                       savelabels,
                       self.train_dir,
                       self.ckpt.step)
        
    
    def evaluate_and_save_checkpoint(self,
                                     data_test,
                                     data_test_ood,
                                     data_train_unlabeled_raw,
                                     data_test_unseen,
                                     data_valid,
                                     **kwargs):

        batch = FLAGS.batch

        if FLAGS.savestats:
            self.save_stats(data_train_unlabeled_raw)

        # predictions on (ID) test set using non-ema model
        for inputs in tqdm(data_test,
                           desc="Evaluating test set (non-ema)",
                           unit="step",
                           leave=False,
                           mininterval=10):
            results = self.strategy.run(self.test_step, args=(inputs,))
            logits = self.strategy.gather(results["logits"], axis=0)
            labels = self.strategy.gather(results["labels"], axis=0)
            l2_loss = self.strategy.reduce("MEAN", results["l2_loss"], axis=None)
            xe_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))            

            self.metrics["test/accuracy"].update_state(labels, tf.nn.softmax(logits))
            self.metrics["test/xe_loss"].update_state(xe_loss)

        self.metrics["monitors/wd_loss"].update_state(l2_loss)

        current_lr = self.cosine_learning_rate(self.ckpt.step)
        self.metrics["monitors/lr"].update_state(current_lr)

        # predictions on test sets using ema model
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(tqdm(data_test,
                                            desc="Evaluating test set (ema)",
                                            unit="step",
                                            leave=False,
                                            mininterval=10)):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_id[key][b*batch:(b+1)*batch].assign(scores)
                
                logits = self.strategy.gather(results["logits"], axis=0)
                labels = self.strategy.gather(results["labels"], axis=0)
                self.metrics["test/accuracy_ema"].update_state(labels, tf.nn.softmax(logits))

            for b, inputs in enumerate(data_test_ood):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_ood[key][b*batch:(b+1)*batch].assign(scores)

            for b, inputs in enumerate(data_test_unseen):
                results = self.strategy.run(self.test_step, args=(inputs,))
                for key in self.scores:
                    scores = self.strategy.gather(results[key], axis=0)
                    self.scores_testset_unseen[key][b*batch:(b+1)*batch].assign(scores)

        # predictions on validation set using ema-model
        with EmaPredictor(self.models["classifier"], self.strategy):
            for b, inputs in enumerate(data_valid):
                results = self.strategy.run(self.test_step, args=(inputs,))                
                logits = self.strategy.gather(results["logits"], axis=0)
                labels = self.strategy.gather(results["labels"], axis=0)
                self.metrics["valid/accuracy_ema"].update_state(labels, tf.nn.softmax(logits))
            
        total_results = {name: metric.result() for name, metric in self.metrics.items()}

        with self.summary_writer.as_default():
            for name, result in total_results.items():
                tf.summary.scalar(name, result, step=self.ckpt.step)

            # log images
            tf.summary.image("Weak aug",
                             tf.expand_dims(self.log_weak_augmentation, 0),
                             step=self.ckpt.step)
            tf.summary.image("Strong aug",
                             tf.expand_dims(self.log_strong_augmentation, 0),
                             step=self.ckpt.step)

            # pairwise test auroc
            for key in self.scores:
                id_scores = self.scores_testset_id[key]
                ood_scores = self.scores_testset_ood[key]
                unseen_scores = self.scores_testset_unseen[key]
                auroc = auroc_rank_calculation(id_scores, ood_scores)
                auroc_unseen = auroc_rank_calculation(id_scores, unseen_scores)
                tf.summary.scalar("ood/auroc_{}".format(key), auroc, step=self.ckpt.step)
                tf.summary.scalar("ood/auroc_{}_{}".format(FLAGS.datasetunseen, key),
                                  auroc_unseen,
                                  step=self.ckpt.step)

            tf.summary.scalar("densities/alpha1", self.alpha1, step=self.ckpt.step)
            tf.summary.scalar("densities/beta1", self.beta1, step=self.ckpt.step)
            tf.summary.scalar("densities/alpha2", self.alpha2, step=self.ckpt.step)
            tf.summary.scalar("densities/beta2", self.beta2, step=self.ckpt.step)
                
        for metric in self.metrics.values():
            metric.reset_states()
        
        print("Saved checkpoint for step {}".format(self.ckpt.step.numpy()))
        print("Current test accuracy (ema): {}".format(total_results["test/accuracy_ema"]))

    def train(self, train_steps, eval_steps):

        batch = FLAGS.batch
        uratio = FLAGS.uratio

        shift = int(self.datasets.shape[0]*0.125)
        weakaug = lambda x: weak_augmentation(x, shift)
        weakaugpair = lambda x: weak_augmentation_pair(x, shift)
        parse = lambda x: PARSEDICT[FLAGS.dataset](x, self.datasets.shape)
        
        dl = self.datasets.train_labeled.shuffle(self.labelset_size)

        dvalid = dl.take(FLAGS.nvalid)
        dvalid = dvalid.map(parse, tf.data.AUTOTUNE).batch(batch)

        dl = dl.skip(FLAGS.nvalid).map(parse, tf.data.AUTOTUNE).cache()
        dl = dl.shuffle(FLAGS.shuffle).repeat().batch(batch).map(weakaug, tf.data.AUTOTUNE).prefetch(16)
        
        dul = self.datasets.train_unlabeled_id.concatenate(self.datasets.train_unlabeled_ood)
        dul = dul.shuffle(self.ulset_size).repeat()
        dul = dul.map(parse, tf.data.AUTOTUNE).batch(batch*uratio).map(weakaugpair, tf.data.AUTOTUNE)
        dul = dul.map(rand_augment_cutout_batch, tf.data.AUTOTUNE).prefetch(16)
        
        dul_raw = self.datasets.train_unlabeled_id.concatenate(self.datasets.train_unlabeled_ood)
        dul_raw = dul_raw.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)

        dt = self.datasets.test.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)

        dtood = self.datasets.test_ood.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)

        dtunseen = self.datasets.test_unseen.map(parse, tf.data.AUTOTUNE).batch(batch).prefetch(16)
        
        data_train_labeled = self.strategy.experimental_distribute_dataset(dl)
        data_train_unlabeled = self.strategy.experimental_distribute_dataset(dul)
        data_train_unlabeled_raw = self.strategy.experimental_distribute_dataset(dul_raw)
        data_test = self.strategy.experimental_distribute_dataset(dt)
        data_test_ood = self.strategy.experimental_distribute_dataset(dtood)
        data_test_unseen = self.strategy.experimental_distribute_dataset(dtunseen)
        data_valid = self.strategy.experimental_distribute_dataset(dvalid)

        labeled_iterator = iter(data_train_labeled)
        unlabeled_iterator = iter(data_train_unlabeled)

        assert FLAGS.trainsteps >= FLAGS.pretrainsteps
        nr_evals = math.ceil(FLAGS.trainsteps/FLAGS.evalsteps)
        
        # training loop
        while self.ckpt.step < FLAGS.trainsteps:
            
            self.evaluate_and_save_checkpoint(data_test,
                                              data_test_ood,
                                              data_train_unlabeled_raw,
                                              data_test_unseen,
                                              data_valid)
            
            
            desc = "Evaluation {}/{}".format(
                1+self.ckpt.step//FLAGS.evalsteps, nr_evals)
            loopstart = self.ckpt.step%FLAGS.evalsteps
            loopend = min(FLAGS.evalsteps,
                          loopstart + FLAGS.trainsteps - self.ckpt.step)

            loop = trange(loopstart,
                          loopend,
                          leave=False,
                          unit="step",
                          desc=desc,
                          mininterval=10)

            # evaluation loop
            for _ in loop:
                labeled_item = next(labeled_iterator)
                unlabeled_item = next(unlabeled_iterator)
                self.strategy.run(self.train_step, args=(labeled_item, unlabeled_item))
                self.ckpt.step.assign_add(1)
                

            # place augmented images in logging variables
            if self.strategy.num_replicas_in_sync > 1:
                unlabeled_images = unlabeled_item["image"].values[0]
            else:
                unlabeled_images = unlabeled_item["image"]

            self.log_weak_augmentation.assign((unlabeled_images[0,0,:,:,:]+1.0)/2.0)
            self.log_strong_augmentation.assign((unlabeled_images[0,1,:,:,:]+1.0)/2.0)

        assert self.ckpt.step == FLAGS.trainsteps
        assert self.optimizer.iterations == FLAGS.trainsteps
            
        # final evaluation
        self.evaluate_and_save_checkpoint(data_test,
                                          data_test_ood,
                                          data_train_unlabeled_raw,
                                          data_test_unseen,
                                          data_valid)



        
def main(argv):

    datasets = OpenSSLDataSets(FLAGS.dataset,
                               FLAGS.datasetood,
                               FLAGS.nlabeled,
                               FLAGS.seed,
                               FLAGS.datasetunseen)
    
    model = ProSub(
        os.path.join(FLAGS.traindir, datasets.name),
        datasets,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        ws=FLAGS.ws,
        wpl=FLAGS.wpl,
        wsub=FLAGS.wsub,
        nclass=datasets.nclass,
        momentum=FLAGS.momentum,
        batch=FLAGS.batch,
        arch=FLAGS.arch,
    )

    model.train(FLAGS.trainsteps, FLAGS.evalsteps)
    
if __name__ == '__main__':
    flags.DEFINE_string("datadir", None, "Directory for data")
    flags.DEFINE_string("traindir", "./experiments", "Directory for results and checkpoints")        
    flags.DEFINE_string("arch", "WRN-28-2", "Network architecture")
    flags.DEFINE_integer("trainsteps", int(1e5), "Number of training steps")
    flags.DEFINE_integer("evalsteps", int(5e3), "Number of steps between model evaluations")
    flags.DEFINE_integer("pretrainsteps", int(50e3), "Number of pretraining steps")
    flags.DEFINE_integer("batch", 64, "Batch size")
    flags.DEFINE_string("dataset", "cifar10", "Name of dataset")
    flags.DEFINE_string("datasetood", "svhn", "Name of ood dataset")
    flags.DEFINE_string("datasetunseen", "cifar100", "Name of unseen OOD set")
    flags.DEFINE_integer("uratio", 7, "Unlabeled batch size ratio")
    flags.DEFINE_integer("seed", 1, "Seed for labeled data")
    flags.DEFINE_integer("nlabeled", 40, "Number of labeled data")
    flags.DEFINE_integer("keepckpt", 1, "Number of checkpoints to keep")
    flags.DEFINE_float("wd", 0.0005, "Weight decay regularization")
    flags.DEFINE_float("ws", 5.0, "Weight for self-supervised loss")
    flags.DEFINE_float("wpl", 1.0, "Pseudo labeling loss weight")
    flags.DEFINE_float("wsub", 1.0, "Subspace loss weight")
    flags.DEFINE_integer("nvalid", 0, "Number of validation data")
    flags.DEFINE_boolean("savestats", False, "Save stats as csv files")
    flags.DEFINE_float("lr", 0.03, "Learning rate")
    flags.DEFINE_float("pi", 0.5, "Estimated fraction of ID data in unlabeled set")
    flags.DEFINE_float("decayfactor", 7/8, "Decay factor for cosine decay of learning rate")
    flags.DEFINE_float("tau", 0.95, "Threshold for pseudolabeleling (FixMatch style)")
    flags.DEFINE_float("ema_decay", 0.999, "Momentum parameter for exponential moving average for model parameters")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_string("rerun", "", "Additional experiment specification")
    flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')

    app.run(main)
