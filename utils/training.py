# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
#
# This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)

"""Training class and some functions for training RAMS"""
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import Mean
import os

def random_flip(lr_img, hr_img, hr_img_mask):
    """Data Augmentation: flip data samples randomly"""
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img, hr_img_mask),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img),
                            tf.image.flip_left_right(hr_img_mask)))


def random_rotate(lr_img, hr_img, hr_img_mask):
    """Data Augmentation: rotate data samples randomly of a 90 degree angle"""
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn), tf.image.rot90(hr_img_mask, rn)

class Trainer(object):
    """
    Train a network and manage weights loading and saving
    
    ...
    
    Attributes
    ----------
    model: obj
        model to be trained
    band: string
        band to train with
    image_hr_size: int
        size of the HR image
    name_net: string
        name of the network
    loss: obj
        loss function
    metric: obj
        metric function
    optimizer: obj
        optimizer of the training
    checkpoint_dir: string
        weights path
    log_dir: string
        logs path
 
    Methods
    -------
    restore()
        Restore a previous version found in 'checkpoint_dir' path
    fit(self, x=None, y=None, batch_size=None, buffer_size=512, epochs=100,
            verbose=1, evaluate_every=100, val_steps=100,
            validation_data=None, shuffle=True, initial_epoch=0, save_best_only=True,
           data_aug = False)
        Train the network with the configuration passed to the function
    train_step(self, lr, hr, mask)
        A single training step
    test_step(self, lr, hr, mask)
        A single testing step
    """
    def __init__(self,
                 model, band, image_hr_size, name_net,
                 loss, metric,
                 optimizer,
                 checkpoint_dir='./checkpoint', log_dir='logs'):

        self.now = None
        self.band = band
        self.name_net = name_net
        self.loss = loss
        self.image_hr_size = image_hr_size
        self.metric = metric
        self.log_dir = log_dir
        self.train_loss = Mean(name='train_loss')
        self.train_psnr = Mean(name='train_psnr')

        self.test_loss = Mean(name='test_loss')
        self.test_psnr = Mean(name='test_psnr')
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(1.0),
                                              optimizer=optimizer,
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

    @property
    def model(self):
        return self.checkpoint.model
    
    def fit(self, x=None, y=None, batch_size=None, buffer_size=512, epochs=100,
            verbose=1, evaluate_every=100, val_steps=100,
            validation_data=None, shuffle=True, initial_epoch=0, save_best_only=True,
           data_aug = False):

        ds_len = x.shape[0]
        # Create dataset from slices
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x, *y)).shuffle(buffer_size, 
                              reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        if data_aug:
            train_ds.map(random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_ds.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        

        val_ds = tf.data.Dataset.from_tensor_slices(
            (validation_data[0], *validation_data[1])).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).take(val_steps)

        # Tensorboard logger
        writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir, f'train_{self.band}_{self.name_net}'))
        writer_test = tf.summary.create_file_writer(os.path.join(self.log_dir, f'test_{self.band}_{self.name_net}'))

        global_step = tf.cast(self.checkpoint.step,tf.int64)
        total_steps = tf.cast(ds_len/batch_size,tf.int64)
        step = tf.cast(self.checkpoint.step,tf.int64) % total_steps 
        
        
        for epoch in range(epochs - initial_epoch):
            # Iterate over the batches of the dataset.
            print("\nEpoch {}/{}".format(epoch + 1 + initial_epoch, epochs))
            pb_i = Progbar(ds_len, stateful_metrics=['Loss', 'PSNR', 'Val Loss', 'Val PSNR'])
            
            for x_batch_train, y_batch_train, y_mask_batch_train in train_ds:
                if (total_steps - step) == 0:
                    step = tf.cast(self.checkpoint.step,tf.int64) % total_steps 

                    # Reset metrics
                    self.train_loss.reset_states()
                    self.train_psnr.reset_states()


                step +=1
                global_step += 1
                self.train_step(x_batch_train, y_batch_train, y_mask_batch_train)
                
                self.checkpoint.step.assign_add(1)

                
                with writer_train.as_default():
                    tf.summary.scalar(
                        'PSNR', self.train_psnr.result(), step=global_step)

                    tf.summary.scalar(
                        'Loss', self.train_loss.result(), step=global_step)
                    

                if step != 0 and (step % evaluate_every) == 0:
                    # Reset states for test
                    self.test_loss.reset_states()
                    self.test_psnr.reset_states()

                    for x_batch_val, y_batch_val, y_mask_batch_val in val_ds:
                        self.test_step(x_batch_val, y_batch_val, y_mask_batch_val)
                        
                    with writer_test.as_default():                            
                        tf.summary.scalar(
                            'Loss', self.test_loss.result(), step=global_step)
                        tf.summary.scalar(
                            'PSNR', self.test_psnr.result(), step=global_step)
                    

                    writer_train.flush()
                    writer_test.flush()

                    if save_best_only and (self.test_psnr.result() <= self.checkpoint.psnr):
                        # skip saving checkpoint, no PSNR improvement
                        continue
                    self.checkpoint.psnr = self.test_psnr.result()
                    self.checkpoint_manager.save()
                    
                values = [('Loss', self.train_loss.result()), ('PSNR', self.train_psnr.result()),
                          ('Val Loss', self.test_loss.result()), ('Val PSNR', self.test_psnr.result())]
                pb_i.add(batch_size, values=values)

    @tf.function
    def train_step(self, lr, hr, mask):
        lr = tf.cast(lr, tf.float32)
        
        with tf.GradientTape() as tape:

            sr = self.checkpoint.model(lr, training=True)
            loss = self.loss(hr, sr, mask, self.image_hr_size)

        gradients = tape.gradient(
            loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(
            zip(gradients, self.checkpoint.model.trainable_variables))

        metric = self.metric(hr, sr, mask)
        self.train_loss(loss)
        self.train_psnr(metric)

    @tf.function
    def test_step(self, lr, hr, mask):
        lr = tf.cast(lr, tf.float32)
        
        sr = self.checkpoint.model(lr, training=False)
        t_loss = self.loss(hr, sr, mask, self.image_hr_size)
        t_metric = self.metric(hr, sr, mask)

        self.test_loss(t_loss)
        self.test_psnr(t_metric)
