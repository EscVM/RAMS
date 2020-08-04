# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
#
# This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)

"""RAMS functions for training"""
import tensorflow as tf

#-------------
# Settings
#-------------
HR_SIZE = 96


def log10(x):
    """
    Compute log base 10
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def l1_loss(y_true, y_pred, y_mask, HR_SIZE=HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = tf.cast(
                cropped_predictions, tf.float32)*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            l1_loss = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.abs(
                    tf.subtract(cropped_labels_masked,
                                corrected_cropped_predictions)
                ), axis=[1, 2]
            )
            X.append(l1_loss)
    X = tf.stack(X)
    min_l1 = tf.reduce_min(X, axis=0)

    return min_l1

def psnr(y_true, y_pred, y_mask, size_image=HR_SIZE):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = tf.cast(
                cropped_predictions, tf.float32)*cropped_y_mask
            cropped_labels_masked = tf.cast(
                cropped_labels, tf.float32)*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            corrected_mse = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.square(
                    tf.subtract(cropped_labels_masked,
                                corrected_cropped_predictions)
                ), axis=[1, 2])

            cPSNR = 10.0*log10((65535.0**2)/corrected_mse)
            X.append(cPSNR)

    X = tf.stack(X)
    max_cPSNR = tf.reduce_max(X, axis=0)  
    return tf.reduce_mean(max_cPSNR)



def ssim(y_true, y_pred, y_mask, size_image=HR_SIZE, clear_only=False):
    """
    Modified SSIM metric to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = tf.cast(
                cropped_predictions, tf.float32)*cropped_y_mask
            cropped_labels_masked = tf.cast(
                cropped_labels, tf.float32)*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])
            
            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            cSSIM = tf.image.ssim(corrected_cropped_predictions, cropped_labels_masked,65535)
            if clear_only:
                cSSIM = (cSSIM-1)*total_pixels_masked/clear_pixels+1
            X.append(cSSIM)

    X = tf.stack(X)
    max_cSSIM = tf.reduce_max(X, axis=0)  
    return tf.reduce_mean(max_cSSIM)
