import tensorflow as tf
from keras import backend as K


# ========================  Loss Functions ========================


def jaccard_loss(class_weights=None):
    """
    Jaccard loss for binary or multi-class segmentation with optional class weights.

    Args:
        class_weights (list or None): List of weights for each class. Default is None.

    Returns:
        loss (function): A loss function.
    """
    def loss(y_true, y_pred):
        """
        Args:
            y_true (tensor): True labels, either binary or one-hot encoded.
            y_pred (tensor): Predictions.

        Returns:
            loss (tensor): Computed Jaccard loss.
        """

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute intersection and union
        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        sum_ = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
        union = sum_ - intersection

        # Compute the Jaccard index
        jaccard = (intersection + K.epsilon()) / (union + K.epsilon())

        # Compute the Jaccard loss
        if class_weights is not None:
            class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
            weighted_jaccard_sum = tf.reduce_sum(class_weights_tensor * jaccard)
            class_weights_sum = tf.reduce_sum(class_weights_tensor)
            jaccard_loss = 1 - weighted_jaccard_sum / class_weights_sum
        else:
            jaccard_loss = 1 - tf.reduce_mean(jaccard)

        return jaccard_loss

    return loss


def dice_loss(class_weights=None):
    """
    Dice loss for binary or multi-class segmentation with optional class weights.

    Args:
        class_weights (list or None): List of weights for each class. Default is None.

    Returns:
        loss (function): A loss function.
    """
    def loss(y_true, y_pred):
        """
        Args:
            y_true (tensor): True labels, either binary or one-hot encoded.
            y_pred (tensor): Predictions.

        Returns:
            loss (tensor): Computed Dice loss.
        """

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute intersection and sums
        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        sum_y_true = tf.reduce_sum(y_true, axis=[0, 1, 2])
        sum_y_pred = tf.reduce_sum(y_pred, axis=[0, 1, 2])

        # Compute the Dice coefficient
        dice = (2 * intersection + K.epsilon()) / (sum_y_true + sum_y_pred + K.epsilon())

        # Compute the Dice loss
        if class_weights is not None:
            class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
            weighted_dice_sum = tf.reduce_sum(class_weights_tensor * dice)
            class_weights_sum = tf.reduce_sum(class_weights_tensor)
            dice_loss = 1 - weighted_dice_sum / class_weights_sum
        else:
            dice_loss = 1 - tf.reduce_mean(dice)

        return dice_loss

    return loss


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification.

    Args:
        gamma (float): Focusing parameter. Default is 2.0.
        alpha (float): Balancing factor. Default is 0.25.

    Returns:
        loss (function): A loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Args:
            y_true (tensor): True labels, one-hot encoded.
            y_pred (tensor): Predictions.

        Returns:
            loss (tensor): Computed focal loss.
        """

        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute cross-entropy loss
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Compute the focal loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses over the last axis (number of classes)
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fixed


# ======================== Metrics ========================


def iou_score():
    """
    Jaccard Index (Intersection over Union) metric.

    Returns:
        metric (function): A metric function.
    """
    def metric(y_true, y_pred):

        y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))


        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        sum_ = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
        union = sum_ - intersection


        jaccard = (intersection + K.epsilon()) / (union + K.epsilon())
        
        return tf.reduce_mean(jaccard)
    
    return metric


def dice_score():
    """
    Dice Score metric.

    Returns:
        metric (function): A metric function.
    """
    def metric(y_true, y_pred):

        y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))


        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        sum_y_true = tf.reduce_sum(y_true, axis=[0, 1, 2])
        sum_y_pred = tf.reduce_sum(y_pred, axis=[0, 1, 2])


        dice = (2 * intersection + K.epsilon()) / (sum_y_true + sum_y_pred + K.epsilon())
        
        return tf.reduce_mean(dice)
    
    return metric

