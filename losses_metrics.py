import tensorflow as tf
from keras.losses import Loss
from keras.metrics import Metric

"""
Inherits from Keras Loss and Metric classes:
https://keras.io/api/metrics/
https://keras.io/api/losses/
"""

# ======================== Constants ========================
# Constant for numerical stability and accuracy metrics.
EPSILON = 1e-7 #1.0

# ======================== Loss Functions ========================
class MSE_Loss(Loss):
    """
    MSE = (1/n) * Σ(y_true - y_pred)^2
    where Σ denotes summation over all samples, y_true is the true value,
    and y_pred is the predicted value.
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, 
                 name='mean_squared_error_loss'):
        super().__init__(reduction=reduction, name=name)


    def call(self, y_true, y_pred):

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        squared_difference = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(squared_difference)

        return mse
    

class DiceLoss(Loss):
    """  
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    where |X ∩ Y| denotes the intersection of the predicted and true sets,
    and |X| + |Y| is the sum of the elements in each set.
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, 
                 name='dice_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice_coeff = (2. * intersection + EPSILON) / (total + EPSILON)

        return 1 - dice_coeff
    

class JaccardLoss(Loss):
    """
    Jaccard Index (IoU) = |X ∩ Y| / |X ∪ Y|
    where |X ∩ Y| denotes the intersection of the predicted and true sets,
    and |X ∪ Y| is the union of the two sets.
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, 
                 name='jaccard_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        union = total - intersection
        jaccard_index = (intersection + EPSILON) / (union + EPSILON)

        return 1 - jaccard_index


class TverskyLoss(Loss):
    """
    The Tversky index is a generalization of the Dice coefficient 
    with additional control over FP and FN through 
    alpha (α) and beta (β) parameters.

    Tversky Index = (TP + EPSILON) / (TP + α * FN + β * FP + EPSILON)
    where TP is true positives, FN is false negatives, FP is false positives.
    """
    def __init__(self, alpha=0.6, beta=0.4, 
                  reduction=tf.keras.losses.Reduction.AUTO, 
                  name='tversky_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))
        
        tversky_index = (true_positives + EPSILON) / (true_positives \
                        + self.alpha * false_negatives + self.beta \
                        * false_positives + EPSILON)
        
        return 1 - tversky_index
    

class BinaryFocalLoss(Loss):
    """
    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t is the probability of the true class according to the model, 
    α_t is a weighting factor for balancing positive/negative examples, and 
    γ is a focusing parameter to put more emphasis on hard, misclassified examples.
    """
    def __init__(self, gamma=2.0, alpha=0.25, 
                 reduction=tf.keras.losses.Reduction.AUTO, 
                 name='binary_focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):  

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Calculate the binary cross-entropy loss
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Apply the focal loss adjustment
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Calculate the final focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        return tf.reduce_mean(focal_loss)
    
    
# ======================== Metrics ========================

class JaccardIndex(Metric):

    def __init__(self, name='jaccard_index', **kwargs):
        super(JaccardIndex, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='tp', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.union.assign_add(total - intersection)

    def result(self):
        jaccard_index = (self.intersection + EPSILON) / (self.union + EPSILON)

        return jaccard_index

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class DiceScore(Metric):
    """
    """
    def __init__(self, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.total.assign_add(total)

    def result(self):
        dice_coeff = (2. * self.intersection + EPSILON) / (self.total + EPSILON)

        return dice_coeff

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)



# ======================== Instances of Losses and Metrics ========================

# The following instances can be imported or the indivdual classes 
# can be imported to have the ability to adjust the parameters
        
# Losses
mse_loss = MSE_Loss()
dice_loss = DiceLoss()
jaccard_loss = JaccardLoss()
tversky_loss = TverskyLoss()
binary_focal_loss = BinaryFocalLoss()


# Metrics
jaccard_index = JaccardIndex()
dice_score = DiceScore()


