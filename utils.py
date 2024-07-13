import os
import tifffile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from tifffile import imread
import tensorflow as tf
from custom_datagen import DataGenerator
from keras.models import load_model
import numpy as np
import matplotlib.colorbar
scaler = MinMaxScaler()


def check_gpu_status():
    print("Num GPUs Available:", 
          len(tf.config.list_physical_devices('GPU')))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth 
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU(s),", 
                  len(logical_gpus), "Logical GPU(s)")
        except RuntimeError as e:
            print(e)


def scale_img(image, scaler=scaler):
    """
    Preprocesses an individual image. Scales image values between 0 and 1.
    Additionally, clips values to ensure they stay within the [0.0, 1.0] range.
    """
    reshaped_image = image.reshape(-1, image.shape[-1])
    scaled_image = scaler.fit_transform(reshaped_image).reshape(image.shape)
    # Clip values to ensure they are within [0.0, 1.0]
    clipped_image = np.clip(scaled_image, 0.0, 1.0)

    return clipped_image.astype('float32')


def load_test_datasets(image_directory=None, mask_directory=None,
                       calculate_pisi=False, calculate_ndvi=False,
                       img_extension='.tif', channels=None):
    """
    Load image and mask datasets from the specified directories, 
    normalize the data. Returns lists of arrays.
    """
    image_dataset = []
    mask_dataset = []

    if image_directory:
        images = os.listdir(image_directory)
        for image_name in tqdm(images, desc='Processing Images'):
            if image_name.endswith(f'{img_extension}'):
                image_path = os.path.join(image_directory, image_name)
                image = imread(image_path)

                if calculate_pisi:
                    PISI = DataGenerator.calculate_pisi_(image, channels)
                    image = np.concatenate((image, PISI), axis=-1) 

                if calculate_ndvi:
                    NDVI = DataGenerator.calculate_ndvi_(image, channels)
                    image = np.concatenate((image, NDVI), axis=-1)

                image = scale_img(image, scaler=scaler)
                image_dataset.append(image)

    if mask_directory:
        mask_files = os.listdir(mask_directory)
        for mask_name in tqdm(mask_files, desc='Processing Masks'):
            if mask_name.endswith(f'{img_extension}'): 
                mask_path = os.path.join(mask_directory, mask_name)

                mask = imread(mask_path)
                mask_dataset.append(mask)

    return image_dataset, mask_dataset


def predict_test_images(test_img, model_path, num_classes=1, threshold=0.5, 
                        augment_test=False, pred_type='threshold'):
    """
    Predicts and thresholds images using a specified deep learning model. 
    It optionally augments each image by rotating and mirroring to reduce 
    variance in the predictions and can return either probability maps or 
    thresholded predictions based on the 'pred_type' parameter.

    Parameters:
    - test_img (list of ndarray): List of test images to be predicted.
    - model_path (str): Path to the trained model.
    - threshold (float): Threshold for converting model predictions to 
      binary masks. Default is 0.5.
    - augment_test (bool, optional): If True, performs rotation and mirroring 
      on each image to reduce prediction variance. Default is False.
    - pred_type (str, optional): Type of prediction to return ('probability' or 'threshold'). 
      Default is 'threshold'.
    - save_pred_path (str, optional): Path to save the predictions. Default is None.
    - base_name (str, optional): Base name for saved prediction files. Default is None.

    Returns:
    - List of ndarray: Depending on 'pred_type', returns either thresholded predictions 
      or probability maps.
    
    Augmentation for Variance Reduction:
    - Each test image is rotated by 0, 90, 180, and 270 degrees, and then mirrored,  
      resulting in 8 different orientations.
    - The model predicts each orientation, and these predictions are then inversely 
      transformed to align with the original image orientation.
    - Predictions from all orientations are averaged to obtain a final prediction. 
      This averaging process reduces the variance in predictions.

    For an image I and its transformed versions T(I), the model prediction P(T(I)) 
    is transformed back to the original orientation, and then averaged:
    - Avg_P(I) = (1/N) Σ P(T(I)), where N is the number of transformations.
    - This average prediction is then thresholded to obtain a binary mask.
    """
    def rotate_and_mirror_image(img):
        rotations = [np.rot90(img, k=i) for i in range(4)]
        mirrored = [np.fliplr(x) for x in rotations]
        return rotations + mirrored

    def undo_transformations(predictions):
        untransformed_predictions = []
        for i, pred in enumerate(predictions):
            if i < 4: 
                untransformed_predictions.append(np.rot90(pred, k=-i))
            else:
                untransformed_predictions.append(np.rot90(np.fliplr(pred), 
                                                          k=-(i - 4)))

        return untransformed_predictions

    model = load_model(model_path, compile=False)


    predictions = []
    # counter = 0

    for img in tqdm(test_img, desc='Predicting'):
        if augment_test and num_classes==1:

            transformed_images = rotate_and_mirror_image(img)
            
            # Predict each transformed image
            transformed_predictions = []
            for transformed_img in transformed_images:
                transformed_img_batch = np.expand_dims(transformed_img, axis=0) 
                # transformed_pred_mask = model.predict(transformed_img_batch, verbose=0)
                transformed_pred_mask = model(transformed_img_batch, training=False)
                transformed_predictions.append(np.squeeze(transformed_pred_mask))
            
            # Undo the transformations
            aligned_predictions = undo_transformations(transformed_predictions)
            
            # Average the predictions
            avg_pred_mask = np.mean(aligned_predictions, axis=0)
        else:
            img_batch = np.expand_dims(img, axis=0)
            # avg_pred_mask = model.predict(img_batch, verbose=0)
            avg_pred_mask = model(img_batch, training=False)
            avg_pred_mask = np.squeeze(avg_pred_mask)

        if pred_type == 'argmax' and num_classes>1:
            avg_pred_mask_thresholded = np.argmax(avg_pred_mask, axis=2).astype('uint8')
            predictions.append(avg_pred_mask_thresholded)
        elif pred_type == 'threshold':
            # Threshold the prediction to create a binary mask                              
            avg_pred_mask_thresholded = (avg_pred_mask > threshold).astype('uint8')
            predictions.append(avg_pred_mask_thresholded)
        elif pred_type == 'probability':
            predictions.append(avg_pred_mask)

        # if save_pred_path and base_name:
        #     file_name = f"{base_name}{counter}.tif"
        #     full_path = os.path.join(save_pred_path, file_name)
        #     tifffile.imwrite(full_path, avg_pred_mask)
        #     counter += 1

    return predictions


def calculate_metrics_binary(test_mask, predictions, model_path):
    """
    Calculate IoU, precision, recall, F1 score, overall accuracy, and Dice score
    for the dataset and return the metrics dictionary and the confusion 
    matrix.
    """

    # Initialize count for TP, FP, FN, TN
    tp = np.uint64(0)
    fp = np.uint64(0)
    fn = np.uint64(0)
    tn = np.uint64(0)

    # Process each image to calculate the confusion matrix
    for mask, pred in zip(test_mask, predictions):
        mask_flat = mask.flatten()
        pred_flat = pred.flatten()

        # Convert to boolean arrays
        mask_flat = mask_flat.astype(bool)
        pred_flat = pred_flat.astype(bool)

        # Update TP, FP, FN, TN
        tp += np.sum(mask_flat & pred_flat)
        fp += np.sum(pred_flat & ~mask_flat)
        fn += np.sum(~pred_flat & mask_flat)
        tn += np.sum(~pred_flat & ~mask_flat)

    # Calculate IoU, precision, recall, F1 score, accuracy, and Dice score
    iou_score = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) 
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + fn + tn) 
    # dice_score = 2 * tp / (2 * tp + fp + fn)

    # Create a metrics dictionary
    metrics_dict = {
        'iou_score': iou_score,
        'precision': precision,
        'recall': recall, # sensitivity
        'f1_score': f1_score,
        'accuracy': accuracy,
        # 'dice_score': dice_score
    }

    # Print metric scores in a table format
    model_name = os.path.basename(model_path)
    metrics_table = (
        f"{'Model':<50} | {'IoU':<15} | {'Precision':<15} | {'Recall':<15} | "
        f"{'F1 Score':<15} | {'Accuracy':<15} \n"
        f"{'=' * 140}\n"
        f"{model_name:<50} | {metrics_dict['iou_score'] * 100:<15.4f} | "
        f"{metrics_dict['precision'] * 100:<15.4f} | "
        f"{metrics_dict['recall'] * 100:<15.4f} | "
        f"{metrics_dict['f1_score'] * 100:<15.4f} | "
        f"{metrics_dict['accuracy'] * 100:<15.4f} \n"
        f"{'=' * 140}"
    )
    print(metrics_table)

    # Construct the confusion matrix
    conf_matrix = np.array([[tp, fn], [fp, tn]])

    return metrics_dict, conf_matrix


def plot_conf_matrix_binary(cm, class_labels, title=None, cmap=plt.cm.Blues):
    plt.rcParams['font.family'] = 'Times New Roman'

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_ticks(np.arange(0, 1.01, 0.2)) 
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    ax.grid(False)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels, yticklabels=class_labels,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    fmt = '.4f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_normalized[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.show()


def plot_loss_and_metric(data, x, y_loss, y_val_loss, y_metric, y_val_metric, 
                        best_metric=None, best_loss=None, save_path=None):
    """

    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    sns.set_style("whitegrid")

    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = '#cccccc'  
    plt.rcParams['axes.axisbelow'] = True 
    plt.rcParams['grid.alpha'] = 0.5

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5)) 


    training_color = 'forestgreen'
    validation_color = 'crimson'


    ax1.plot(data[x], data[y_loss], label='Training Loss', linewidth=1.8, color=training_color)
    ax1.plot(data[x], data[y_val_loss], label='Validation Loss', linewidth=1.8, color=validation_color)


    if best_loss is not None:
        best_loss_value = data[y_val_loss].iloc[best_loss]
        ax1.scatter(best_loss, best_loss_value, color='red', s=60, zorder=5)
        ax1.axvline(x=best_loss, color='gray', linestyle='--', linewidth=0.8) 
    
    ax1.set_title('Loss over Epochs', fontsize=15, fontname='Times New Roman')
    ax1.set_xlabel('Epoch', fontsize=13, fontname='Times New Roman')
    ax1.set_ylabel('Loss', fontsize=13, fontname='Times New Roman')
    ax1.legend(loc='center right', fontsize=11, prop={'family': 'Times New Roman'})


    ax2.plot(data[x], data[y_metric], label='Training IoU', linewidth=1.8, color=training_color)
    ax2.plot(data[x], data[y_val_metric], label='Validation IoU', linewidth=1.8, color=validation_color)


    if best_metric is not None:
        best_metric_value = data[y_val_metric].iloc[best_metric]
        ax2.scatter(best_metric, best_metric_value, color='red', s=60, zorder=5)
        ax2.axvline(x=best_metric, color='gray', linestyle='--', linewidth=0.8) 
    
    ax2.set_title('IoU over Epochs', fontsize=15, fontname='Times New Roman')
    ax2.set_xlabel('Epoch', fontsize=13, fontname='Times New Roman')
    ax2.set_ylabel('IoU', fontsize=13, fontname='Times New Roman')
    ax2.legend(loc='center right', fontsize=11, prop={'family': 'Times New Roman'})

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def calculate_metrics_multiclass(test_masks, predictions, num_classes):

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)


    for mask, pred in zip(test_masks, predictions):
        mask_flat = mask.flatten()
        pred_flat = pred.flatten()
        conf_matrix += confusion_matrix(mask_flat, pred_flat, 
                                        labels=np.arange(num_classes)).astype(np.uint64)

    # Calculate metrics from the confusion matrix
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Calculate IoU for each class
    iou = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
    average_iou = np.mean(iou)

    # expected_agreement = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1))
    # total = np.sum(conf_matrix)
    # observed_agreement = np.trace(conf_matrix)
    # kappa = (total * observed_agreement - expected_agreement) / (total**2 - expected_agreement)
    # https://www.asprs.org/wp-content/uploads/pers/1995journal/apr/1995_apr_435-439.pdf
    
    expected_agreement = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (np.sum(conf_matrix) ** 2)
    # total = np.sum(conf_matrix)
    observed_agreement = overall_accuracy

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    # Division by zero
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    iou = np.nan_to_num(iou)


    metrics_dict = {
        'overall_accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score,
        'iou': iou,
        'average_iou': average_iou,
        'kappa': kappa,
    }

    # # Print metric scores in a table format
    # metrics_table = (
    #     f"{'Metric':<20} | {'Score':<15} \n"
    #     f"{'=' * 35}\n"
    #     f"{'Overall Accuracy':<20} | {metrics_dict['overall_accuracy']:<15.4f} \n" # * 100
    #     f"{'Precision':<20} | {str(metrics_dict['precision'])} \n"
    #     f"{'Recall':<20} | {str(metrics_dict['recall'])} \n"
    #     f"{'F1 Score':<20} | {str(metrics_dict['f1_score'])} \n"
    #     f"{'Kappa':<20} | {metrics_dict['kappa']:<15.4f} \n"
    #     f"{'=' * 35}"
    # )
    # print(metrics_table)

    return metrics_dict, conf_matrix


def plot_metrics_multi(metrics_dict, class_names):

    overall_accuracy = metrics_dict['overall_accuracy']
    precision = metrics_dict['precision']
    recall = metrics_dict['recall']
    f1_score = metrics_dict['f1_score']
    kappa = metrics_dict['kappa']
    iou = metrics_dict['iou']
    average_iou = metrics_dict['average_iou']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Overall Accuracy, Kappa, and Average IoU
    width = 0.25
    ax1.bar(['Overall Accuracy'], [overall_accuracy], color='b', width=width, label='Overall Accuracy')
    ax1.bar(['Kappa'], [kappa], color='r', width=width, label='Kappa')
    ax1.bar(['Average IoU'], [average_iou], color='g', width=width, label='Average IoU')
    ax1.set_ylim(0, 1)
    # ax1.set_ylabel('Score')
    ax1.set_title('Overall Accuracy, Kappa, and Average IoU', pad=20)
    ax1.legend()

    # Precision, Recall, F1 Score
    x = np.arange(len(class_names))
    width = 0.2

    ax2.bar(x - width, precision, width, label='Precision')
    ax2.bar(x, recall, width, label='Recall')
    ax2.bar(x + width, f1_score, width, label='F1 Score')

    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.set_ylim(0, 1)
    # ax2.set_ylabel('Score')
    ax2.set_title('Precision, Recall, F1 Score', pad=20)
    ax2.legend()

    # IoU for each class
    ax3.bar(x, iou, width=0.2, color='c', label='IoU')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names)
    ax3.set_ylim(0, 1)
    # ax3.set_ylabel('Score')
    ax3.set_title('IoU for Each Class', pad=20)
    ax3.legend()

    plt.tight_layout(pad=2.0)
    plt.show()


def plot_conf_matrix_multi(conf_matrix, class_labels, normalize=False):
    """
    Plot the confusion matrix using a heatmap.

    Parameters:
    conf_matrix (numpy.ndarray): The confusion matrix.
    class_labels (list): List of class labels.
    normalize (bool): Whether to normalize the values in the confusion matrix.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.set_theme(font="Times New Roman")
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Reds', 
                xticklabels=class_labels, yticklabels=class_labels, cbar_kws={'label': 'Scale'})
    
    plt.xlabel('Predicted Label', fontsize=14, labelpad=20, fontname='Times New Roman')
    plt.ylabel('True Label', fontsize=14, labelpad=20, fontname='Times New Roman')
    plt.xticks(fontsize=12, rotation=45, ha='right', fontname='Times New Roman')
    plt.yticks(fontsize=12, rotation=0, fontname='Times New Roman')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix', 
              fontsize=16, fontname='Times New Roman', pad=30)
    plt.show()


def create_colored_mask(ground_truth, prediction):

    colored_mask = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    
    # False positives (prediction is 1, ground truth is 0) - Blue
    colored_mask[(ground_truth == 0) & (prediction == 1)] = [0, 0, 255]
    
    # False negatives (prediction is 0, ground truth is 1) - Red
    colored_mask[(ground_truth == 1) & (prediction == 0)] = [255, 0, 0]
    
    # True positives (prediction is 1, ground truth is 1) - White
    colored_mask[(ground_truth == 1) & (prediction == 1)] = [255, 255, 255]
    
    # True negatives (prediction is 0, ground truth is 0) - Black
    colored_mask[(ground_truth == 0) & (prediction == 0)] = [0, 0, 0]
    
    return colored_mask



