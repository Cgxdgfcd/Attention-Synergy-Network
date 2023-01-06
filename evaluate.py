import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import AS_Net
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import tensorflow as tf

np.set_printoptions(threshold=np.inf)


####################################  Load Data #####################################
te_data = np.load('data_val.npy').astype(np.float32)
te_mask = np.load('mask_val.npy').astype(np.float32)
te_mask = np.expand_dims(te_mask, axis=-1)

te_data = tf.image.adjust_gamma(te_data / 255., gamma=1.6)
te_mask /= 255.

print('ISIC18 Dataset loaded')

model = AS_Net()
model.load_weights('./checkpoint_best/weights_best.hdf5')
predictions = model.predict(te_data, batch_size=8, verbose=1)

y_scores = predictions.reshape(
    predictions.shape[0] * predictions.shape[1] * predictions.shape[2] * predictions.shape[3], 1)

y_true = te_mask.reshape(te_mask.shape[0] * te_mask.shape[1] * te_mask.shape[2] * te_mask.shape[3], 1)

y_scores = np.where(y_scores > 0.5, 1, 0)
y_true = np.where(y_true > 0.5, 1, 0)

output_folder = 'output/'

# Area under the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print("\nArea under the ROC curve: " + str(AUC_ROC))
roc_curve = plt.figure()
plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(output_folder + "ROC.png")

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision, recall)
print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(output_folder + "Precision_recall.png")

# Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i] >= threshold_confusion:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion)) != 0:
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))
specificity = 0
if float(confusion[0, 0] + confusion[0, 1]) != 0:
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
print("Specificity: " + str(specificity))
sensitivity = 0
if float(confusion[1, 1] + confusion[1, 0]) != 0:
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
print("Sensitivity: " + str(sensitivity))
precision = 0
if float(confusion[1, 1] + confusion[0, 1]) != 0:
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
print("Precision: " + str(precision))

# Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print("\nJaccard similarity score: " + str(jaccard_index))

# F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " + str(F1_score))

# Save the results
file_perf = open(output_folder + 'performances.txt', 'w')
file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                + "\nJaccard similarity score: " + str(jaccard_index)
                + "\nF1 score (F-measure): " + str(F1_score)
                + "\n\nConfusion matrix:"
                + str(confusion)
                + "\nACCURACY: " + str(accuracy)
                + "\nSENSITIVITY: " + str(sensitivity)
                + "\nSPECIFICITY: " + str(specificity)
                + "\nPRECISION: " + str(precision)
                )
file_perf.close()

# Save 10 results with error rate lower than threshold
threshold = 300
predictions = np.where(predictions > 0.5, 1, 0)
te_mask = np.where(te_mask > 0.5, 1, 0)
good_prediction = np.zeros([predictions.shape[0], 1], np.uint8)
id_m = 0
for idx in range(predictions.shape[0]):
    esti_sample = predictions[idx]
    true_sample = te_mask[idx]
    esti_sample = esti_sample.reshape(esti_sample.shape[0] * esti_sample.shape[1] * esti_sample.shape[2], 1)
    true_sample = true_sample.reshape(true_sample.shape[0] * true_sample.shape[1] * true_sample.shape[2], 1)
    er = 0
    for idy in range(true_sample.shape[0]):
        if esti_sample[idy] != true_sample[idy]:
            er = er + 1
    if er < threshold:
        good_prediction[id_m] = idx
        id_m += 1

fig, ax = plt.subplots(10, 3, figsize=[15, 15])

for idx in range(10):
    ax[idx, 0].imshow(te_data[good_prediction[idx, 0]])
    ax[idx, 1].imshow(np.squeeze(te_mask[good_prediction[idx, 0]]), cmap='gray')
    ax[idx, 2].imshow(np.squeeze(predictions[good_prediction[idx, 0]]), cmap='gray')

plt.savefig(output_folder + 'sample_results.png')
