import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_recall
from torchmetrics.functional.classification import multiclass_precision, multiclass_f1_score
from torchmetrics.functional.classification import multiclass_average_precision, multiclass_confusion_matrix
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_stats(preds, target):
    # Because there are many negative samples for each class, ROC and AUROC scores are not used.
    """Calculate statistics for classification task
    Args:
      output: 2d tensor, (samples_num, classes_num)
      target: 2d tensor, (samples_num, classes_num)
    Returns:
      stats: dict, statistics for classification task
    """

    classes_num = target.shape[-1]
    stats = {}

    target = torch.argmax(target, dim=1)

    # Accuracy
    stats['weighted_acc'] = multiclass_accuracy(preds, target, num_classes=classes_num, average='weighted')
    stats['macro_acc'] = multiclass_accuracy(preds, target, num_classes=classes_num, average='macro')
    stats['micro_acc'] = multiclass_accuracy(preds, target, num_classes=classes_num, average='micro')

    # Recall
    stats['weighted_recall'] = multiclass_recall(preds, target, num_classes=classes_num, average='weighted')
    stats['macro_recall'] = multiclass_recall(preds, target, num_classes=classes_num, average='macro')
    stats['micro_recall'] = multiclass_recall(preds, target, num_classes=classes_num, average='micro')

    # Precision
    stats['weighted_precision'] = multiclass_precision(preds, target, num_classes=classes_num, average='weighted')
    stats['macro_precision'] = multiclass_precision(preds, target, num_classes=classes_num, average='macro')
    stats['micro_precision'] = multiclass_precision(preds, target, num_classes=classes_num, average='micro')

    # F1
    stats['weighted_f1'] = multiclass_f1_score(preds, target, num_classes=classes_num, average='weighted')
    stats['macro_f1'] = multiclass_f1_score(preds, target, num_classes=classes_num, average='macro')
    stats['micro_f1'] = multiclass_f1_score(preds, target, num_classes=classes_num, average='micro')

    # Average precision
    stats['weighted_avg_precision'] = multiclass_average_precision(preds, target, num_classes=classes_num, average='weighted')
    stats['macro_avg_precision'] = multiclass_average_precision(preds, target, num_classes=classes_num, average='macro')

    # Confusion matrix
    stats['confusion_matrix'] = multiclass_confusion_matrix(preds, target, num_classes=classes_num)

    # Class-wise recall for analysis
    stats['class_wise_recall'] = multiclass_recall(preds, target, num_classes=classes_num, average=None)

    return stats


def worse_k_bar_fig(stats, label, k, dataset, split):
    """Plot the bar chart of the k worst classes in terms of recall
    Args:
      stats: dict, statistics for classification task
      k: int, number of classes to be plotted
      dataset: dataset object
      split: str, 'val' or 'test'
    Returns:
      fig: figure object
    """

    # Get the k worst classes in terms of recall
    class_wise_recall = stats['class_wise_recall']
    sorted_recall, _indices = torch.sort(class_wise_recall, descending=False)
    k_indices = _indices[:k]
    k_recall = sorted_recall[:k]

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # show Chinese
    plt.rcParams["axes.unicode_minus"] = False  # show minus

    k_label = [label[i] for i in k_indices]

    counts = {}
    # Loop through each element in the list
    for elem in dataset.label:
        # If the element is not already in the dictionary, add it with a count of 1
        if elem not in counts:
            counts[elem] = 1
        # If the element is already in the dictionary, increment its count
        else:
            counts[elem] += 1
    k_label = [f'{ label[i] }({ counts[label[i]] })' for i in k_indices]

    # Create the bar graph with a custom figure size
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(range(k), k_recall)
    ax.set_xticks(range(k))
    ax.set_xticklabels(k_label, rotation=45)
    ax.set_xlabel('Class')
    ax.set_ylabel('Recall')
    ax.set_title(f'50 Worst Classes on {split} Set')

    return fig