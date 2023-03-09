import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_recall
from torchmetrics.functional.classification import multiclass_precision, multiclass_f1_score
from torchmetrics.functional.classification import multiclass_average_precision, multiclass_confusion_matrix


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