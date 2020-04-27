# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScoreSeg(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        # 2D matrix counting number of pixels for each pred_class-label_class
        self.confusion_matrix = np.zeros((n_classes, n_classes)) 

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) # boolean mask, ignoring values outside of valid range
        hist = np.bincount( # count number of pixels for each pred_class-label_class combination
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int), minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_image_score(self, label_trues, label_preds):

        hist = self._fast_hist(label_trues.flatten(), label_preds.flatten(), self.n_classes)
        acc = np.diag(hist).sum() / hist.sum()

        return acc

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        
        # diagonal of pred-label class matrix is correct prediction
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1) # true positive / true positive + false positive
        acc_cls = np.nanmean(acc_cls)

        # diag is intersection, sum across horizontal and vertical is union
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc, # pixel accuracy
                "Mean Acc : \t": acc_cls, # mean class precision
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu, # mean of IoU
            },
            {
                "cls": cls_iu
            }
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScoreClassifier(runningScoreSeg):

    def __init__(self, n_classes):
        super().__init__(n_classes)

    def get_scores(self):
        
        hist = self.confusion_matrix

        precision = np.diag(hist) / hist.sum(axis=0) # axis0 = prediction
        precision[np.isnan(precision)] = 0
        recall = np.diag(hist) / hist.sum(axis=1) # axis1 = labels
        f1 = 2 * ((precision*recall) / (precision + recall))
        f1[np.isnan(f1)] = 0
        support = hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist)

        accuracy = np.diag(hist).sum() / hist.sum()
        avg_precision = (precision * support).sum() / support.sum()
        avg_recall = (recall * support).sum() / support.sum()
        avg_f1 = (f1 * support).sum() / support.sum()

        return (
            {
                "Overall Accuracy: \t": accuracy,
                "Average precision: \t": avg_precision,
                "Average recall: \t": avg_recall,
                "Average f1: \t": avg_f1
            },
            {
                "precision": dict(zip(range(self.n_classes), precision)),
                "recall": dict(zip(range(self.n_classes), recall)),
                "f1": dict(zip(range(self.n_classes), f1))
            }
        )


class averageMeter(object):
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
