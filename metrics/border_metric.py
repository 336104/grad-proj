import numpy as np


class BorderMetric:
    @staticmethod
    def decode_labels(predictions, labels):
        predictions.append(-100)
        labels.append(-100)
        e_pred = set()
        e_ref = set()
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                if labels[i - 1] == 0 or labels[i - 1] == 2:
                    if i - start != 1:
                        e_ref.add((start, i))
                if labels[i] == 0 or labels[i] == 2:
                    start = i
        start = 0
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i - 1]:
                if labels[i] != -100 and (
                    predictions[i - 1] == 0 or predictions[i - 1] == 2
                ):
                    if i - start != 1:
                        e_pred.add((start, i))
                if labels[i] == 0 or labels[i] == 2:
                    start = i
        return e_pred, e_ref

    @staticmethod
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        tp, fn, fp = [1e-6] * 3
        for prediction, label in zip(predictions, labels):
            e_pred, e_ref = BorderMetric.decode_labels(
                prediction.tolist(), label.tolist()
            )
            tp += len(e_pred & e_ref)
            fn += len(e_ref - e_pred)
            fp += len(e_pred - e_ref)
        precision = tp / (fp + tp)
        recall = tp / (fn + tp)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}
