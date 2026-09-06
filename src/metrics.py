import numpy as np
from src.utils import restore_lemma

def make_compute_lemmatization_metrics(gold_dataset):
    def compute(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)

        correct, total, seq_correct = 0, 0, 0

        for sent_idx, (pred_sent, mask_sent) in enumerate(zip(pred_ids, labels)):
            gold_labels_str = gold_dataset.raw_labels[sent_idx]
            gold_words_str = gold_dataset.raw_words[sent_idx]

            word_pos = 0
            is_correct = True

            for p, g in zip(pred_sent, mask_sent):
                if g == -100:
                    continue

                pred_str = gold_dataset.tags_[p]
                gold_str = gold_labels_str[word_pos]
                current_word = gold_words_str[word_pos]

                predicted_lemma = restore_lemma(current_word, pred_str)
                gold_lemma = restore_lemma(current_word, gold_str)

                if predicted_lemma == gold_lemma:
                    correct += 1
                else:
                    is_correct = False

                total += 1
                word_pos += 1

            seq_correct += int(is_correct)

        return {
            "Lemma_Accuracy": 100 * correct / total if total > 0 else 0,
            "Sentence_Lemma_Accuracy": 100 * seq_correct / len(labels) if len(labels) > 0 else 0
        }

    return compute

def make_compute_metrics(gold_dataset):
    """Замыкание для подсчета Accuracy и Sentence Accuracy."""
    def compute(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        correct, total, seq_correct = 0, 0, 0
        
        for sent_idx, (pred_sent, mask_sent) in enumerate(zip(pred_ids, labels)):
            gold_labels_str = gold_dataset.raw_labels[sent_idx]
            word_pos = 0
            is_correct = True
            
            for p, g in zip(pred_sent, mask_sent):
                if g == -100:
                    continue
                pred_str = gold_dataset.tags_[p]
                gold_str = gold_labels_str[word_pos]
                
                if pred_str == gold_str:
                    correct += 1
                else:
                    is_correct = False
                    
                total += 1
                word_pos += 1
                
            seq_correct += int(is_correct)
            
        return {
            "Accuracy": 100 * correct / total if total > 0 else 0.0,
            "Sentence accuracy": 100 * seq_correct / len(labels) if len(labels) > 0 else 0.0
        }
    return compute

def make_compute_multitask_metrics(gold_dataset, id2label_dict):
    def compute(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        task_names = list(preds.keys())
        any_task = task_names[0]
        batch_size, seq_len = preds[any_task].shape[0], preds[any_task].shape[1]

        correct, total, seq_correct = 0, 0, 0
        task_correct = {t: 0 for t in task_names}
        task_total = {t: 0 for t in task_names}

        for sent_idx in range(batch_size):
            gold_labels_str = gold_dataset.raw_labels[sent_idx]
            word_pos = 0
            is_correct_seq = True

            for t in range(seq_len):
                if labels[any_task][sent_idx, t] == -100:
                    continue

                pred_features = {}
                for task in task_names:
                    task_logits = preds[task][sent_idx, t]
                    pred_id = np.argmax(task_logits)
                    mapping = id2label_dict.get(task, [])

                    pred_val = str(mapping[pred_id]) if 0 <= pred_id < len(mapping) else str(pred_id)
                    pred_features[task] = pred_val

                    gold_task_id = labels[task][sent_idx, t]
                    if gold_task_id != -100:
                        task_total[task] += 1
                        if pred_id == gold_task_id:
                            task_correct[task] += 1

                pos = pred_features.get("POS", "None")
                feats_list = sorted([f"{k}={v}" for k, v in pred_features.items() if k != "POS" and v != "None"])
                pred_full_tag = f"{pos},{'|'.join(feats_list)}" if feats_list else pos

                gold_full_tag = gold_labels_str[word_pos]
                if pred_full_tag == gold_full_tag:
                    correct += 1
                else:
                    is_correct_seq = False

                total += 1
                word_pos += 1

            seq_correct += int(is_correct_seq)

        metrics = {
            "Full_Tag_Accuracy": 100 * correct / total if total > 0 else 0,
            "Sentence_Accuracy": 100 * seq_correct / batch_size if batch_size > 0 else 0
        }
        for task in task_names:
            if task_total[task] > 0:
                metrics[f"{task}_acc"] = 100 * task_correct[task] / task_total[task]

        return metrics

    return compute
