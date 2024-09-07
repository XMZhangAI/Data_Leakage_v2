from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

arch = "CodeLlama-7b-hf"
model_dir = "/home/jiangxue/LLMs/"
dataset = load_from_disk("datasets/original_data_contamination_detection_dataset")['test']

tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{arch}", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


def evaluate_classification(y_true, y_pred, y_pred_prob=None):
    metrics = {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    if y_pred_prob is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_prob)
    return metrics


def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def generate_ngrams_token(text, n):
    text = tokenizer.tokenize(text)
    if len(text) < n:
        return [" ".join(text)]
    ngrams = [" ".join(text[i:i+n]) for i in range(len(text)-n+1)]
    return ngrams

def generate_ngrams_char(text, n):
    if len(text) < n:
        return [text]
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ngrams

def detect_duplicates(text1, text2, level="token_level", n=13):
    """
    Detect duplicate n-grams between two texts.

    :param text1: The first input text string.
    :param text2: The second input text string.
    :param n: The number of characters in each n-gram.
    :return: A set of duplicate n-grams found in both texts.
    """

    if level=="token_level":
        ngrams1 = set(generate_ngrams_token(text1, n))
        ngrams2 = set(generate_ngrams_token(text2, n))
    else:
        ngrams1 = set(generate_ngrams_char(text1, n))
        ngrams2 = set(generate_ngrams_char(text2, n))

    # Find the intersection of the two sets
    duplicates = ngrams1.intersection(ngrams2)
    if len(duplicates) > 0:
        return 1
    else:
        return 0

def n_gram_detection(dataset, level="token_level", n=13):
    y_true = []
    y_pred = []
    for task in dataset:
        duplicates = detect_duplicates(task['completion'], task['leaked_data'], level, n)
        y_true.append(task['label'])
        y_pred.append(duplicates)
    return y_true, y_pred


if __name__ == "__main__":
    y_true, y_pred =  n_gram_detection(dataset, level="char_level", n=13)
    metrics = evaluate_classification(y_true, y_pred, y_pred)

    print("ACC:", metrics['Accuracy'])  
    print("Precision:", metrics['Precision']) 
    print("Recall:", metrics['Recall'])
    print("F1:", metrics['F1 Score'])
    print("AUC:", metrics['AUC'])

    
    
    