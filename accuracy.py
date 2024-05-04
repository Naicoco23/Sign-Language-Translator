from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def calculate_metrics_multiclass(ground_truth, predictions):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


df = pd.read_csv("C:\\Users\\Nico\\OneDrive - Bina Nusantara\\Documents\\BINUS\\SMT 4\\Research Methodology in Computer Science\\Research\\dataset.csv")  

metrics = calculate_metrics_multiclass(df['ground_truth'], df['predicted'])

print(metrics)
