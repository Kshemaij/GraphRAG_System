import pandas as pd
import evaluate
import matplotlib.pyplot as plt

# Load evaluation metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Load CSV file
file_path = "C:/Users/91990/Downloads/Masters Thesis/Code/Final_evaluation.csv"
df = pd.read_csv(file_path)

# Ensure columns exist and convert to string
df['Reference'] = df['Reference'].astype(str)
df['GRAG_answers'] = df['GRAG_answers'].astype(str)
df['RAG_answer'] = df['RAG_answer'].astype(str)

# Function to compute metrics per row
def compute_metrics(pred, ref):
    bleu = bleu_metric.compute(predictions=[pred], references=[[ref]])["bleu"]
    rouge = rouge_metric.compute(predictions=[pred], references=[ref])
    return bleu, rouge["rouge1"], rouge["rougeL"]

# Apply metrics row by row for RAG
df["BLEU_RAG"], df["ROUGE1_RAG"], df["ROUGEL_RAG"] = zip(*df.apply(lambda row: compute_metrics(row["RAG_answer"], row["Reference"]), axis=1))

# Apply metrics row by row for GraphRAG
df["BLEU_GRAG"], df["ROUGE1_GRAG"], df["ROUGEL_GRAG"] = zip(*df.apply(lambda row: compute_metrics(row["GRAG_answers"], row["Reference"]), axis=1))

# 📊 Boxplots
metrics = ["BLEU", "ROUGE1", "ROUGEL"]

for metric in metrics:
    plt.figure(figsize=(6,5))
    df[[f"{metric}_RAG", f"{metric}_GRAG"]].boxplot()
    plt.title(f"{metric} Score Comparison (RAG vs GraphRAG)")
    plt.ylabel("Score")
    plt.show()
