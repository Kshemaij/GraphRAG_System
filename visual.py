import matplotlib.pyplot as plt

# Scores from your evaluation
metrics = ['BLEU', 'ROUGE-1 F1', 'ROUGE-L F1']
rag_scores = [2.11, 0.52, 0.13]
grag_scores = [6.16, 0.83, 0.16]

x = range(len(metrics))

plt.figure(figsize=(8, 5))
plt.bar(x, rag_scores, width=0.4, label='RAG', align='center')
plt.bar([p + 0.4 for p in x], grag_scores, width=0.4, label='GraphRAG', align='center')
plt.xticks([p + 0.2 for p in x], metrics)
plt.ylabel("Score")
plt.title("RAG vs GraphRAG Evaluation Metrics")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
