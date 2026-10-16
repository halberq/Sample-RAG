

from eval_rag import evaluate, split_questions
from main_script import retrieve
tune, test = split_questions(questions)
recall, mrr, ranks = evaluate(retrieve, tune)
print(f"baseline tuning: recall@3={recall:.2f}  MRR={mrr:.2f}")