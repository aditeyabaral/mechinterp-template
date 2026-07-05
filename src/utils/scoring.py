"""How an answer is scored correct -- shared by main.py (ablation accuracy) and lasso.py (row filtering).

`is_correct` lives here, in one place, so the two callers can't drift apart: main.py uses it to score
baseline vs ablated accuracy, and lasso.py's keep_row can import the SAME check to keep only the
prompts the model answered correctly. It knows nothing about any particular task -- it just compares
the answer inference.py captured against a ground-truth "answer" your prompts carry in their metadata.
"""


def is_correct(row: dict) -> bool:
    """Whether the model's generated answer matches the ground truth for one result row.

    Used to score intervention runs (baseline vs ablated accuracy) and, optionally, to keep only
    correctly-answered rows in lasso.py. The default compares the answer token inference.py captured
    (row["result"]["answer"]["token"]) against an "answer" stored in that prompt's metadata (see
    PromptDataset.generate_prompts in src/utils/dataset.py). It therefore only does something useful
    if your prompts carry a ground-truth "answer" in their metadata; otherwise every row counts as
    wrong and the accuracies come out 0.

    TODO (optional): adjust this if "correct" means something else for your task, or if your
    metadata stores the truth under a different key.
    """
    return row["result"]["answer"]["token"] == row["metadata"].get("answer")
