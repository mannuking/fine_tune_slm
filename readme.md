# Fine-tuning Small Language Model (SLM)

This project contains scripts and resources for fine-tuning a Small Language Model (SLM) like Phi-3 using a custom dataset extracted from a PDF book.

## Project Structure:

- `readme.txt`: This file (project description and instructions).
- `requirements.txt`: Python dependencies for the project.
- `data_prep.py`: Script to prepare the JSON data for fine-tuning.
- `fine_tune.py`: Script to perform the fine-tuning process.
- `output.json`: (This file will be used as input data, but is not created by these scripts)
- `fine-tuned-model/`: Directory to save the fine-tuned model.
- `results/`: Directory to store training results and logs.

## Instructions:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare Data:**
    Run `data_prep.py` to format your `output.json` data.
    ```bash
    python data_prep.py
    ```
3.  **Fine-tune Model:**
    Run `fine_tune.py` to start the fine-tuning process.
    ```bash
    python fine_tune.py
    ```

## Notes:

-   This project is set up to use the Phi-3-mini-4k-instruct model as the base model.
-   Adjust hyperparameters and configurations in `fine_tune.py` as needed.
-   Ensure you have a suitable GPU environment set up for fine-tuning (e.g., RTX 3070).
