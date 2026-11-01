import json
from datetime import datetime


def log_training_run(model_name, metrics: dict, log_filepath):
    run_log = {
        "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
        "model": model_name,
        "metrics": metrics,
    }

    with open(log_filepath, "a") as f:
        f.write(json.dumps(run_log) + "\n")