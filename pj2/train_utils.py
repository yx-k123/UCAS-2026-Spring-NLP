import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainLogger:
    def __init__(
        self,
        result_dir: str,
        model_name: str,
        run_name: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.makedirs(result_dir, exist_ok=True)
        suffix = f"_{run_name}" if run_name else ""
        self.log_path = os.path.join(result_dir, f"train_{model_name}{suffix}.jsonl")
        self.summary_path = os.path.join(result_dir, f"train_{model_name}{suffix}_summary.json")
        self._steps = 0
        self._started = datetime.now().isoformat(timespec="seconds")
        self._config = config or {}

    def log_epoch(self, epoch: int, epochs: int, loss: float, lr: float) -> None:
        self._steps += 1
        payload = {
            "step": self._steps,
            "epoch": epoch,
            "epochs": epochs,
            "loss": round(loss, 8),
            "lr": lr,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(
            f"[train] epoch={epoch}/{epochs} loss={loss:.4f} lr={lr:.6g} "
            f"log={self.log_path}"
        )

    def save_summary(self, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "started_at": self._started,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "config": self._config,
        }
        if extra:
            payload.update(extra)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

