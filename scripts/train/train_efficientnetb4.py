#!/usr/bin/env python
"""Train EfficientNetB4 model on corn leaf disease dataset."""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "train.py", "--config", "configs/efficientnetb4.yaml"],
        cwd=".",
    )
    sys.exit(result.returncode)