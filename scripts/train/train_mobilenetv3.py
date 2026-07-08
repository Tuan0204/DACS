#!/usr/bin/env python
"""Train MobileNetV3-Small model on corn leaf disease dataset."""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "train.py", "--config", "configs/mobilenetv3.yaml"],
        cwd=".",
    )
    sys.exit(result.returncode)
