from __future__ import annotations

from absl import flags

if not flags.FLAGS.is_parsed():
    flags.FLAGS(["test_script"], known_only=True)
