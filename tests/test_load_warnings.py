# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.

"""The upstream torch_dtype rename notice stays out of a stranger's terminal.

Found by the 0.0.1 post-publish pass: newer transformers prints a deprecation notice for
the torch_dtype load kwarg, which is cosmetic noise to someone running the quickstart. The
mute is message-scoped and context-scoped, filters both channels the notice can arrive on
(warnings module and the transformers logger), and passes everything else through. These
tests run torch-free: they exercise the filter, not a model load.
"""

from __future__ import annotations

import logging
import warnings

from gaige.detectors.base import mute_torch_dtype_deprecation

MSG = "`torch_dtype` is deprecated! Use `dtype` instead!"


def test_warnings_channel_muted_inside_context_only():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with mute_torch_dtype_deprecation():
            warnings.warn(MSG, FutureWarning, stacklevel=1)
            warnings.warn("an unrelated caution", UserWarning, stacklevel=1)
        warnings.warn(MSG, FutureWarning, stacklevel=1)  # after exit: passes again
    messages = [str(w.message) for w in seen]
    assert "an unrelated caution" in messages
    assert messages.count(MSG) == 1  # the in-context one was dropped, the after-exit one kept


def test_logger_channel_muted_at_source_and_restored():
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append
    tf_logger = logging.getLogger("transformers")
    module_logger = logging.getLogger("transformers.modeling_utils")
    tf_logger.addHandler(handler)
    try:
        with mute_torch_dtype_deprecation():
            module_logger.warning(MSG)
            module_logger.warning("an unrelated notice")
        module_logger.warning(MSG)  # after exit: passes again
    finally:
        tf_logger.removeHandler(handler)
    messages = [r.getMessage() for r in captured]
    assert "an unrelated notice" in messages
    assert messages.count(MSG) == 1
