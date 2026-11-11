"""Test isolation: no test may read or write the real journal, profile or cache.

config.DATA_DIR is resolved when velveteentrade.config is first imported, and
Settings() defaults its db/profile/cache paths from it. Point it at a fresh
temporary directory BEFORE anything imports the package (pytest imports this
file first). Without this, tests that build Settings() without overriding
profile_path read the deployment's real profile.yaml (found 24 Oct 2026).
"""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="velveteen-tests-")
