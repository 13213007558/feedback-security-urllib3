# Feedback task starting environment

Upstream: https://github.com/urllib3/urllib3
Upstream commit: `c5347bbd39203d68d7b0e3ffb83ced3e3b4b9109`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= --noconftest test/test_retry.py test/test_fields.py test/test_filepost.py test/test_collections.py -k 'not respect_retry_after_header_sleep'
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `127 passed, 27 deselected in 0.21s`
Whole-command wall time: 0.371 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
