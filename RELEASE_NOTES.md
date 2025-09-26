# Release Notes

## Unreleased

- Added incremental JSON streaming with live token callbacks and configurable invalid-path handling.
- Documented streaming workflow and expanded structured generation examples.
- Broadened evaluation coverage with stub suites and additional streaming-focused unit tests.
- Verified structured adherence end-to-end on Gemma 2B (MLX backend).
- Added a `make publish` convenience target to bump the patch version, push the git tag, push to PyPI, and now auto-create the GitHub release entry.
- Adjusted packaging metadata (license handling) to keep PyPI uploads compatible with current validators.
