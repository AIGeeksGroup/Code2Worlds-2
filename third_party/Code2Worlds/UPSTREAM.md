# Code2Worlds source snapshot

- Upstream: https://github.com/AIGeeksGroup/Code2Worlds
- Base commit: `03d7edb9242b4095bfb9b3315b1fd67a20d433a1`
- Included: upstream agent, library, scripts, tests, requirements, README, and overview image.
- Excluded: generated `output/`, nested Infinigen checkout, and Git history. Infinigen is installed separately.

This snapshot includes three local integration changes: the Infinigen `src/` configuration path in `agent/scene_stream/realizer.py`, resilient JSON extraction in `agent/scene_stream/resolver.py`, and an optional `CODE2WORLDS_PYTHON` interpreter override in `scripts/scene_stream.sh`. The release also reads LLM credentials from environment variables instead of overriding them with empty strings.

The upstream snapshot does not include a top-level license file. This copy preserves that status and does not assign a new license.
