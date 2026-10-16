import importlib.util
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_notes.py"
SPEC = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_notes)

# A synthetic changelog, so these tests exercise the extraction rather than
# restating vorflow's release history.
CHANGELOG = """# Changelog

## [Unreleased]

## [2.0.0] - 2030-01-02

A summary that is
wrapped over two lines.

### Added

- `thing()` does a
  thing ([#1](https://example.org/1)).
- Another item.
  - A nested item.

```bash
pip install vorflow
pip check
```

| a | b |
|---|---|

## [1.0.0] - 2029-01-01

Old notes.

[Unreleased]: https://example.org/compare/v2.0.0...HEAD
[2.0.0]: https://example.org/compare/v1.0.0...v2.0.0
"""


def test_release_notes_unwrap_paragraphs_and_list_items():
    notes = release_notes.release_notes(CHANGELOG, "2.0.0")
    assert "A summary that is wrapped over two lines." in notes
    assert "- `thing()` does a thing ([#1](https://example.org/1))." in notes
    assert "- Another item.\n  - A nested item." in notes


def test_release_notes_keep_code_blocks_tables_and_headings():
    notes = release_notes.release_notes(CHANGELOG, "2.0.0")
    assert "```bash\npip install vorflow\npip check\n```" in notes
    assert "| a | b |\n|---|---|" in notes
    assert "### Added\n\n- `thing()`" in notes


def test_release_notes_stop_at_the_next_section_and_add_links():
    notes = release_notes.release_notes(CHANGELOG, "2.0.0")
    assert "Old notes." not in notes
    assert "## [" not in notes
    assert "**Install:** `pip install vorflow==2.0.0`" in notes
    assert notes.endswith(
        "**Full changelog:** https://example.org/compare/v1.0.0...v2.0.0\n"
    )


def test_last_section_stops_before_reference_links():
    notes = release_notes.release_notes(CHANGELOG, "1.0.0")
    assert notes.startswith("Old notes.\n\n**Install:**")
    assert "**Full changelog:**" not in notes


def test_missing_section_is_an_error():
    with pytest.raises(ValueError, match="no section for 3.0.0"):
        release_notes.release_notes(CHANGELOG, "3.0.0")


def test_project_version_has_a_changelog_section():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_notes.changelog_section(changelog, version)
