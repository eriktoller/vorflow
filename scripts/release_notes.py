"""Write the GitHub release notes for a tag from its CHANGELOG.md section.

The section under ``## [X.Y.Z]`` is copied with its hard-wrapped lines joined,
because GitHub keeps single line breaks in release notes. An install line and
the section's compare link (from the reference links at the end of the
changelog) are appended. A tag without a changelog section is an error, so the
release workflow fails before anything is uploaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BLOCK_STARTS = ("- ", "* ", "#", "|", "```")


def changelog_section(changelog: str, version: str) -> str:
    """Return the body of the ``## [version]`` section, without its heading."""
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)"
    match = re.search(pattern, changelog, re.DOTALL | re.MULTILINE)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    return match.group(1).strip()


def compare_link(changelog: str, version: str) -> str | None:
    """Return the URL of the ``[version]:`` reference link, if there is one."""
    match = re.search(rf"^\[{re.escape(version)}\]: (\S+)", changelog, re.MULTILINE)
    return None if match is None else match.group(1)


def unwrap(text: str) -> str:
    """Join hard-wrapped paragraph and list-item lines, leaving code blocks alone."""
    lines = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        continues = (
            not in_code
            and stripped
            and lines
            and lines[-1].strip()
            and not lines[-1].lstrip().startswith(("#", "|", "```"))
            and not stripped.startswith(BLOCK_STARTS)
        )
        if continues:
            lines[-1] = f"{lines[-1].rstrip()} {stripped}"
        else:
            lines.append(line)
    return "\n".join(lines)


def release_notes(changelog: str, version: str) -> str:
    """Return the release notes for ``version``."""
    notes = [
        unwrap(changelog_section(changelog, version)),
        f"**Install:** `pip install vorflow=={version}`",
    ]
    link = compare_link(changelog, version)
    if link is not None:
        notes.append(f"**Full changelog:** {link}")
    return "\n\n".join(notes) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="release tag, e.g. v0.3.0")
    parser.add_argument("output", type=Path, help="file to write the notes to")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=ROOT / "CHANGELOG.md",
        help="changelog to read the section from",
    )
    args = parser.parse_args()
    version = args.tag.removeprefix("v")
    notes = release_notes(args.changelog.read_text(encoding="utf-8"), version)
    args.output.write_text(notes, encoding="utf-8")
    print(f"Wrote release notes for {version} to {args.output}")


if __name__ == "__main__":
    main()
