#!/usr/bin/env python3
"""Safely preview, back up, remove, or restore source-code comments.

Run from any working directory; the project root is the parent of ``scripts``.
The default operation is a dry run. JavaScript is intentionally skipped unless
a safe source-preserving parser is added; this script never guesses with regex.
"""
from __future__ import annotations

import argparse
import codecs
import fnmatch
import hashlib
import html.parser
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKUP_ROOT = Path(tempfile.gettempdir()) / "arecanut_comment_cleanup_backups"
PYTHON_EXTENSIONS = {".py"}
HTML_EXTENSIONS = {".html", ".htm"}
CSS_EXTENSIONS = {".css"}
SQL_EXTENSIONS = {".sql"}
JS_EXTENSIONS = {".js", ".mjs", ".cjs"}
SUPPORTED_EXTENSIONS = PYTHON_EXTENSIONS | HTML_EXTENSIONS | CSS_EXTENSIONS | SQL_EXTENSIONS | JS_EXTENSIONS

PROTECTED_DIRS = {
    ".git", ".venv", "venv", "env", "virtualenv", "node_modules", "__pycache__", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dataset", "datasets", "verification_dataset",
    "saved_models", "uploads", "heatmaps", "reports",
}
PROTECTED_FILENAMES = {
    "config.py", "requirements.txt", "pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini",
    "Pipfile", "Pipfile.lock", "poetry.lock", "uv.lock", "package.json", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "go.sum",
}
PROTECTED_SUFFIXES = {
    ".keras", ".h5", ".pt", ".pth", ".onnx", ".pb", ".weights", ".db", ".sqlite", ".sqlite3",
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".mp4",
    ".mov", ".avi", ".zip", ".env",
}
PRESERVE_PYTHON_DIRECTIVES = (
    "type:", "noqa", "nosec", "pragma:", "pylint:", "pyright:", "mypy:", "fmt:", "isort:",
    "coverage:", "# region", "# endregion",
)


class UnsafeSource(ValueError):
    """The source could not be parsed safely enough to edit."""


@dataclass
class CleanResult:
    text: str
    removed_lines: int


def _newline_only(value: str) -> str:
    """Keep exact CR/LF characters while removing the non-newline contents."""
    return "".join(char for char in value if char in "\r\n")


def _line_count_for_spans(text: str, spans: list[tuple[int, int]]) -> int:
    touched: set[int] = set()
    for start, end in spans:
        first = text.count("\n", 0, start) + 1
        last = first + text.count("\n", start, end)
        touched.update(range(first, last + 1))
    return len(touched)


def _apply_spans(text: str, spans: list[tuple[int, int]], *, preserve_lines: bool,
                 separate_tokens: bool = False) -> CleanResult:
    spans = sorted(set(spans))
    result = text
    for start, end in reversed(spans):
        replacement = _newline_only(result[start:end]) if preserve_lines else ""
        if separate_tokens and not replacement:
            before = result[start - 1] if start else ""
            after = result[end] if end < len(result) else ""
            if before and after and not before.isspace() and not after.isspace():
                replacement = " "
        result = result[:start] + replacement + result[end:]
    return CleanResult(result, _line_count_for_spans(text, spans))


def _python_comment_is_directive(comment: str, line_number: int) -> bool:
    lowered = comment.strip().lower()
    if line_number <= 2 and (lowered.startswith("#!") or "coding" in lowered):
        return True
    normalized = lowered.lstrip("# ")
    return any(normalized.startswith(directive.lstrip("# ")) for directive in PRESERVE_PYTHON_DIRECTIVES)


def clean_python(text: str) -> CleanResult:
    spans: list[tuple[int, int]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and not _python_comment_is_directive(token.string, token.start[0]):
                spans.append((token.start[1] + _line_offset(text, token.start[0]),
                              token.end[1] + _line_offset(text, token.end[0])))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise UnsafeSource(f"Python tokenizer could not safely parse the file: {exc}") from exc
    return _apply_spans(text, spans, preserve_lines=False)


def _line_offset(text: str, line_number: int) -> int:
    offset = 0
    for _ in range(line_number - 1):
        newline = text.find("\n", offset)
        if newline < 0:
            return len(text)
        offset = newline + 1
    return offset


class _HTMLCommentParser(html.parser.HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.spans: list[tuple[int, int]] = []
        self.lines: set[int] = set()
        self.line_starts = [0]
        self.line_starts.extend(i + 1 for i, char in enumerate(source) if char == "\n")

    def handle_comment(self, data: str) -> None:
        
        normalized = data.strip().lower()
        if normalized.startswith(("[if", "[endif", "<![endif")):
            return
        line, column = self.getpos()
        approximate = self.line_starts[line - 1] + column
        start = self.source.find("<!--", approximate, min(len(self.source), approximate + 8))
        if start < 0:
            raise UnsafeSource("Could not map an HTML comment back to its source text")
        close = self.source.find("-->", start + 4)
        if close < 0:
            raise UnsafeSource("Unterminated HTML comment")
        end = close + 3
        self.spans.append((start, end))
        first = self.source.count("\n", 0, start) + 1
        last = first + self.source.count("\n", start, end)
        self.lines.update(range(first, last + 1))


def clean_html(text: str) -> CleanResult:
    parser = _HTMLCommentParser(text)
    try:
        parser.feed(text)
        parser.close()
    except UnsafeSource as exc:
        raise UnsafeSource(f"HTML parser could not safely parse the file: {exc}") from exc
    except Exception as exc:  
        raise UnsafeSource(f"HTML parser could not safely parse the file: {exc}") from exc
    return _apply_spans(text, parser.spans, preserve_lines=True)


def clean_css(text: str) -> CleanResult:
    """CSS lexical scan: comments are recognized only outside quoted strings."""
    spans: list[tuple[int, int]] = []
    i, length = 0, len(text)
    while i < length:
        char = text[i]
        if char in "\"'":
            quote = char
            i += 1
            while i < length:
                if text[i] == "\\":
                    i += 2
                elif text[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise UnsafeSource("Unterminated CSS string")
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise UnsafeSource("Unterminated CSS comment")
            body = text[i + 2:end].lstrip()
            if not body.startswith(("!", "#", "@")):
                spans.append((i, end + 2))
            i = end + 2
            continue
        i += 1
    return _apply_spans(text, spans, preserve_lines=True, separate_tokens=True)


def _dollar_quote_end(text: str, start: int) -> int | None:
    """Find a PostgreSQL dollar-quoted string delimiter, if one starts here."""
    close = text.find("$", start + 1)
    if close < 0:
        return None
    tag = text[start + 1:close]
    if tag and (not (tag[0].isalpha() or tag[0] == "_") or
                any(not (char.isalnum() or char == "_") for char in tag)):
        return None
    delimiter = text[start:close + 1]
    end = text.find(delimiter, close + 1)
    if end < 0:
        raise UnsafeSource("Unterminated SQL dollar-quoted string")
    return end + len(delimiter)


def clean_sql(text: str) -> CleanResult:
    """SQL lexical scan for common SQLite/PostgreSQL/MySQL comment and quote forms."""
    spans: list[tuple[int, int]] = []
    i, length = 0, len(text)
    while i < length:
        char = text[i]
        if char == "'" or char == '"' or char == "`":
            quote = char
            i += 1
            while i < length:
                if text[i] == quote:
                    if i + 1 < length and text[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                if text[i] == "\\" and i + 1 < length:
                    i += 2
                else:
                    i += 1
            else:
                raise UnsafeSource("Unterminated SQL quoted value or identifier")
            continue
        if char == "[":
            close = text.find("]", i + 1)
            if close < 0:
                raise UnsafeSource("Unterminated SQL bracket-quoted identifier")
            i = close + 1
            continue
        if char == "$":
            end = _dollar_quote_end(text, i)
            if end is not None:
                i = end
                continue
        if text.startswith("--", i):
            end = i + 2
            while end < length and text[end] not in "\r\n":
                end += 1
            spans.append((i, end))
            i = end
            continue
        if text.startswith("/*", i):
            
            if text.startswith("/*+", i) or text.startswith("/*!", i):
                close = text.find("*/", i + 2)
                if close < 0:
                    raise UnsafeSource("Unterminated SQL hint/executable comment")
                i = close + 2
                continue
            start, depth, i = i, 1, i + 2
            while i < length and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise UnsafeSource("Unterminated SQL block comment")
            spans.append((start, i))
            continue
        i += 1
    return _apply_spans(text, spans, preserve_lines=True, separate_tokens=True)


def _decode_source(raw: bytes, suffix: str) -> tuple[str, str]:
    if suffix in PYTHON_EXTENSIONS:
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            return raw.decode(encoding), encoding
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise UnsafeSource(f"Could not determine Python source encoding: {exc}") from exc
    
    for bom, encoding in ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF32_LE, "utf-32"),
                          (codecs.BOM_UTF32_BE, "utf-32"), (codecs.BOM_UTF16_LE, "utf-16"),
                          (codecs.BOM_UTF16_BE, "utf-16")):
        if raw.startswith(bom):
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError as exc:
                raise UnsafeSource(f"Invalid {encoding} source encoding: {exc}") from exc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        raise UnsafeSource("Non-Python source is not UTF-8/BOM-encoded; skipped to preserve encoding") from exc


def _clean_text(text: str, suffix: str) -> CleanResult:
    if suffix in PYTHON_EXTENSIONS:
        return clean_python(text)
    if suffix in HTML_EXTENSIONS:
        return clean_html(text)
    if suffix in CSS_EXTENSIONS:
        return clean_css(text)
    if suffix in SQL_EXTENSIONS:
        return clean_sql(text)
    if suffix in JS_EXTENSIONS:
        raise UnsafeSource("No JavaScript parser/tokenizer is installed; skipped instead of risking strings, regexes, or templates")
    raise UnsafeSource(f"Unsupported source extension: {suffix or '(none)'}")


def _is_test_file(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    return bool(parts & {"test", "tests"}) or name.startswith("test_") or name.endswith("_test.py")


def _is_virtual_environment(path: Path) -> bool:
    """Recognize virtualenv/Conda directories by their metadata, whatever they are named."""
    return (path / "pyvenv.cfg").is_file() or (path / "conda-meta").is_dir()


def _is_protected(relative: Path, include_tests: bool) -> str | None:
    parts = [part.lower() for part in relative.parts]
    name = relative.name.lower()
    if any(part in PROTECTED_DIRS for part in parts[:-1]):
        return "inside a protected directory"
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        if _is_virtual_environment(current):
            return "inside a virtual environment"
    if name.startswith(".env") or name in PROTECTED_FILENAMES or name.endswith((".lock", ".lockb")):
        return "protected configuration, lockfile, secret, or credential file"
    if relative.suffix.lower() in PROTECTED_SUFFIXES:
        return "protected data/model/media file type"
    if not include_tests and _is_test_file(relative):
        return "test file (pass --include-tests to opt in)"
    return None


def _excluded(relative: Path, patterns: list[str]) -> bool:
    value = relative.as_posix()
    return any(fnmatch.fnmatchcase(value, pattern) or fnmatch.fnmatchcase(relative.name, pattern)
               for pattern in patterns)


def _eligible(relative: Path, include_tests: bool, excludes: list[str]) -> str | None:
    protected = _is_protected(relative, include_tests)
    if protected:
        return protected
    if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"unsupported extension {relative.suffix or '(none)'}"
    if _excluded(relative, excludes):
        return "matched an explicit --exclude pattern"
    return None


def _candidate_paths(single_file: str | None, include_tests: bool, excludes: list[str]):
    if single_file:
        target = Path(single_file)
        if not target.is_absolute():
            target = ROOT / target
        resolved = target.resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError("--file must refer to a file inside the project root")
        if not resolved.is_file():
            raise ValueError(f"--file does not exist or is not a regular file: {single_file}")
        yield resolved.relative_to(ROOT), resolved
        return

    for current, directories, files in os.walk(ROOT, followlinks=False):
        current_path = Path(current)
        kept = []
        for directory in directories:
            full = current_path / directory
            relative = full.relative_to(ROOT)
            reason = "protected directory" if directory.lower() in PROTECTED_DIRS else None
            if not reason and _is_virtual_environment(full):
                reason = "virtual environment directory"
            if full.is_symlink():
                reason = "symbolic-link directory"
            if reason:
                print(f"SKIPPED DIRECTORY {relative.as_posix()} — {reason}")
            else:
                kept.append(directory)
        directories[:] = kept
        for filename in files:
            path = current_path / filename
            if path.is_symlink():
                print(f"SKIPPED {path.relative_to(ROOT).as_posix()} — symbolic link")
                continue
            if path.suffix.lower() in SUPPORTED_EXTENSIONS or filename.lower().startswith(".env"):
                yield path.relative_to(ROOT), path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    except Exception:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _warn_git_changes() -> None:
    try:
        check = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
                               check=False, capture_output=True, text=True)
        if check.returncode != 0:
            print("WARNING: Git metadata is unavailable; uncommitted changes could not be checked.")
            return
        status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                check=True, capture_output=True, text=True)
        if status.stdout.strip():
            print("WARNING: Git reports uncommitted changes. Review them before applying cleanup.")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARNING: Could not check Git status ({exc}); review project changes before applying.")


def _new_backup_dir() -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_ROOT / f"{ROOT.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    backup.mkdir(exist_ok=False)
    if backup.resolve().is_relative_to(ROOT):
        raise OSError("Backup directory must be outside the source tree")
    return backup


def _create_verified_backup(changes: list[dict], include_tests: bool) -> Path:
    backup = _new_backup_dir()
    records = []
    for change in changes:
        relative: Path = change["relative"]
        source: Path = change["path"]
        original: bytes = change["original"]
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        saved = target.read_bytes()
        if saved != original:
            raise OSError(f"Backup verification failed for {relative.as_posix()}")
        records.append({"path": relative.as_posix(), "sha256": _sha256(original),
                        "encoding": change["encoding"], "mode": stat.S_IMODE(source.stat().st_mode)})
    manifest = {"format": 1, "project_root": str(ROOT), "include_tests": include_tests, "files": records}
    manifest_path = backup / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise OSError("Backup manifest verification failed")
    return backup


def _plan(include_tests: bool, excludes: list[str], single_file: str | None) -> tuple[list[dict], int, int]:
    changes, errors = [], 0
    total_comments = 0
    for relative, path in _candidate_paths(single_file, include_tests, excludes):
        label = relative.as_posix()
        reason = _eligible(relative, include_tests, excludes)
        if reason:
            print(f"SKIPPED {label} — {reason}")
            continue
        print(f"INSPECTED {label}")
        try:
            original = path.read_bytes()
            text, encoding = _decode_source(original, path.suffix.lower())
            result = _clean_text(text, path.suffix.lower())
            if result.text == text:
                print(f"SKIPPED {label} — no removable comments")
                continue
            cleaned = result.text.encode(encoding)
            if cleaned == original:
                print(f"SKIPPED {label} — no byte-level change")
                continue
            print(f"PROPOSED {label} — {result.removed_lines} comment line(s)")
            changes.append({"relative": relative, "path": path, "original": original, "cleaned": cleaned,
                            "encoding": encoding, "removed_lines": result.removed_lines})
            total_comments += result.removed_lines
        except UnsafeSource as exc:
            print(f"SKIPPED {label} — {exc}")
        except Exception as exc:  # noqa: BLE001 - report per-file failures and continue
            errors += 1
            print(f"FAILED {label} — {type(exc).__name__}: {exc}")
    return changes, total_comments, errors


def _apply(changes: list[dict], total_comments: int, include_tests: bool) -> int:
    if not changes:
        print(f"SUMMARY: 0 files modified; {total_comments} comment line(s) proposed.")
        return 0
    try:
        backup = _create_verified_backup(changes, include_tests)
    except Exception as exc:
        for change in changes:
            print(f"FAILED {change['relative'].as_posix()} — backup not verified; source left unchanged")
        print(f"FAILED BACKUP — {type(exc).__name__}: {exc}")
        return 1
    print(f"BACKUP VERIFIED {backup}")
    failures = 0
    removed_lines = 0
    for change in changes:
        relative: Path = change["relative"]
        try:
            _atomic_write(change["path"], change["cleaned"], stat.S_IMODE(change["path"].stat().st_mode))
            print(f"MODIFIED {relative.as_posix()} — {change['removed_lines']} comment line(s)")
            removed_lines += change["removed_lines"]
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAILED {relative.as_posix()} — {type(exc).__name__}: {exc}")
    print(f"SUMMARY: {len(changes) - failures} file(s) modified; {removed_lines} comment line(s) removed; "
          f"{failures} failure(s). Backup: {backup}")
    return 1 if failures else 0


def _restore(backup_arg: str) -> int:
    backup = Path(backup_arg).resolve()
    if not backup.is_dir() or backup.is_relative_to(ROOT):
        raise ValueError("Restore source must be an existing backup directory outside the project")
    manifest_path = backup / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Backup manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or Path(manifest.get("project_root", "")).resolve() != ROOT:
        raise ValueError("Backup format or project root does not match this project")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Backup manifest file list is invalid")
    failures = 0
    for record in records:
        try:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe relative path in backup manifest")
            reason = _is_protected(relative, bool(manifest.get("include_tests")))
            if reason or relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"restore target is protected or unsupported: {reason or relative.suffix}")
            source = (backup / relative).resolve()
            if not source.is_relative_to(backup) or not source.is_file():
                raise ValueError("backup file is missing or escapes the backup directory")
            original = source.read_bytes()
            if _sha256(original) != record["sha256"]:
                raise ValueError("backup file hash does not match manifest")
            target = (ROOT / relative).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError("restore target escapes the project root")
            _atomic_write(target, original, int(record.get("mode", 0o644)))
            print(f"RESTORED {relative.as_posix()}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAILED RESTORE {record.get('path', '<unknown>')} — {type(exc).__name__}: {exc}")
    print(f"RESTORE SUMMARY: {len(records) - failures} file(s) restored; {failures} failure(s).")
    return 1 if failures else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview only (the default)")
    mode.add_argument("--apply", action="store_true", help="back up and apply comment removal")
    mode.add_argument("--restore", metavar="BACKUP_DIRECTORY", help="restore originals from a verified backup")
    parser.add_argument("--file", help="inspect one file relative to the project root (or an absolute in-root path)")
    parser.add_argument("--include-tests", action="store_true", help="allow test-file edits; excluded by default")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="exclude a relative-path or filename glob (repeatable)")
    args = parser.parse_args(argv)
    if args.restore and (args.file or args.include_tests or args.exclude):
        parser.error("--restore cannot be combined with --file, --include-tests, or --exclude")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.restore:
        try:
            return _restore(args.restore)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED RESTORE — {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.apply:
        _warn_git_changes()
    try:
        changes, comment_lines, errors = _plan(args.include_tests, args.exclude, args.file)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.apply:
        try:
            result = _apply(changes, comment_lines, args.include_tests)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED APPLY — {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 1 if errors or result else result
    print(f"DRY-RUN SUMMARY: {len(changes)} file(s) would change; {comment_lines} comment line(s) proposed; "
          f"{errors} unexpected error(s). No files were modified.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
