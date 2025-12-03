"""Prints the code statistics."""

import os
import re


def count_type_hints(code):
    """Count the number of type hints in a line of code."""
    # Match patterns like 'param: Type', 'variable: Type', and function signatures with return types
    if "website:" in code:
        return 0
    if ": utf" in code:
        return 0
    if '"' in code or "'" in code:
        return 0
    if "lambda" in code:
        return 0
    if code.strip().startswith("#"):
        return 0
    type_hint_pattern = r"\w+: \w+"
    matches = re.findall(type_hint_pattern, code)
    n = len(matches)
    if n > 0:
        pass
    return n


def analyze_file(file_path):
    """Analyze a single Python file for docstrings, code lines, and type hints."""
    docstring_lines = 0
    code_lines = 0
    type_hint_count = 0
    in_docstring = False

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line_ = line.strip()
            if line_ == "":
                continue  # Skip empty lines

            if (
                line_.startswith('"""')
                or line_.startswith("'''")
                or line_.startswith('r"""')
                or line_.startswith("r'''")
            ):
                in_docstring = not in_docstring
                docstring_lines += 1
                continue

            if in_docstring:
                docstring_lines += 1
                continue

            # Count lines of code
            code_lines += 1

            # Count type hints in the line
            type_hint_count += count_type_hints(line_)

    return docstring_lines, code_lines, type_hint_count


def exclude_in_string(string, exclude_folders) -> bool:
    """Check if any exclusion matches string.

    Returns
    -------
    bool
        If exclusion is found in string.
    """
    return any(exc in string for exc in exclude_folders)


def analyze_directory(directory_path, exclue_folders):
    """Analyze all Python files in a directory."""
    total_docstring_lines = 0
    total_code_lines = 0
    total_type_hint_count = 0

    for root, _, files in os.walk(directory_path):
        if exclude_in_string(root, exclue_folders):
            continue

        for file in files:
            if file == "__init__.py":
                continue
            if file == "_version.py":
                continue
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(file_path)
                docstring_lines, code_lines, type_hint_count = analyze_file(
                    file_path
                )
                total_docstring_lines += docstring_lines
                total_code_lines += code_lines
                total_type_hint_count += type_hint_count

    return total_docstring_lines, total_code_lines, total_type_hint_count


def main():
    """Prints the code statistics."""
    package_directory = "/home/slauber/PycharmProjects/deleteme/blonder/blond"

    docstring_lines, code_lines, type_hint_count = analyze_directory(
        package_directory,
        exclue_folders=("legacy/", "experimental/", "performance_blond3/"),
    )

    print(f"Total docstring lines: {docstring_lines}")
    print(f"Total code lines: {code_lines}")
    print(f"Total type hints: {type_hint_count}")


if __name__ == "__main__":
    main()
