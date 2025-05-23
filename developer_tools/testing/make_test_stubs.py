import ast
import json
import os
import re
from pathlib import Path

COVERAGE_JSON_PATH = (
    Path(__file__).parent / Path("../../unittests/coverage.json")
).resolve()
PROJECT_ROOT = (Path(__file__).parent / Path("../../blond/")).resolve()
TEST_ROOT = (Path(__file__).parent / Path("../../unittests")).resolve()


def classname_to_varname(name):
    # Insert underscore before each uppercase letter that follows a lowercase letter or number
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase letters or end of string
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def load_coverage_data(path: str | os.PathLike):
    with open(path, "r") as f:
        return json.load(f)


def get_function_end_lineno(node):
    """Estimate the last line number of a function."""
    if hasattr(node, "end_lineno"):
        return node.end_lineno
    max_lineno = node.lineno
    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            max_lineno = max(max_lineno, child.lineno)
    return max_lineno


def extract_untested_functions(cov_data):
    untested = {}
    files = cov_data.get("files", {})

    for filepath, info in files.items():
        missing_lines = set(info.get("missing_lines", []))
        if not missing_lines:
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)

        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = []
                self.results = []  # List of (func_name, class_name, args)
                self.class_methods = {}  # class_name -> list of (func_name, args)
                self.class_missing = (
                    set()
                )  # classes with at least one missing method (not __init__)

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.class_methods[self.stack[-1]] = []
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                start = node.lineno
                end = get_function_end_lineno(node)
                class_name = self.stack[-1] if self.stack else None
                args = [arg.arg for arg in node.args.args]
                if class_name and args and args[0] == "self":
                    args = args[1:]  # skip 'self' for methods

                # Save all class methods for possible __init__ add later
                if class_name:
                    self.class_methods.setdefault(class_name, []).append(
                        (node.name, args)
                    )

                if any(start <= line <= end for line in missing_lines):
                    # Mark class as having missing methods (except __init__)
                    if class_name and node.name != "__init__":
                        self.class_missing.add(class_name)

                    self.results.append((node.name, class_name, args))

                self.generic_visit(node)

        visitor = FunctionVisitor()
        visitor.visit(tree)

        # Now add __init__ methods for classes that had missing other methods
        for cls in visitor.class_missing:
            methods = visitor.class_methods.get(cls, [])
            for func_name, args in methods:
                if func_name == "__init__":
                    # Only add if not already in results
                    if not any(
                        rn == "__init__" and rcls == cls
                        for rn, rcls, _ in visitor.results
                    ):
                        visitor.results.append(("__init__", cls, args))

        if visitor.results:
            untested[filepath] = visitor.results

    return untested


def write_boilerplate_tests(untested_functions):
    for src_path, functions in untested_functions.items():
        functions = list(
            sorted(
                functions, key=lambda x: ("", x[0]) if x[1] is None else (x[1], x[0])
            )
        )
        rel_path = os.path.relpath(src_path, PROJECT_ROOT)
        test_path_dir = os.path.join(TEST_ROOT, os.path.dirname(rel_path))
        test_filename = f"test_{os.path.basename(src_path)}"
        test_file = os.path.join(test_path_dir, test_filename)

        os.makedirs(test_path_dir, exist_ok=True)

        # Read existing content to avoid duplicate stubs
        existing = ""
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                existing = f.read()

        content = ""
        if "import unittest" not in existing:
            content += "import unittest\n\n"

        classes = {}

        for func_name, class_name, args in functions:
            test_func = f"test_{func_name}"
            if test_func in existing:
                continue

            # Build function call string with keyword args set to None
            call_args = ", ".join([f"{arg}=None" for arg in args])
            if class_name:
                var_name = classname_to_varname(class_name)
                if test_func == "test___init__":
                    call_line = (
                        f"        self.{var_name} = {class_name}({call_args})\n"
                        f"    @unittest.skip\n"
                        f"    def {test_func}(self):\n"
                        f"        pass # calls __init__ in  self.setUp"
                    )

                    test_code = (
                        f"    @unittest.skip\n"
                        f"    def setUp(self):\n        # TODO: "
                        f"implement test for `{func_name}`\n{call_line}\n"
                    )
                else:
                    call_line = f"        self.{var_name}.{func_name}({call_args})"
                    test_code = (
                        f"    @unittest.skip\n"
                        f"    def {test_func}(self):\n        # TODO: implement test for `{func_name}`\n{call_line}\n"
                    )
                test_class = f"Test{class_name}"
            else:
                call_line = f"        {func_name}({call_args})"
                test_code = (
                    f"    @unittest.skip\n"
                    f"    def {test_func}(self):\n        # TODO: implement test for `{func_name}`\n{call_line}\n"
                )
                test_class = "TestFunctions"

            classes.setdefault(test_class, []).append(test_code)

        for class_name, methods in classes.items():
            if f"class {class_name}" not in existing:
                content += f"\n\nclass {class_name}(unittest.TestCase):"
            for method in methods:
                content += "\n" + method

        # Write to file
        with open(test_file, "a", encoding="utf-8") as f:
            f.write(content)


def main():
    cov_data = load_coverage_data(COVERAGE_JSON_PATH)
    untested_funcs = extract_untested_functions(cov_data)
    write_boilerplate_tests(untested_funcs)
    print("✅ Boilerplate test cases generated with mirrored structure.")


if __name__ == "__main__":
    main()
