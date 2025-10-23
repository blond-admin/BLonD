import os
import re

from blond.utils.legacy_support import __new_by_old


def all_python_files():
    ret = []

    blond_dir = os.path.abspath("../blond")
    assert os.path.isdir(blond_dir)
    for root, dirs, files in os.walk(blond_dir):
        for file in filter(lambda s: s.endswith(".py"), files):
            pythonfile = os.path.join(root, file)
            ret.append(pythonfile)
    return ret

def old_by_new(new):
    ret = []
    for old_tmp, new_tmp in __new_by_old.items():
        if new_tmp == new:
            ret.append(old_tmp)
    if len(ret) == 1:
        return ret[0]
    else:
        return "AMBIGUOUS"

def any_search_for_in_line(search_for: set, line: str, print_reason=False):
    for s in search_for:
        if s in line:
            if print_reason:
                print(f"\nBecause of {s}")
            return True, s
    return False, None


def check_definitions():
    search_for = set(__new_by_old.keys()) | set(val for key, val in __new_by_old.items())

    myregex = re.compile(r"\n\ndef[\s|\S]*?\:\n")

    for pythonfile in all_python_files():
        with open(pythonfile, "r") as fobj:
            content = fobj.read()
            lines = myregex.findall(content)
            for line in lines:
                if any_search_for_in_line(search_for, line, print_reason=True)[0]:
                    line = line[line.index("def"):]
                    if "\n" in line:
                        line = line[:line.index("\n")]
                    print(line)


def check_attributes():
    search_for_old = set(f"self.{old}" for old in __new_by_old.keys())

    for pythonfile in all_python_files():
        with open(pythonfile, "r") as fobj:
            for line in fobj.readlines():
                if any_search_for_in_line(search_for_old, line)[0]:
                    print(line)


def check_attributes_propierties():
    search_for_new = set(f"self.{new} =" for old, new in __new_by_old.items())
    class_tmp = None
    for pythonfile in all_python_files():
        with open(pythonfile, "r") as fobj:
            for line in fobj.readlines():
                if line.strip().startswith("class"):
                    class_tmp = line
                is_inline, string = any_search_for_in_line(search_for_new, line)
                if is_inline:
                    print("--"* 20)
                    print(pythonfile)
                    print(class_tmp)
                    print(line)
                    newww = string.replace("self.", "").replace(" =", "")
                    print(propose_property(newww, old_by_new(newww)))


def propose_property(new:str, old:str, todo=True):
    ret = f'''
    @property
    def {old}(self):  # TODO
        from warnings import warn
        warn("{old} is deprecated, use {new}", DeprecationWarning)  # TODO
        return self.{new}
    
    @{old}.setter  # TODO
    def {old}(self, val):# TODO
        from warnings import warn
        warn("{old} is deprecated, use {new}", DeprecationWarning)  # TODO
        self.{new} = val
'''
    if not todo:
        ret = ret.replace("  # TODO", "")
    return ret




if __name__ == "__main__":
    # check_definitions()
    # check_attributes()
    check_attributes_propierties()
