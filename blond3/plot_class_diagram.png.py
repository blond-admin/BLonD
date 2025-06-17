import subprocess
import pylint  # NOQA

subprocess.run(["ls"], check=True)
subprocess.run("pyreverse classes.py".split(" "), check=True)
subprocess.run("dot -Tpng classes.dot -o class_diagram.png".split(" "))
