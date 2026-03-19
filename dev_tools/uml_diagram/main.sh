cd "/home/slauber/PycharmProjects/deleteme/blonder/"
pyreverse -o xml -p blond $(cat ./dev_tools/uml_diagram/files.txt)
pyreverse -o svg -p blond $(cat ./dev_tools/uml_diagram/files.txt)
pyreverse -o dot -p blond $(cat ./dev_tools/uml_diagram/files.txt)

mv "./packages_blond.dot" "./dev_tools/uml_diagram/packages_blond.dot"
mv "./packages_blond.svg" "./dev_tools/uml_diagram/packages_blond.svg"
mv "./classes_blond.dot" "./dev_tools/uml_diagram/classes_blond.dot"
mv "./classes_blond.svg" "./dev_tools/uml_diagram/classes_blond.svg"
