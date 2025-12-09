rm -f modules/*.rst
rm -rf _build/*

shopt -s globstar nullglob
sphinx-apidoc --implicit-namespaces -o modules ../blond ../blond/*/*/__init__.py -f -e -M -d 2 -P

rm -f modules/modules.rst

python3 create_doc_blond.py
python3 create_doc_blond_main_objects.py
# Add diagrams to each .rst file automatically
for f in ./modules/blond.*.rst; do
    modname="${f##*/}"       # strip path
    modname="${modname%.rst}" # strip .rst
    cat >> "$f" <<EOF

.. inheritance-diagram:: $modname
   :parts: 4
EOF
done

sphinx-build -b html . ./_build/html -W
