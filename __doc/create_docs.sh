rm modules/*.rst
rm -r _build/*

shopt -s globstar nullglob
sphinx-apidoc --implicit-namespaces -o modules ../blond ../blond/*/*/__init__.py -f -e -M -d 5 -P

rm modules/modules.rst

cat > ./modules/blond.rst <<'EOF'
API Documentation
=================

.. toctree::
   blond._core
   blond._generals
   blond.acc_math
   blond.beam_preparation
   blond.cycles
   blond.examples
   blond.experimental
   blond.handle_results
   blond.interfaces
   blond.legacy
   blond.performance_blond3
   blond.physics
   blond.specifics
EOF

# Add diagrams to each .rst file automatically
for f in ./modules/blond.*.rst; do
    modname="${f##*/}"       # strip path
    modname="${modname%.rst}" # strip .rst
    cat >> "$f" <<EOF

.. inheritance-diagram:: $modname
   :parts: 4
EOF
done

sphinx-build -b html -c . -D html_theme=sphinx_rtd_theme -D html_theme_options.navigation_depth=5 . ./_build/html -W