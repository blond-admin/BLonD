# Abort on the first failing command, so a failed sphinx-build fails the
# script (CI sources this file with errexit; plain `bash create_docs.sh`
# would otherwise report success even for a broken build).
set -euxo pipefail

rm -f modules/*.rst
rm -rf _build/*

cp -r ../blond/examples/* ./examples/ # copy examples into the `docs` scope


shopt -s globstar nullglob

# apidoc is passed a list of __init__.py files to EXCLUDE (the glob below,
# blond/<pkg>/<subpkg>/__init__.py). An excluded __init__ makes apidoc render
# that sub-package as a bare namespace stub -- its __init__ docstring and
# re-exported names are not documented. This blanket exclusion predates this
# script's current form; we preserve it unchanged, with one exception: xsuite.
#
# blond/interfaces/xsuite/ has no direct modules, only __init__.py plus the
# sub-packages elements/ and beam_preparation/. apidoc links such an
# __init__-only package into its parent's toctree only when the __init__ is
# documented; exclude it and the generated blond.interfaces.xsuite.* pages are
# orphaned, so sphinx-build -W aborts ("document isn't included in any
# toctree"). Keeping its __init__ documented both fixes the orphan and renders
# xsuite as a full package, like every other interface.
apidoc_excludes=()
for init_file in ../blond/*/*/__init__.py; do
    if [[ $init_file != ../blond/interfaces/xsuite/__init__.py ]]; then
        apidoc_excludes+=("$init_file")
    fi
done
# rf_noise_cpp ships only a compiled .so, so there is nothing to document.
apidoc_excludes+=(../blond/interfaces/rf_noise_cpp)

sphinx-apidoc --implicit-namespaces -o modules ../blond \
    "../blond/**/*.so" "${apidoc_excludes[@]}" -f -e -M -d 2 -P

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


rm -rf ./examples/ # remove the previously copied examples
mkdir "examples"
touch ./examples/.gitkeep
