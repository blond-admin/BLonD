@echo off
rem ===========================================================================
rem Windows port of create_docs.sh -- builds the BLonD HTML documentation.
rem
rem Run it from a shell where the docs toolchain is on PATH (e.g. an activated
rem virtualenv), the same assumption create_docs.sh makes:
rem     - python, sphinx-apidoc, sphinx-build
rem     - graphviz `dot` on PATH (required by the inheritance diagrams below;
rem       without it the final `sphinx-build ... -W` step fails)
rem
rem The script cd's into its own directory (docs\) so the relative paths below
rem resolve regardless of the current working directory.
rem ===========================================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

rem --- use the project virtualenv if present ---------------------------------
rem Activates ..\..\.venv so python / sphinx-apidoc / sphinx-build resolve to
rem the venv regardless of how this script is launched (PyCharm run config,
rem double-click, or a plain shell). Guarded so it no-ops where the venv is
rem absent (e.g. the Linux CI, which activates its own venv before the build).
if exist "%~dp0..\..\.venv\Scripts\activate.bat" call "%~dp0..\..\.venv\Scripts\activate.bat"

rem --- clean previous build artifacts ---------------------------------------
if exist modules\*.rst del /q modules\*.rst
if exist _build rmdir /s /q _build

rem --- copy examples into the `docs` scope ----------------------------------
xcopy "..\blond\examples" "examples" /E /I /Y >nul

rem --- generate the API docs ------------------------------------------------
rem Build the exclude list: every ../blond/*/*/__init__.py (two directory
rem levels deep), reproducing the shell's `shopt -s globstar nullglob` glob.
set "EXCLUDES="
for /d %%A in (..\blond\*) do (
    for /d %%B in (%%A\*) do (
        if exist "%%B\__init__.py" set "EXCLUDES=!EXCLUDES! "%%B\__init__.py""
    )
)
sphinx-apidoc --implicit-namespaces -o modules ..\blond !EXCLUDES! -f -e -M -d 2 -P

if exist modules\modules.rst del /q modules\modules.rst

python create_doc_blond.py
python create_doc_blond_main_objects.py

rem --- append an inheritance diagram to each blond.*.rst module page ---------
for %%F in (modules\blond.*.rst) do (
    >>"%%F" echo(
    >>"%%F" echo .. inheritance-diagram:: %%~nF
    >>"%%F" echo    :parts: 4
)

rem --- build the HTML site (warnings treated as errors) ---------------------
sphinx-build -b html . _build\html -W
set "BUILD_RC=%ERRORLEVEL%"

rem --- remove the copied examples, leaving an empty placeholder dir ----------
if exist examples rmdir /s /q examples
mkdir examples
type nul > examples\.gitkeep

endlocal & exit /b %BUILD_RC%
