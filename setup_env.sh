#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
if [ "$(uname)" = Darwin ]; then
  omp_dir=$(.venv/bin/python -c 'import pathlib,sklearn; print(pathlib.Path(sklearn.__file__).parent/".dylibs")')
  xgb_lib=$(.venv/bin/python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("xgboost").origin).parent/"lib/libxgboost.dylib")')
  if [ -f "$omp_dir/libomp.dylib" ]; then
    if ! otool -l "$xgb_lib" | grep -Fq '@loader_path/../../sklearn/.dylibs'; then
      install_name_tool -add_rpath '@loader_path/../../sklearn/.dylibs' "$xgb_lib"
    fi
  fi
fi
.venv/bin/python -c 'import numpy,pandas,sklearn,exchange_calendars,xgboost; print("Model environment ready")'
