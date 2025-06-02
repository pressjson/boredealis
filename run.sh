#!/usr/bin/env bash

# for i in "$@"; do
#     echo "$i"
# done

venv_base="venv"
path_to_venv="venv/bin/activate"

if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed and in \$PATH. Exiting . . ." >&2
    exit
fi

was_in_venv=false
# check if venv. if not, bootstrap
if [[ "$VIRTUAL_ENV" ]]; then
    echo "In a virtual enviornment already."
    was_in_venv=true
elif [[ ! -d "$venv_base" ]]; then
    echo "Virtual enviornment not detected. Installing a virtual enviornment . . ."
    python3 -m venv venv
    . "$path_to_venv"
    pip install -r requirements.txt
else
    echo "Activating virtual enviornment."
    . "$path_to_venv"
fi

# if [[ "$OSTYPE" == "darwin"* ]]; then
#     osacript -e 'tell application "Terminal" to do shell script "python3 main.py $@"'
#     exit
# fi

"python3" "main.py" "$@"

# deactivate venv if was in venv
if [ "$was_in_venv" = false ]; then
    deactivate
fi
