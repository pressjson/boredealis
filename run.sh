#!/usr/bin/env bash

# for i in "$@"; do
#     echo "$i"
# done

on_exit() {
    echo "Caught a signal. Cleaning up real quick . . ."
    # rm -rf -- "tmp"
}

trap on_exit SIGINT SIGHUP SIGQUIT SIGILL SIGABRT SIGFPE SIGSEGV SIGPIPE SIGALRM SIGTERM

# github_models_url_base="https://github.com/pressjson/boredealis/releases/download/Models"
# filters=(96 128)
# extension="_checkpoint_best.pth"
# models_dir="models"
# if [[ ! -d "$models_dir" ]]; then
#     echo "$models_dir directory not found. Cloning models from $github_models_url_base"
#     mkdir "$models_dir"
#     for item in "${filters[@]}"; do
#         url="$github_models_url_base/$item$extension"
#         wget -q --show-progress "$url" --directory-prefix "$models_dir"
#     done
# fi
venv_base="venv"
path_to_venv="venv/bin/activate"

# if ! command -v ffmpeg &>/dev/null; then
#     echo "FFmpeg is not installed and in \$PATH. Exiting . . ." >&2
#     exit
# fi

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
    echo "If you want to use CUDA/ROCm, uninstall torch torchaudio torchvision and install directly from PyTorch."
else
    echo "Activating virtual enviornment."
    . "$path_to_venv"
fi

# if [[ "$OSTYPE" == "darwin"* ]]; then
#     osacript -e 'tell application "Terminal" to do shell script "python3 main.py $@"'
#     exit
# fi

"python3" "-u" "main.py" "$@"

# deactivate venv if was in venv
if [ "$was_in_venv" = false ]; then
    deactivate
fi
