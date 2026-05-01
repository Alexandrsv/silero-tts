#!/usr/bin/env bash
source venv/bin/activate

find_free_port() {
    local port=$1
    local max_port=7910
    while [ $port -le $max_port ]; do
        if ! nc -z localhost $port 2>/dev/null; then
            echo $port
            return 0
        fi
        ((port++))
    done
    return 1
}

DEFAULT_PORT=7860
PORT=$(find_free_port $DEFAULT_PORT)

if [ -z "$PORT" ]; then
    echo "No free port found in range $DEFAULT_PORT-7910"
    exit 1
fi

echo "Using port: $PORT"
python run.py --server-port $PORT "$@"