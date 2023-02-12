#!/usr/bin/bash
device="cuda"

if [ -f $device-venv ]; then
    echo conflict with file \'$device-venv\', expected no file or a directory
    exit 1
fi

if [ ! -d $device-venv ]; then
    virtualenv $device-venv
fi

source $device-venv/bin/activate
pip install -r requirements-$device.txt
