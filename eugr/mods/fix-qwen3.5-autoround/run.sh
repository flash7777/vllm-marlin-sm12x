#!/bin/bash
set -e
patch -p1 -d /usr/local/lib/python3.12/dist-packages < transformers.patch