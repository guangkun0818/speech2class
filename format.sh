#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19

# Code format script

# Python, yapf version 0.32.0
find ./ -path "./runtime" -prune -o -iname "*.py" -print | xargs yapf -i --style google
