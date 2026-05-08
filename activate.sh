#!/bin/bash
source ~/Desktop/ncs2/env/bin/activate
source ~/Desktop/ncs2/openvino_2022.3/setupvars.sh 2>/dev/null
echo "✓ NCS2 env listo (Python $(python --version | awk '{print $2}'))"
