#!/bin/bash
cd /home/ubuntu/agentic-framework-main
source .venv/bin/activate
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / (1024**3), 'GB')
else:
    print('No CUDA available')
"