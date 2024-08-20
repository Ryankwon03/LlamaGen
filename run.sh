#!/bin/bash

bash scripts/tokenizer/train_vq.sh --cloud-save-path . --data-path ~/train --image-size 256 --vq-model VQ-16
