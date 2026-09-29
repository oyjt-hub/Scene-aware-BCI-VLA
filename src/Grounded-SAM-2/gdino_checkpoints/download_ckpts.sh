#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"
swinb_cogcoor_url="${BASE_URL}v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"



# 下载第一个权重 (Swint OGC) - 约 662MB
wget https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# 下载第二个权重 (Swinb Cogcoor) - 约 900MB+
wget https://hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
echo "All checkpoints are downloaded successfully."
