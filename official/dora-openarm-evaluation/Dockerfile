# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12

SHELL ["/bin/bash", "-c"]
WORKDIR /project

COPY --from=uv /uv /uvx /bin/
COPY src/ src/

RUN uv venv .venv 
RUN uv pip install lerobot==0.6.1 pyarrow
# For CUDA 12.8, use the following line instead to install PyTorch with CUDA support. Make sure to match the CUDA version with your host machine.
#RUN uv pip install torch torchvision --torch-backend=cu128 --upgrade

ENV VIRTUAL_ENV=/project/.venv \
    PATH="/project/.venv/bin:$PATH"

RUN python -c "\
from lerobot.configs import PreTrainedConfig; \
from lerobot.policies.factory import get_policy_class, make_pre_post_processors; \
cfg = PreTrainedConfig.from_pretrained('enactic/act-openarm-2-cell-pick_up_cube_mujoco'); \
cfg.pretrained_path = 'enactic/act-openarm-2-cell-pick_up_cube_mujoco'; \
get_policy_class(cfg.type).from_pretrained(config=cfg, pretrained_name_or_path=cfg.pretrained_path); \
make_pre_post_processors(cfg, pretrained_path=cfg.pretrained_path, preprocessor_overrides={'device_processor': {'device': cfg.device}})" \
&& python -c "import torchvision; torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)"

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "src/docker_policy_server.py"]
