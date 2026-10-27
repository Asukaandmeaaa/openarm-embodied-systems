# Inference configurations for OpenArm with dora-rs

This repository provides inference configurations for [OpenArm](https://openarm.dev/) with [dora-rs](https://dora-rs.ai/).

## Configurations

### Real configuration

The current two-camera workstation configuration is
[`dataflow-real-act-two-camera-host.yaml`](dataflow-real-act-two-camera-host.yaml).
Physical ACT is still unverified. The original J7 HOME fault is retained as a
historical diagnosis; the Dora HOME, feedback-age timing and persistent-session
software fixes are complete. Use the
[current session handoff](../../docs/ACT_DORA_NEXT_SESSION_HANDOFF.md) for the
remaining Test A/B/C work and operator gates. Do not start the full physical
ACT graph until its first-action admission and execution budget are implemented.
The [two-camera baseline](REAL_ACT_TWO_CAMERA.md) records the graph contract.

### Simulation configuration

#### Local Inference

set up the local policy server:

```bash
uv venv .venv_server -p 3.12
source .venv_server/bin/activate
uv pip install lerobot==0.6.1 pyarrow
# uv pip install torch torchvision torchaudio --torch-backend=cu128 --upgrade  # for CUDA 12.8
python src/local_policy_server.py /dev/shm/policy-server.socket
```

in another terminal, run the dataflow 

```bash
uv venv .venv -p 3.12
uv pip install dora-rs-cli
source .venv/bin/activate
dora build dataflow-local-inference.yaml --uv
SOCKET=/dev/shm/policy-server.socket dora run dataflow-local-inference.yaml --uv
```

#### Docker Inference

Build the Docker image:

```bash
docker build -t openarm-inference-image-lerobot:latest .
```

Run the dataflow.

```bash
dora build dataflow-docker-inference.yaml --uv
dora run dataflow-docker-inference.yaml --uv
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
