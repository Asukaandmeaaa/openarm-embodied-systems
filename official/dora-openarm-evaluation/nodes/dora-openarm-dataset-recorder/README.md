# dora-openarm-dataset-recorder

A [Dora](https://dora-rs.ai/) node that records data as an OpenArm dataset.

Set `RECORD_HZ` or pass `--record-hz` to cap each arm action/observation stream.
An optional `episode_metadata` JSON input is snapshotted on `start` and merged
into that episode's metadata entry (used for simulation seed and object pose).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
