# dora-openarm-quitter

A [Dora](https://dora-rs.ai/) node that exits when it receives a `quit` command.

## Usage

This node exists just for quitting looping dora-rs dataflow smoothly. Dora-rs nodes in dataflow quit when all input nodes are quit. This dora-node can quit by `command{data=[quit]}` input. For example, [dora-openarm-data-collection-ui](https://github.com/enactic/dora-openarm-data-collection-ui) outputs `command{data=[quit]}`. If this dora-rs node receives `command{data=[quit]}`, this dora-rs node quit by itself. Other dora-rs nodes that use this dora-rs node as input are also quit after this dora-rs node quit.

See also [enactic/dora-openarm-data-collection](https://github.com/enactic/dora-openarm-data-collection) how to use this dora-rs node.

### Inputs

| Input ID | Type | Description |
| --- | --- | --- |
| `command` | Arrow array where first value is a string | If the first value is `"quit"`, the node quit. Any other `command` value is ignored. |
| any other input ID | Any Arrow value | The input is forwarded to an output with the same ID and metadata. |

### Outputs

| Output ID | Type | Description |
| --- | --- | --- |
| same as input ID (except `command`) | Same as input value | For each non-`command` input, the node sends an output with the same ID, value, and metadata. |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
