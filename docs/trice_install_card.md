# TRICE Installability Card

- Scope: `TraceRazor wheel installability`
- Install level: `full_cli_install_ready`
- Install score: **100/100**
- Expected version: `1.0.3`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| wheel_present | yes | dist/tracerazor-1.0.3-py3-none-any.whl | built wheel exists |
| venv_created | yes | 0 | clean virtual environment can be created |
| wheel_installs | yes | 0 | wheel installs with pip --no-deps |
| version_matches | yes | 1.0.3 | 1.0.3 |
| schemas_importable | yes | {"crates_schema_title": "TRICE crates publish card", "install_schema_title": "TRICE installability card", "research_schema_title": "TRICE research card"} | install, crates, and research schemas import from wheel |
| trice_api_importable | yes | {"build_crates_card": true, "build_install_card": true, "build_research_card": true, "verify_install_card_file": true, "verify_research_card_file": true} | public tracerazor.trice install/crates/research APIs import |
| trice_console_works | yes | 0 | tracerazor-trice console script works after wheel install |
| rust_cli_bundled | yes | exit=0; tracerazor 1.0.3
 | tracerazor console script can find a bundled Rust auditor binary |

## Commands

| Command | Exit | Status |
|---|---:|---|
| create_venv | 0 | ok |
| install_wheel | 0 | ok |
| import_probe | 0 | ok |
| trice_console | 0 | ok |
| rust_console | 0 | ok |

## Next Actions

- Publish the install card with the release evidence bundle.

## Hash

- install card: `fec3eb86291021441e1a3f610183c5ba2d57ebbeae885e09473e7782fd41d448`
