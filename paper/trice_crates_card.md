# TRICE Crates Publish Card

- Scope: `TraceRazor crates.io staged publication`
- Workspace version: `1.0.3`
- Crates level: `publish_plan_locked`
- Publish score: **80/100**
- Local publish plan locked: `true`
- Cargo install claim allowed: `false`

## Checks

| Check | Passed | Observed | Required |
|---|---:|---|---|
| workspace_manifest_present | yes | Cargo.toml | workspace Cargo.toml is present |
| crate_manifests_present | yes | 6/6 | all publish crate manifests are present |
| version_alignment | yes | {"packages": {"tracerazor": "1.0.3", "tracerazor-core": "1.0.3", "tracerazor-ingest": "1.0.3", "tracerazor-semantic": "1.0.3", "tracerazor-server": "1.0.3", "tracerazor-store": "1.0.3"}, "workspace": "1.0.3"} | workspace and crate package versions match |
| publish_order_topological | yes | {"tracerazor": ["tracerazor-core", "tracerazor-ingest", "tracerazor-semantic", "tracerazor-server", "tracerazor-store"], "tracerazor-core": [], "tracerazor-ingest": ["tracerazor-core"], "tracerazor-semantic": [], "tracerazor-server": ["tracerazor-core", "tracerazor-ingest", "tracerazor-semantic", "tracerazor-store"], "tracerazor-store": ["tracerazor-core"]} | each local dependency appears earlier in the publish order |
| dependency_versions_pinned | yes | {"tracerazor": {"tracerazor-core": "1.0.3", "tracerazor-ingest": "1.0.3", "tracerazor-semantic": "1.0.3", "tracerazor-server": "1.0.3", "tracerazor-store": "1.0.3"}, "tracerazor-core": {}, "tracerazor-ingest": {"tracerazor-core": "1.0.3"}, "tracerazor-semantic": {}, "tracerazor-server": {"tracerazor-core": "1.0.3", "tracerazor-ingest": "1.0.3", "tracerazor-semantic": "1.0.3", "tracerazor-store": "1.0.3"}, "tracerazor-store": {"tracerazor-core": "1.0.3"}} | all local crate dependencies pin the workspace version |
| metadata_complete | yes | {"tracerazor": [], "tracerazor-core": [], "tracerazor-ingest": [], "tracerazor-semantic": [], "tracerazor-server": [], "tracerazor-store": []} | description, license, repository, readme, keywords, and categories are present |
| stage_one_publishable | yes | {"tracerazor-core": true, "tracerazor-semantic": true} | first-stage crates have no unpublished local dependencies |
| readme_install_honesty | yes | contains_cargo_install=False; missing: target=1.0.3 latest=none | README does not claim cargo install until tracerazor is live on crates.io |
| public_crates_live | no | {"tracerazor": "missing", "tracerazor-core": "missing", "tracerazor-ingest": "missing", "tracerazor-semantic": "missing", "tracerazor-server": "missing", "tracerazor-store": "missing"} | all six crates are published at the local version |
| cargo_install_truth | no | missing: target=1.0.3 latest=none | cargo install tracerazor is true for the local version |

## Publish Order

| Stage | Crate | Local dependencies | Registry | Currently publishable |
|---:|---|---|---|---:|
| 1 | `tracerazor-core` | none | missing | yes |
| 1 | `tracerazor-semantic` | none | missing | yes |
| 2 | `tracerazor-ingest` | tracerazor-core | missing | no |
| 2 | `tracerazor-store` | tracerazor-core | missing | no |
| 3 | `tracerazor-server` | tracerazor-core, tracerazor-ingest, tracerazor-semantic, tracerazor-store | missing | no |
| 4 | `tracerazor` | tracerazor-core, tracerazor-ingest, tracerazor-semantic, tracerazor-server, tracerazor-store | missing | no |

## Commands

- `cargo package -p tracerazor-core --allow-dirty`
- `cargo publish -p tracerazor-core`
- `cargo package -p tracerazor-semantic --allow-dirty`
- `cargo publish -p tracerazor-semantic`
- `cargo package -p tracerazor-ingest --allow-dirty`
- `cargo publish -p tracerazor-ingest`
- `cargo package -p tracerazor-store --allow-dirty`
- `cargo publish -p tracerazor-store`
- `cargo package -p tracerazor-server --allow-dirty`
- `cargo publish -p tracerazor-server`
- `cargo package -p tracerazor --allow-dirty`
- `cargo publish -p tracerazor`
- `cargo install tracerazor --locked`

## Next Actions

- Publish stage-ready crates for 1.0.3: tracerazor-core, tracerazor-semantic.
- Regenerate this crates card after crates.io indexes each published crate.
- Continue stage by stage until the final `tracerazor` crate is live, then verify `cargo install tracerazor --locked`.

## Hash

- crates card: `ee61306c7dbc2566c0be2de4e5db3fd101ced780725c78cea483440ca127b7ea`
