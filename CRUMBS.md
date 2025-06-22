# Sunstone `.crumbs` files

`*.crumbs` files capture how an output was generated so downstream tools can detect stale results and rebuild artifacts when inputs change. A `.crumbs` file sits alongside the output it describes and uses JSON format.

## Structure

```json
{
  "generator": "think.ponder_day",          // module or command that produced the file
  "output": "20250524/ponder_day.md",       // path to the generated artifact
  "generated_at": "2025-05-24T12:00:01Z",   // ISO 8601 timestamp of generation
  "args": {"--pro": true},                  // CLI arguments or parameters
  "dependencies": [                          // list of inputs relied upon
    {
      "type": "file",                        // a single file dependency
      "path": "think/ponder_day.txt",
      "mtime": 1716543210,                   // modification time (epoch seconds)
      "sha256": "..."                        // optional checksum
    },
    {
      "type": "glob",                        // files selected via a pattern
      "pattern": "20250524/*_diff.json",
      "files": {                              // each match and its mtime
        "20250524/112000_monitor_1_diff.json": 1716540723,
        "20250524/112000_monitor_2_diff.json": 1716540724
      }
    },
    {
      "type": "model",
      "name": "gemini-2.5-pro-preview-06-05"  // AI model used
    },
    {
      "type": "code",                         // version of the generating script
      "path": "think/ponder_day.py",
      "commit": "0a1b2c3d"                    // git commit hash
    }
  ]
}
```

### Dependency objects

- **file** – path to a single file with its `mtime`. Optionally include `sha256` for stronger change detection.
- **glob** – records the glob `pattern` and a mapping of files matched when the command ran. New files matching the same pattern or updates to existing ones mark the output as stale.
- **model** – name of the model used (e.g. `gemini-2.5-pro-preview-06-05`).
- **code** – path to the generating code and its git `commit` hash.

Additional fields may be added in the future (environment info, run identifiers, etc.) but these core types allow determining if an output should be regenerated.

## Usage

Each tool writes a `.crumbs` file next to its output after generation. Future utilities can read the file to build a dependency graph and decide whether the output is out of date. When creating new tools, capture all input files, globs and models, then write the crumb before exiting.

## Fixture

An example crumb file can be found at `fixtures/ponder_day.crumbs`.
