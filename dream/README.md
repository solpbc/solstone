# sunstone-dream

Exploratory tools that present captured data in a web browser. These utilities serve simple HTML interfaces for reviewing daily summaries. They read from a **journal** directory of daily folders.

## Installation

```bash
pip install -e .
```

## Usage

Two commands are provided:

- `entity-review` builds an index of entities mentioned in your recordings and serves it on a local web server.
- `meeting-calendar` generates a calendar view of meetings pulled from daily folders.

```bash
entity-review <journal> [--port PORT]
meeting-calendar <journal> [--port PORT]
```

Open the printed URL in your browser to explore the results.
