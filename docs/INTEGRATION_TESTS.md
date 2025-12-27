# Integration Tests

This directory contains integration tests for Sunstone that are separate from the unit tests in the parent `tests/` directory.

## Purpose

Integration tests verify that different components of the system work together correctly. Unlike unit tests, these tests:

- May require external services or APIs
- Test complete workflows across multiple modules
- May take longer to execute
- Test real file I/O and system interactions

## Running Integration Tests

```bash
# Run all integration tests
make test-integration

# Run with coverage
make test-integration-cov

# Run a specific integration test
make test-integration-only TEST=test_example_integration.py

# Run all tests (unit + integration)
make test-all
```

## Test Markers

Tests can be marked with pytest markers to categorize them:

- `@pytest.mark.integration` - General integration test marker
- `@pytest.mark.slow` - Tests that take > 5 seconds
- `@pytest.mark.requires_api` - Tests requiring external API access
- `@pytest.mark.requires_journal` - Tests requiring full journal setup

## Writing Integration Tests

1. Place test files in this directory with `test_` prefix
2. Use the `integration_journal_path` fixture for temporary journal operations
3. Mark tests appropriately with pytest markers
4. Document any special setup requirements in the test docstring

## Example

```python
import pytest
from pathlib import Path

@pytest.mark.integration
@pytest.mark.slow
def test_complete_workflow(integration_journal_path):
    """Test the complete capture-to-analysis workflow."""
    # Your integration test code here
    pass
```

## Exclusion from Default Tests

These tests are automatically excluded from `make test` through the `--ignore=tests/integration` flag. They run only when explicitly requested via:

- `make test-integration`
- `make test-all`
- Direct pytest invocation with `pytest tests/integration/`