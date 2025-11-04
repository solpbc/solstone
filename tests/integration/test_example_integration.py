"""Example integration test to demonstrate the structure."""

import pytest


@pytest.mark.integration
def test_journal_creation_workflow(integration_journal_path):
    """Test creating a complete journal entry workflow."""
    # This is an example integration test that would test
    # the complete workflow from capture to storage

    # Create facet structure
    facet_path = integration_journal_path / "facets" / "test-facet"
    facet_path.mkdir(parents=True)

    # Create facet.json
    facet_json = facet_path / "facet.json"
    facet_json.write_text(
        '{"title": "Test Facet", "description": "Integration test facet"}'
    )

    # Verify structure was created
    assert facet_path.exists()
    assert facet_json.exists()

    # In a real integration test, you would:
    # 1. Use hear module to capture audio
    # 2. Use see module to capture screenshots
    # 3. Use think module to process the data
    # 4. Verify the complete workflow produces expected results


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_processing():
    """Test end-to-end processing pipeline."""
    # This would test the complete pipeline from raw data to processed output
    # Mark as slow since integration tests may take longer
    pass


@pytest.mark.integration
@pytest.mark.requires_api
def test_api_integration():
    """Test integration with external APIs."""
    # Tests that require actual API calls (not mocked)
    # Should be marked with requires_api
    pass
