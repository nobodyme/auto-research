"""Pytest configuration and fixtures."""

import pytest
import os
from dotenv import load_dotenv


def pytest_configure(config):
    """Configure pytest."""
    # Load environment variables
    load_dotenv()

    # Register custom markers
    config.addinivalue_line("markers", "integration: integration tests that connect to AWS")
    config.addinivalue_line("markers", "e2e: end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test items."""
    # Skip integration tests if AWS_PROFILE not set
    if not os.getenv('AWS_PROFILE'):
        skip_integration = pytest.mark.skip(reason="AWS_PROFILE not set")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip e2e tests if ANTHROPIC_API_KEY not set
    if not os.getenv('ANTHROPIC_API_KEY'):
        skip_e2e = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
