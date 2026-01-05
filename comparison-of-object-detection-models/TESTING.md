# Testing Guide

This project follows Test-Driven Development (TDD) principles. Tests are written before implementation to define expected behavior.

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_model_loader.py -v
```

### Run Specific Test
```bash
pytest tests/test_model_loader.py::TestModelLoader::test_load_yolov8_model -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

## Test Categories

### Unit Tests
- `test_model_loader.py` - Model loading and initialization
- `test_benchmark.py` - Benchmark evaluation components
- `test_detect.py` - Object detection functionality

### Integration Tests
Integration tests may download models and require internet connection:
- COCO evaluation tests
- Full model comparison tests

### Quick Testing
To run tests quickly without downloading large datasets:
```bash
pytest tests/ -m "not slow"
```

## Test Structure

Each test file follows this pattern:
```python
class TestComponent:
    """Test suite for Component."""

    def test_basic_functionality(self):
        """Test basic feature."""
        # Arrange
        component = Component()

        # Act
        result = component.method()

        # Assert
        assert result is not None
```

## Fixtures

Common fixtures are defined in test files:
- `sample_image` - Creates a temporary test image
- `sample_images_dir` - Creates a directory with multiple test images

## CI/CD Considerations

For continuous integration:
1. Tests may require GPU (will fall back to CPU)
2. Some tests download pre-trained models (~10-50MB each)
3. COCO evaluation tests download validation set (~1GB) on first run
4. Use `--quick` flag in benchmark scripts for CI

## Debugging Tests

Run with more verbose output:
```bash
pytest tests/ -vv -s
```

Run with Python debugger on failure:
```bash
pytest tests/ --pdb
```

## Performance Testing

Benchmark tests measure:
- Model loading time
- Inference latency
- Memory usage
- Throughput (FPS)

To profile tests:
```bash
pytest tests/ --durations=10
```

## Expected Test Duration

- Unit tests: < 10 seconds
- Model loading tests: 10-30 seconds (downloads on first run)
- Latency benchmarks: 30-60 seconds
- Full COCO evaluation: 5-15 minutes (depending on hardware)

## Test Data

Test images are generated programmatically using NumPy/OpenCV. No external test data required.

## Mocking

For tests that would be too slow or require external resources:
- Mock COCO dataset downloads
- Mock model inference for unit tests
- Use small iteration counts for latency tests
