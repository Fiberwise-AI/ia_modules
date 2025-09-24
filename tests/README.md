# Pipeline Tests

This directory contains unit tests, integration tests, and end-to-end tests for the pipeline infrastructure components.

## Test Structure

- **Unit Tests** (`tests/`): 
  - Fast-running tests for individual components
  - Focus on core functionality and edge cases
  - Use mocks and stubs where appropriate

- **Integration Tests** (`tests/integration/`):
  - Tests that verify components work together
  - Test full pipeline execution flows
  - Test database integration and service dependencies

- **End-to-End Tests** (`tests/e2e/`):
  - Complete system tests from start to finish
  - Test real-world scenarios with actual pipeline configurations
  - Verify complete pipeline execution including file I/O

## Running Tests

To run all tests:

