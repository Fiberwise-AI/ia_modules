Coverage Improvement Plan
Current status: 50% coverage → Target: 100% coverage
Priority Tiers (by impact and dependencies)
Tier 1: Critical Infrastructure (0% coverage - 1,944 statements)
auth/ (242 statements) - Security foundation
middleware.py, models.py, security.py, session.py
pipeline/pipeline_models.py (124 statements) - Core data structures
pipeline/hitl.py (211 statements) - Human-in-the-loop functionality
pipeline/llm_provider_service.py (150 statements) - LLM integration
agents/roles.py (124 statements) - Agent role definitions
reliability/metrics.py (152 statements) - Core metrics system
reliability/decision_trail.py (180 statements) - Decision tracking
reliability/evidence_collector.py (72 statements) - Evidence gathering
reliability/mode_enforcer.py (97 statements) - Mode management
reliability/replay.py (108 statements) - Replay functionality
reliability/slo_tracker.py (100 statements) - SLO tracking
benchmarking/ci_integration.py (95 statements) - CI/CD integration
dashboard/ (486 statements) - Web dashboard
Tier 2: Low Coverage Modules (8-28% coverage - 802 statements)
checkpoint/redis.py (96 uncovered) - Redis checkpoint backend
checkpoint/sql.py (61 uncovered) - SQL checkpoint backend
pipeline/graph_pipeline_runner.py (195 uncovered) - Graph execution
pipeline/routing.py (95 uncovered) - Pipeline routing logic
pipeline/runner.py (24 uncovered) - Pipeline execution
memory/redis.py (88 uncovered) - Redis memory backend
memory/sql.py (64 uncovered) - SQL memory backend
plugins/loader.py (110 uncovered) - Plugin loading
plugins/builtin/ (276 uncovered) - Built-in plugins
benchmarking/profilers.py (122 uncovered) - Performance profiling
benchmarking/reporters.py (95 uncovered) - Benchmark reporting
cli/visualize.py (62 uncovered) - Visualization commands
Tier 3: Medium Coverage (50-80% coverage - 412 statements)
database/manager.py (83 uncovered) - Database management
database/migrations.py (52 uncovered) - Schema migrations
database/interfaces.py (21 uncovered) - Database interfaces
pipeline/importer.py (56 uncovered) - Pipeline import
pipeline/services.py (11 uncovered) - Pipeline services
benchmarking/comparison.py (24 uncovered) - Benchmark comparison
benchmarking/telemetry_bridge.py (10 uncovered) - Telemetry integration
telemetry/exporters.py (63 uncovered) - Metric exporters
telemetry/integration.py (29 uncovered) - Integration layer
telemetry/tracing.py (31 uncovered) - Distributed tracing
reliability/redis_metric_storage.py (13 uncovered) - Redis metrics
plugins/registry.py (48 uncovered) - Plugin registry
plugins/decorators.py (30 uncovered) - Plugin decorators
plugins/base.py (17 uncovered) - Plugin base classes
tools/core.py (25 uncovered) - Tool framework
Tier 4: High Coverage (80-97% coverage - 202 statements)
agents/orchestrator.py (18 uncovered) - 82.98%
agents/state.py (14 uncovered) - 81.03%
pipeline/core.py (34 uncovered) - 82.50%
pipeline/loop_detector.py (16 uncovered) - 84.70%
pipeline/loop_executor.py (6 uncovered) - 90.41%
cli/validate.py (44 uncovered) - 84.54%
cli/main.py (5 uncovered) - 95.80%
benchmarking/framework.py (35 uncovered) - 80.09%
reliability/ - Multiple files 80-92%
memory/memory_backend.py (4 uncovered) - 92.98%
scheduler/core.py (15 uncovered) - 86.41%
Recommended Execution Order
Phase 1: Foundation (Weeks 1-2)
Test auth module completely
Test pipeline_models.py (data structures)
Test agents/roles.py
Phase 2: Core Systems (Weeks 3-4) 4. Test pipeline/hitl.py 5. Test pipeline/llm_provider_service.py 6. Test graph_pipeline_runner.py and routing.py 7. Complete pipeline/runner.py coverage Phase 3: Reliability & Observability (Weeks 5-6) 8. Test reliability/metrics.py, decision_trail.py, evidence_collector.py 9. Test reliability mode_enforcer, replay, slo_tracker 10. Complete telemetry modules (exporters, tracing, integration) Phase 4: Storage & State (Week 7) 11. Test checkpoint backends (redis, sql) 12. Test memory backends (redis, sql) 13. Complete database modules Phase 5: Extensions (Week 8) 14. Test plugins system (loader, registry, decorators, builtin) 15. Test dashboard module 16. Test benchmarking/ci_integration Phase 6: Polish (Week 9) 17. Complete all 80-95% modules to 100% 18. Add edge case and error path tests 19. Integration test coverage Phase 7: Verification (Week 10) 20. Run full test suite with coverage 21. Identify and fix any remaining gaps 22. Document test patterns and approaches
Key Testing Strategies
Mocking: Use pytest fixtures for external dependencies (Redis, PostgreSQL, LLMs)
Parameterization: Use @pytest.mark.parametrize for multiple input scenarios
Integration Tests: Test component interactions, not just units
Error Paths: Explicitly test exception handling and edge cases
Async Testing: Use pytest-asyncio for async code paths
Test Data Builders: Create factories for complex test objects