"""Real local monitor, validation and workflow boundary regressions."""

import asyncio
import unittest
from unittest.mock import patch

from core.config_runtime_validator import RuntimeConfigValidator, ValidationResult, ValidationSeverity
from core.performance_monitor import PerformanceMonitor, count_calls, track_performance
from core.workflow_effort_calculator import EffortCalculator
from core.workflow_formatter import WorkflowFormatter
from core.workflow_models import (
    EffortEstimation,
    OutputFormat,
    PersonaType,
    Requirement,
    Workflow,
    WorkflowMilestone,
    WorkflowPhase,
)


class MonitoringWorkflowContracts(unittest.TestCase):
    def test_unestimated_milestone_formats_default_tasks(self):
        workflow = Workflow(
            name="Unestimated", phases=[WorkflowPhase(name="Phase", milestones=[WorkflowMilestone(name="Work")])]
        )
        rendered = WorkflowFormatter().format_workflow(workflow, OutputFormat.TASKS)
        self.assertIn("设计和规划 (1h)", rendered)
        self.assertIn("开发实现 (2h)", rendered)
        self.assertIn("测试验证 (1h)", rendered)

    def test_zero_total_effort_formats_zero_percent_persona(self):
        workflow = Workflow(
            name="Zero",
            effort_estimation=EffortEstimation(total_hours=0, breakdown_by_persona={PersonaType.BACKEND: 0}),
        )
        rendered = WorkflowFormatter().format_workflow(workflow, OutputFormat.DETAILED)
        self.assertIn("0h (0.0%)", rendered)

    def test_critical_validation_result_cannot_report_pass(self):
        report = RuntimeConfigValidator().generate_validation_report(
            [ValidationResult(False, ValidationSeverity.CRITICAL, "storage", "Controlled critical failure")]
        )
        self.assertIn("1 个错误", report)
        self.assertNotIn("✅ 验证通过", report)

    def test_validation_report_preserves_warning_and_pass_states(self):
        validator = RuntimeConfigValidator()
        self.assertIn("未发现问题", validator.generate_validation_report([]))
        warning = ValidationResult(False, ValidationSeverity.WARNING, "batch", "Review batch size")
        self.assertIn("1 个警告", validator.generate_validation_report([warning]))
        self.assertNotIn("个错误", validator.generate_validation_report([warning]))

    def test_iso_metrics_keep_history_and_absolute_counter_values(self):
        from datetime import datetime

        from core.performance_monitor import MetricsCollector, MetricType, PerformanceMetric

        collector = MetricsCollector()
        timestamp = datetime.now().isoformat()
        first = PerformanceMetric("network.total", 100, MetricType.COUNTER, timestamp)
        collector.record_metric(first)
        collector.record_metric(PerformanceMetric("network.total", 150, MetricType.COUNTER, timestamp))
        self.assertEqual(first.to_dict()["timestamp"], timestamp)
        self.assertEqual(collector.get_current_value("network.total"), 150)
        self.assertEqual(len(collector.get_metrics_history("network.total", datetime.fromisoformat(timestamp))), 2)

    def test_explicit_zero_effort_is_not_missing_estimate(self):
        self.assertEqual(EffortCalculator().calculate_base_effort([Requirement(estimated_effort=0)]), 0)

    def test_reused_count_decorator_separates_functions_and_accumulates(self):
        monitor = PerformanceMonitor()
        decorate = count_calls()

        @decorate
        def first():
            return "first"

        @decorate
        def second():
            return "second"

        with patch("core.performance_monitor.get_performance_monitor", return_value=monitor):
            self.assertEqual(first(), "first")
            self.assertEqual(first(), "first")
            self.assertEqual(second(), "second")
        prefix = f"function.{__name__}."
        self.assertEqual(monitor.metrics_collector.get_current_value(prefix + "first.calls"), 2)
        self.assertEqual(monitor.metrics_collector.get_current_value(prefix + "second.calls"), 1)

    def test_reused_performance_decorator_preserves_sync_async_names(self):
        monitor = PerformanceMonitor()
        decorate = track_performance()

        @decorate
        def first():
            return 42

        @decorate
        async def second():
            return "async"

        with patch("core.performance_monitor.get_performance_monitor", return_value=monitor):
            self.assertEqual(first(), 42)
            self.assertEqual(asyncio.run(second()), "async")
        prefix = f"function.{__name__}."
        self.assertEqual(len(monitor.metrics_collector.get_metrics_history(prefix + "first.duration")), 1)
        self.assertEqual(len(monitor.metrics_collector.get_metrics_history(prefix + "second.duration")), 1)


class FormatterOutputCompatibility(unittest.TestCase):
    """Complete valid workflow output captured from legacy commit 5400754."""

    @staticmethod
    def full_workflow():
        from datetime import datetime, timedelta

        from core.workflow_models import (
            ComplexityAnalysis,
            CriticalPath,
            Dependency,
            MCPResults,
            ParallelStream,
            PRDStructure,
            Priority,
            RequirementCategories,
            Risk,
            RiskLevel,
            WorkflowStep,
        )

        requirements = [
            Requirement(title=f"Requirement {i}", description="Requirement details", acceptance_criteria=["Verified"])
            for i in range(6)
        ]
        step = WorkflowStep(
            name="Build",
            description="Build details",
            estimated_effort=8,
            persona=PersonaType.BACKEND,
            tools_required=["Local tool"],
            code_examples=["result = 42"],
            mcp_context={"local": "Context"},
            deliverables=["Implementation"],
            acceptance_criteria=["Test passes"],
        )
        full = WorkflowMilestone(
            name="Full milestone",
            description="Milestone details",
            estimated_effort=16,
            priority=Priority.HIGH,
            steps=[step, WorkflowStep(name="Minimal step")],
            success_criteria=["Contract met"],
            dependencies=["external"],
            risks=["high"],
            personas_involved=[PersonaType.BACKEND],
        )
        default = WorkflowMilestone(name="Default steps", estimated_effort=8, priority=Priority.CRITICAL)
        phase = WorkflowPhase(
            id="phase",
            name="Implementation",
            description="Phase details",
            estimated_effort=80,
            milestones=[full, default],
            dependencies=["external"],
            risks=["high"],
            personas_involved=[PersonaType.BACKEND],
        )
        return Workflow(
            name="Compatibility",
            description="Project overview",
            created_at=datetime(2026, 1, 2, 3, 4, 5),
            prd_structure=PRDStructure(title="PRD", author="Owner", overview="PRD overview"),
            requirements=requirements,
            requirement_categories=RequirementCategories(functional_requirements=requirements),
            phases=[phase, WorkflowPhase(name="Empty phase", estimated_effort=40)],
            risks=[
                Risk(
                    id="high",
                    name="High risk",
                    description="High details",
                    likelihood=RiskLevel.HIGH,
                    mitigation_strategies=["Review", "Test"],
                ),
                Risk(name="Medium risk", description="Medium details"),
                Risk(name="Low risk", likelihood=RiskLevel.LOW, impact=RiskLevel.LOW),
            ],
            dependencies=[
                Dependency(name="External", type="external", owner="Owner", estimated_resolution_time=2),
                Dependency(name="Internal"),
            ],
            parallel_streams=[
                ParallelStream(
                    name="Parallel", description="Parallel details", estimated_effort=16, required_team_size=2
                )
            ],
            critical_path=CriticalPath(
                total_duration=timedelta(days=3),
                total_effort=80,
                bottlenecks=["missing", "phase"],
                optimization_opportunities=["Review schedule"],
            ),
            complexity_analysis=ComplexityAnalysis(
                complexity_factors=["Integration"], simplification_opportunities=["Reuse"]
            ),
            effort_estimation=EffortEstimation(total_hours=120, breakdown_by_persona={PersonaType.BACKEND: 80}),
            activated_personas=[PersonaType.BACKEND, PersonaType.QA],
            persona_recommendations={PersonaType.QA: "Verify contracts"},
            mcp_results=MCPResults(
                context7_results={"pattern": "Known"},
                sequential_results={"analysis": "Complete"},
                magic_results={"component": "Existing"},
                integration_recommendations=["Keep contract"],
            ),
            quality_gates=["Checks pass"],
            success_metrics=["No regression"],
            validation_criteria=["Reviewed"],
        )

    def test_all_formats_preserve_complete_valid_workflow(self):
        import hashlib

        expected = {
            "roadmap": "583d3f2ce39b639be57341e1b5a7579b0e61731d6bd824b170a9ed8133cc6726",
            "tasks": "3ec42bce75ebeeaeafaa9d8c82a61120a85294f38b72cbfdd4a3c1a2dd23e198",
            "detailed": "f1ea26bdf2ea99e5c02582c082fa4203107a87c3e04fd978dd37453ba4f880d5",
        }
        workflow = self.full_workflow()
        for output_format in OutputFormat:
            with self.subTest(format=output_format.value):
                rendered = WorkflowFormatter().format_workflow(workflow, output_format)
                self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), expected[output_format.value])

    def test_unsupported_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的输出格式"):
            WorkflowFormatter().format_workflow(Workflow(name="Invalid"), "unsupported")
