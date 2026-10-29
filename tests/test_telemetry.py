"""OpenTelemetry (agenticwf.kb.core.common.telemetry): a caller's trace reaches searches, model calls and tasks."""

import asyncio
import io

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agenticwf.kb.core.common import telemetry
from agenticwf.kb.runtime.models import clients

from conftest import Caller, drain, make_client, user
from test_knowledge import DOC

EXPORTER = InMemorySpanExporter()
READER = InMemoryMetricReader()
RW = ("app:access", "knowledge:read", "knowledge:write")
# What the agent sends with its kb_* calls: its `execute_tool` span.
AGENT_TRACE = "0af7651916cd43dd8448eb211c80319c"
AGENT_SPAN = "b7ad6b7169203331"


@pytest.fixture(scope="module", autouse=True)
def providers():
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(tracer_provider)   # once per process; nothing else sets one in the tests
    metrics.set_meter_provider(MeterProvider(metric_readers=[READER]))
    tracing = telemetry._tracing
    telemetry._tracing = True   # as setup() leaves it: apps get the request instrumentation
    yield
    telemetry._tracing = tracing


@pytest.fixture
def spans():
    EXPORTER.clear()
    return lambda: EXPORTER.get_finished_spans()


def points(name):
    """The data points of one metric, from the in-memory reader."""
    found = []
    for resource in READER.get_metrics_data().resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name == name:
                    found.extend(metric.data.data_points)
    return found


def test_on_only_with_an_endpoint(monkeypatch):
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
                 "OTEL_METRICS_EXPORTER", "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(name, raising=False)
    assert not telemetry.enabled() and not telemetry.metrics_enabled()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    assert telemetry.enabled() and not telemetry.metrics_enabled()      # Jaeger takes no metrics
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    assert telemetry.metrics_enabled()
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert not telemetry.enabled() and not telemetry.metrics_enabled()


def test_spans_do_nothing_without_the_sdk(monkeypatch):
    monkeypatch.setattr(telemetry, "_trace", None)
    with telemetry.span("anything", a=1) as current:
        telemetry.annotate(current, b=2)
        telemetry.fail(current, "no")
    assert not current.is_recording() and telemetry.traceparent() is None


def test_a_model_call_is_a_gen_ai_span(monkeypatch, spans):
    async def send(spec, path, body, timeout):
        return {"choices": [{"message": {"content": "secret answer"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}

    monkeypatch.setattr(clients, "_send", send)
    spec = clients.ModelSpec(id="m1", name="qwen3", type="KnowledgeQA", provider="vllm")
    asyncio.run(clients.ChatModel(spec).complete([{"role": "user", "content": "secret question"}]))
    (chat,) = [s for s in spans() if s.name == "chat qwen3"]
    assert chat.attributes["gen_ai.operation.name"] == "chat" and chat.attributes["gen_ai.provider.name"] == "vllm"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 12 and chat.attributes["gen_ai.usage.output_tokens"] == 3
    assert not any("secret" in str(v) for v in chat.attributes.values())
    assert {p.attributes["gen_ai.token.type"] for p in points("gen_ai.client.token.usage")
            if p.attributes["gen_ai.request.model"] == "qwen3"} == {"input", "output"}


def test_the_agents_trace_reaches_search_and_tasks(database, spans):
    headers = {"traceparent": f"00-{AGENT_TRACE}-{AGENT_SPAN}-01"}
    with make_client("authsvc") as client:
        alice = Caller(client, user("alice", RW))
        kb = alice.post("/kb/api/v1/knowledge-bases", json={"name": "Docs"}).json()["data"]["id"]
        uploaded = alice.post(f"/kb/api/v1/knowledge-bases/{kb}/knowledge/file", headers=headers,
                              files={"file": ("pilot.md", io.BytesIO(DOC), "text/markdown")})
        assert uploaded.status_code == 201
        assert drain(client) == 1
        searched = alice.post("/kb/api/v1/knowledge-search", headers=headers,
                              json={"query": "restart the pilot", "knowledge_base_ids": [kb]})
        assert searched.status_code == 200 and searched.json()["data"]

    found = spans()
    by_id = {s.context.span_id: s for s in found}
    (searched,) = [s for s in found if s.name == "kb.search"]
    assert searched.context.trace_id == int(AGENT_TRACE, 16)              # inside the agent's trace
    assert searched.attributes["agenticwf.kb.search.results"] >= 1
    assert searched.attributes["agenticwf.kb.search.knowledge_bases"] == 1
    assert {by_id[s.parent.span_id].name for s in found if s.name.startswith("kb.search.")} == {"kb.search"}
    assert not any("pilot" in str(v) for s in found for v in s.attributes.values())   # never the query

    (task,) = [s for s in found if s.name.startswith("task ")]
    assert task.attributes["agenticwf.kb.task.id"] and task.links      # queued by the upload request:
    assert task.links[0].context.trace_id == int(AGENT_TRACE, 16)       # linked to it, a trace of its own
    assert task.context.trace_id != int(AGENT_TRACE, 16)
    assert any(p.attributes["agenticwf.kb.task.outcome"] == "done" for p in points("agenticwf.kb.task.duration"))
    assert points("agenticwf.kb.search.duration")


def test_a_scheduled_task_links_to_its_schedule(database, spans):
    from agenticwf.kb.core.orm.base.session import dispose_async_engine
    from agenticwf.kb.runtime.tasks.runner import HandlerSpec, RunnerSettings, ScheduleSpec, TaskRunner

    async def noop(ctx):
        pass

    async def scenario():
        try:
            runner = TaskRunner({"maint:t": HandlerSpec(noop, 5)}, None, RunnerSettings(queues=("default",)),
                                schedules=(ScheduleSpec("t", 3600, "maint:t"),))
            assert await runner.tick() == ["t"]
            await runner.drain()
        finally:
            await dispose_async_engine()

    asyncio.run(scenario())
    found = spans()
    (scheduled,) = [s for s in found if s.name == "schedule t"]
    (task,) = [s for s in found if s.name == "task maint:t"]
    assert scheduled.attributes["agenticwf.kb.task.type"] == "maint:t"
    assert [link.context.span_id for link in task.links] == [scheduled.context.span_id]


def test_a_cancelled_agentic_search_is_counted_not_failed(spans):
    from types import SimpleNamespace

    from agenticwf.kb.runtime.agentic.engine import Run

    async def hang(current):
        await asyncio.Event().wait()

    run = SimpleNamespace(run_id="r1", preset=SimpleNamespace(name="cancel-test", engine="loop"), output="answer",
                          scope=SimpleNamespace(kb_ids=[]), model_id="m1", stopped=None, _run_main=hang,
                          usage={"llm_calls": 1, "input_tokens": 5, "output_tokens": 2})

    async def scenario():
        task = asyncio.create_task(Run._main(run))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    (agentic,) = [s for s in spans() if s.name == "kb.agentic_search"]
    assert agentic.attributes["agenticwf.kb.agentic.stopped"] == "cancelled"
    assert agentic.attributes["agenticwf.kb.agentic.llm_calls"] == 1
    assert agentic.status.is_ok and not agentic.events      # not an error, no exception recorded
    assert [p.attributes["agenticwf.kb.agentic.stopped"] for p in points("agenticwf.kb.agentic.duration")
            if p.attributes["agenticwf.kb.agentic.preset"] == "cancel-test"] == ["cancelled"]


def test_an_exception_still_fails_a_span(spans):
    with pytest.raises(ValueError):
        with telemetry.span("broken"):
            raise ValueError("bad")
    (broken,) = spans()
    assert not broken.status.is_ok and broken.events[0].name == "exception"
