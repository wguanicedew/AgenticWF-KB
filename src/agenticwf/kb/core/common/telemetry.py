"""
OpenTelemetry tracing and metrics (docs/deployment.md, Tracing and metrics): a span per HTTP
request, search, agentic search, model call (chat, embeddings, rerank) and task, and counters and
histograms of the same. Both go over OTLP/HTTP to a collector (Jaeger for traces, Prometheus or an
OpenTelemetry Collector for metrics), the same one as the agent's.

    agent ── traceparent ──► kb API ─┐
                    kb worker process ┴─ OTLP/HTTP ─► collector ─► UI

A request from the agent carries its `traceparent` (also over the Unix socket), so the KB's spans
sit inside the agent's trace, under its `execute_tool kb_search` span.

Needs the `otel` extra (`pip install 'agenticwf-kb-server[otel]'`; the container installs it), and:

    traces    OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    metrics   OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, or OTEL_EXPORTER_OTLP_ENDPOINT with
              OTEL_METRICS_EXPORTER=otlp (Jaeger takes no metrics: off by default)

The rest is the SDK's own environment: OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES,
OTEL_TRACES_SAMPLER, OTEL_EXPORTER_OTLP_HEADERS, OTEL_METRIC_EXPORT_INTERVAL, OTEL_SDK_DISABLED, ...

Spans carry ids, counts, model names, token counts and durations, never queries, document content,
answers or who asked. Model calls follow the OpenTelemetry GenAI conventions (gen_ai.*).

A task runs after the request that queued it, maybe in another process: the task's payload carries
that request's span context (`_trace`, from `traceparent()`), and its span links to it. A scheduled
task has no request: the scheduler's `schedule <name>` span, around its enqueue, stands in for one.

Everything here is safe to call anywhere: without the SDK, or before `setup()`, it does nothing.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from agenticwf.kb.core.common.logging import get_logger

try:
    from opentelemetry import metrics as _metrics
    from opentelemetry import trace as _trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
except ImportError:   # the `otel` extra is not installed
    _metrics = _trace = None

LOG = get_logger(__name__)

TRACER = "agenticwf.kb"
_tracing = _metering = False


def _env(name: str) -> str:
    return os.environ.get(name, "")


def enabled() -> bool:
    """Traces are asked for: an OTLP endpoint is set and the SDK is not disabled."""
    if _env("OTEL_SDK_DISABLED").lower() == "true" or _env("OTEL_TRACES_EXPORTER") == "none":
        return False
    return bool(_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or _env("OTEL_EXPORTER_OTLP_ENDPOINT"))


def metrics_enabled() -> bool:
    """Metrics are asked for: their own endpoint, or the shared one with OTEL_METRICS_EXPORTER=otlp."""
    if _env("OTEL_SDK_DISABLED").lower() == "true" or _env("OTEL_METRICS_EXPORTER") == "none":
        return False
    return bool(_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
                or (_env("OTEL_EXPORTER_OTLP_ENDPOINT") and _env("OTEL_METRICS_EXPORTER") == "otlp"))


def setup(service: str) -> bool:
    """Install the process's tracer and meter providers, exporting over OTLP. `service` is the
    service.name unless OTEL_SERVICE_NAME says otherwise. Once per process; True when either is on."""
    global _tracing, _metering
    if _tracing or _metering:
        return True
    traces, metrics = enabled(), metrics_enabled()
    if not (traces or metrics):
        return False
    try:
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        LOG.warning("an OTLP endpoint is set but the OpenTelemetry SDK is not installed "
                    "(pip install 'agenticwf-kb-server[otel]'); no traces or metrics")
        return False
    resource = Resource.create({} if _env("OTEL_SERVICE_NAME") else {"service.name": service})
    if traces:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=resource)   # flushed at exit
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        _trace.set_tracer_provider(provider)
        _tracing = True
    if metrics:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(OTLPMetricExporter())   # every OTEL_METRIC_EXPORT_INTERVAL (60 s)
        _metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
        _metering = True
    try:   # outgoing calls: model servers, URL import, web search
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except ImportError:
        pass
    LOG.info("OpenTelemetry as %s: traces %s, metrics %s", _env("OTEL_SERVICE_NAME") or service,
             "on" if _tracing else "off", "on" if _metering else "off")
    return True


def instrument_app(app: Any) -> None:
    """A span and request metrics per HTTP request of a FastAPI app (not the health check)."""
    if not (_tracing or _metering):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    # Not a span per ASGI message: SSE streams (events, agentic search frames) send one per frame.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health$", exclude_spans=["receive", "send"])


# --- traces --------------------------------------------------------------------------------------
class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: dict[str, Any]) -> None: ...

    def set_status(self, *args: Any, **kwargs: Any) -> None: ...

    def record_exception(self, *args: Any, **kwargs: Any) -> None: ...

    def is_recording(self) -> bool:
        return False


_NO_SPAN = _NoSpan()


@contextmanager
def span(name: str, links: Iterable[str | None] = (), **attributes: Any) -> Iterator[Any]:
    """A child of the current span (or a new trace), linked to the spans named by `links`
    (traceparent strings). An exception that leaves it marks it failed, a cancellation does not
    (a caller that went away, a shutdown). Attribute names use `__` for dots
    (gen_ai__request__model=...); None values are left out."""
    if _trace is None:
        yield _NO_SPAN
        return
    linked = [_trace.Link(c) for c in map(_span_context, links) if c is not None]
    with _trace.get_tracer(TRACER).start_as_current_span(name, attributes=_attributes(attributes), links=linked,
                                                         record_exception=False,
                                                         set_status_on_exception=False) as current:
        try:
            yield current
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            current.record_exception(exc)
            current.set_status(_trace.Status(_trace.StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise


def annotate(current: Any, **attributes: Any) -> None:
    """Add attributes to a span from `span()`, named as there."""
    if current.is_recording():
        current.set_attributes(_attributes(attributes))


def fail(current: Any, description: str) -> None:
    """Mark a span failed without an exception (a search that fell back, a tool's error)."""
    if _trace is not None and current.is_recording():
        current.set_status(_trace.Status(_trace.StatusCode.ERROR, description))


def traceparent() -> str | None:
    """The current span as a W3C traceparent, to link to later (a task's origin); None outside one."""
    if _trace is None or not _trace.get_current_span().get_span_context().is_valid:
        return None
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent")


def _span_context(value: str | None) -> Any:
    if not value:
        return None
    context = _trace.get_current_span(TraceContextTextMapPropagator().extract({"traceparent": value})).get_span_context()
    return context if context.is_valid else None


def _attributes(given: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("__", "."): value for key, value in given.items() if value is not None}


# --- metrics -------------------------------------------------------------------------------------
# Instruments on the global meter: no-ops until setup() installs a provider (the API's proxies).
_SECONDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900]
_TOKENS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

if _metrics is not None:
    _meter = _metrics.get_meter(TRACER)
    _model_duration = _meter.create_histogram(
        "gen_ai.client.operation.duration", unit="s", description="Model calls (chat, embeddings, rerank)",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _model_tokens = _meter.create_histogram(
        "gen_ai.client.token.usage", unit="{token}", description="Tokens per model call, by gen_ai.token.type",
        explicit_bucket_boundaries_advisory=_TOKENS)
    _search_duration = _meter.create_histogram(
        "agenticwf.kb.search.duration", unit="s", description="Hybrid searches, end to end",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _agentic_duration = _meter.create_histogram(
        "agenticwf.kb.agentic.duration", unit="s", description="Agentic searches, by preset and how they stopped",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _task_duration = _meter.create_histogram(
        "agenticwf.kb.task.duration", unit="s", description="Tasks (ingestion, enrichment, wiki, ...), by type and outcome",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _tasks_running = _meter.create_up_down_counter(
        "agenticwf.kb.tasks.running", unit="{task}", description="Tasks running in this process")


@contextmanager
def model_call(operation: str, provider: str, model: str) -> Iterator[Any]:
    """A span and the duration of one model call: `operation` is chat, embeddings or rerank.
    Report tokens with `model_tokens()` on the span it yields."""
    started = time.monotonic()
    error = None
    try:
        with span(f"{operation} {model}", gen_ai__operation__name=operation, gen_ai__provider__name=provider,
                  gen_ai__request__model=model) as current:
            yield current
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        if _metrics is not None:
            _model_duration.record(time.monotonic() - started, _attributes({
                "gen_ai__operation__name": operation, "gen_ai__provider__name": provider,
                "gen_ai__request__model": model, "error__type": error}))


def model_tokens(current: Any, operation: str, provider: str, model: str, input_tokens: int | None,
                 output_tokens: int | None = None) -> None:
    """The tokens a model call reported (OpenAI-compatible `usage`), on its span and in the metrics."""
    annotate(current, gen_ai__usage__input_tokens=input_tokens, gen_ai__usage__output_tokens=output_tokens)
    if _metrics is None:
        return
    attributes = {"gen_ai.operation.name": operation, "gen_ai.provider.name": provider, "gen_ai.request.model": model}
    if input_tokens is not None:
        _model_tokens.record(input_tokens, {**attributes, "gen_ai.token.type": "input"})
    if output_tokens is not None:
        _model_tokens.record(output_tokens, {**attributes, "gen_ai.token.type": "output"})


def record_search(seconds: float, *, reranked: bool) -> None:
    if _metrics is not None:
        _search_duration.record(seconds, {"agenticwf.kb.search.reranked": reranked})


def record_agentic(seconds: float, *, preset: str, stopped: str) -> None:
    """`preset`: a configured preset's name (bounded by the configuration)."""
    if _metrics is not None:
        _agentic_duration.record(seconds, {"agenticwf.kb.agentic.preset": preset, "agenticwf.kb.agentic.stopped": stopped})


@contextmanager
def measure_task(task_type: str, queue: str) -> Iterator[dict[str, str]]:
    """Time a task and count it while it runs. Set `outcome` in the dict it yields (done, retry,
    failed); an exception that leaves it is `failed`, a stop (shutdown requeues it) `cancelled`."""
    if _metrics is None:
        yield {}
        return
    found = {"outcome": "done"}
    started = time.monotonic()
    running = {"agenticwf.kb.task.type": task_type, "agenticwf.kb.task.queue": queue}
    _tasks_running.add(1, running)
    try:
        yield found
    except asyncio.CancelledError:
        found["outcome"] = "cancelled"
        raise
    except BaseException:
        found["outcome"] = "failed"
        raise
    finally:
        _tasks_running.add(-1, running)
        _task_duration.record(time.monotonic() - started, {**running, "agenticwf.kb.task.outcome": found["outcome"]})
