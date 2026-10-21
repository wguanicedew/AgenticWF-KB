"""Planning tools (docs/agentic-search.md, Planning tools): thinking, todo_write. Nothing they
record outlives the run."""

from __future__ import annotations

from kb_runtime.agentic.tools import Tool, ToolContext, ToolError

MAX_TODOS = 30
MARKS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


async def thinking(tc: ToolContext, args: dict) -> str:
    thought = str(args.get("thought") or "").strip()
    if not thought:
        raise ToolError("thought is required")
    await tc.emit({"response_type": "thinking", "content": thought})
    return "Thought recorded. Continue with the next step."


async def todo_write(tc: ToolContext, args: dict) -> str:
    todos = args.get("todos")
    if not isinstance(todos, list):
        raise ToolError("todos must be a list of {content, status}")
    out = []
    for todo in todos[:MAX_TODOS]:
        if not isinstance(todo, dict) or not str(todo.get("content") or "").strip():
            raise ToolError("each todo needs content")
        status = todo.get("status") or "pending"
        if status not in MARKS:
            raise ToolError("status must be pending, in_progress or completed")
        out.append({"content": str(todo["content"]).strip()[:300], "status": status})
    tc.todos[:] = out
    return "\n".join(f"{MARKS[t['status']]} {t['content']}" for t in out) or "(empty list)"


TOOLS = [
    Tool("thinking", "Think out loud: plan the searches, check what evidence is missing, reconcile sources.",
         {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]},
         thinking, group="planning"),
    Tool("todo_write", "Replace the task list for a multi-part question (at most 30 items).",
         {"type": "object", "properties": {"todos": {"type": "array", "items": {
             "type": "object", "properties": {"content": {"type": "string"},
                                              "status": {"type": "string",
                                                         "enum": ["pending", "in_progress", "completed"]}},
             "required": ["content", "status"]}}}, "required": ["todos"]},
         todo_write, group="planning"),
]
