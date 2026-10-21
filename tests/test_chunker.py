"""The chunker's own golden cases (docs/ingestion.md, Chunker)."""

from kb_runtime.ingestion.chunker import chunk, heuristic_titles, markdown_headings, resolve_strategy


def check_offsets(text, pieces):
    """Every chunk is the source substring at its offsets, plus a prepended table header."""
    for p in pieces:
        assert p.content[p.content_offset:] == text[p.start:p.end]
        assert len(p.content) - p.content_offset <= max(len(text[p.start:p.end]), 1)


def reconstruct(text, pieces):
    """Without overlap, the chunks cover the text in order (only whitespace between them)."""
    pos = 0
    for p in pieces:
        assert text[pos:p.start].strip() == "", repr(text[pos:p.start])
        pos = p.end
    assert text[pos:].strip() == ""


def test_short_text_is_one_chunk():
    pieces = chunk("Hello world.", {"strategy": "legacy"})
    assert [(p.content, p.start, p.end) for p in pieces] == [("Hello world.", 0, 12)]


def test_paragraphs_merge_up_to_size_and_reconstruct():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(20))
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 300, "chunk_overlap": 0})
    assert len(pieces) > 3 and all(len(p.content) <= 300 for p in pieces)
    check_offsets(text, pieces)
    reconstruct(text, pieces)


def test_overlap_repeats_trailing_units():
    text = "\n".join(f"line {i:03d} " + "x" * 30 for i in range(30))
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 200, "chunk_overlap": 60})
    check_offsets(text, pieces)
    assert any(pieces[i + 1].start < pieces[i].end for i in range(len(pieces) - 1))   # overlap


def test_code_fences_and_math_are_never_split():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```"
    math = "$$\n" + " + ".join(f"a_{i}" for i in range(60)) + "\n$$"
    text = "Intro text.\n\n" + code + "\n\nMiddle.\n\n" + math + "\n\nEnd."
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 120, "chunk_overlap": 0})
    check_offsets(text, pieces)
    assert any(code in p.content for p in pieces)
    assert any(math in p.content for p in pieces)


def test_links_and_images_are_kept_whole():
    text = ("word " * 18) + "[a link](https://example.org/a/very/long/path)" + " tail" + \
        " ![figure](images/fig1.png)" + " end"
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 100, "chunk_overlap": 0})
    check_offsets(text, pieces)
    assert any("[a link](https://example.org/a/very/long/path)" in p.content for p in pieces)
    assert any("![figure](images/fig1.png)" in p.content for p in pieces)


def test_table_header_repeats_in_continuation_chunks():
    header = "| name | value |\n|---|---|"
    rows = "\n".join(f"| item{i} | {i * 10} |" for i in range(40))
    text = "Table:\n\n" + header + "\n" + rows + "\n\nAfter."
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 150, "chunk_overlap": 0})
    check_offsets(text, pieces)
    table_chunks = [p for p in pieces if "| item" in p.content]
    assert len(table_chunks) > 2
    for p in table_chunks:
        assert p.content.lstrip().startswith("| name | value |") or p.content.startswith("Table")
    assert sum(1 for p in pieces if p.content_offset) == len(table_chunks) - 1


def test_cjk_text_splits_on_full_stops():
    text = "。".join("这是一个关于粒子探测器的句子" * 3 for _ in range(30)) + "。"
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 100, "chunk_overlap": 0})
    check_offsets(text, pieces)
    assert all(len(p.content) <= 100 for p in pieces) and len(pieces) > 5


def test_unbreakable_text_is_cut_at_size():
    text = "a" * 1000
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 300, "chunk_overlap": 0})
    check_offsets(text, pieces)
    reconstruct(text, pieces)
    assert all(len(p.content) <= 300 for p in pieces)


def test_heading_breadcrumbs():
    text = ("# Guide\n\nIntro to the guide, long enough to matter here.\n\n"
            "## Install\n\nHow to install it, with several words in the text.\n\n"
            "### Linux\n\nRun the package manager and wait for it to finish properly.\n\n"
            "## Use\n\nUse it daily, with care and attention to detail always.\n")
    pieces = chunk(text, {"strategy": "heading", "chunk_size": 512})
    headers = [p.context_header for p in pieces]
    assert headers == ["Guide", "Guide > Install", "Guide > Install > Linux", "Guide > Use"]
    check_offsets(text, pieces)
    reconstruct(text, pieces)


def test_headings_inside_code_are_ignored():
    text = "# Real\n\ntext\n\n```\n# not a heading\n```\n\n# Second\n\nmore"
    assert [h[2] for h in markdown_headings(text)] == ["Real", "Second"]


def test_heuristic_titles_and_auto():
    text = ("1 Introduction\nSome text about the detector and its purpose in the experiment.\n"
            "1.1 Scope\nWhat the document covers and what it does not cover at all.\n"
            "2 Design\nThe design of the system with all of its parts explained.\n")
    titles = heuristic_titles(text)
    assert [(lvl, t.split()[0]) for _, lvl, t in titles] == [(1, "1"), (2, "1.1"), (1, "2")]
    assert resolve_strategy(text, "auto") == "heuristic"
    pieces = chunk(text, {"strategy": "auto"})
    assert [p.context_header for p in pieces][:2] == ["1 Introduction", "1 Introduction > 1.1 Scope"]
    assert resolve_strategy("just a plain paragraph", "auto") == "legacy"
    assert resolve_strategy("x", "nonsense") == "legacy"


def test_cjk_chapter_titles():
    text = "第一章 总则\n内容甲。\n第二章 设计\n内容乙。\n"
    assert len(heuristic_titles(text)) == 2


def test_token_limit_clamps_size():
    text = "word " * 400
    pieces = chunk(text, {"strategy": "legacy", "chunk_size": 1000, "token_limit": 20, "chunk_overlap": 0})
    assert all(len(p.content) <= 60 for p in pieces)


def test_parent_child_offsets():
    text = "\n\n".join(f"Paragraph {i}. " + "content " * 40 for i in range(12))
    pieces = chunk(text, {"strategy": "legacy", "enable_parent_child": True, "parent_chunk_size": 1200,
                          "child_chunk_size": 300})
    parents = [p for p in pieces if p.chunk_type == "parent_text"]
    children = [p for p in pieces if p.chunk_type == "text"]
    assert parents and children and all(c.parent is not None for c in children)
    for c in children:
        parent = pieces[c.parent]
        assert parent.start <= c.start and c.end <= parent.end
        assert len(c.content) <= 300
    check_offsets(text, pieces)
