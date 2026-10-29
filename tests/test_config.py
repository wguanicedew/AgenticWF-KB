"""kb.cfg parsing (core/common/config.py): comments after values."""

from agenticwf.kb.core.common.config import new_parser


def test_comment_after_value_is_stripped(tmp_path):
    path = tmp_path / "kb.cfg"
    path.write_text("[embeddings]\n"
                    "provider = vllm                 # OpenAI-compatible\n"
                    "base_url = https://example.org/v1#frag\n"
                    "# a whole-line comment\n"
                    "model = e5-large-v2\n")
    parser = new_parser()
    parser.read(path)
    section = parser["embeddings"]
    assert section["provider"] == "vllm"
    assert section["base_url"] == "https://example.org/v1#frag"   # no whitespace before '#': kept
    assert section["model"] == "e5-large-v2"
