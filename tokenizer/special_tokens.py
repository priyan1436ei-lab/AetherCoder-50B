"""Special tokens definition for Code LLM, FIM, Repo-level parsing, and Agent Tool-Use."""

from typing import Dict, List

# Core Sequence Tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# Fill-In-the-Middle (FIM) Tokens (OpenAI / StarCoder standard)
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
EOT_TOKEN = "<|endoftext|>"

# Repository-Aware Context Tokens
REPO_NAME = "<repo_name>"
FILE_PATH = "<file_path>"
FILE_SEP = "<file_sep>"
COMMIT_MSG = "<commit_msg>"

# Agent & Tool-Use Tokens
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"
THOUGHT_START = "<thought>"
THOUGHT_END = "</thought>"
PLAN_START = "<plan>"
PLAN_END = "</plan>"

SPECIAL_TOKENS_LIST: List[str] = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    EOT_TOKEN,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_MIDDLE,
    FIM_PAD,
    REPO_NAME,
    FILE_PATH,
    FILE_SEP,
    COMMIT_MSG,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESULT_START,
    TOOL_RESULT_END,
    THOUGHT_START,
    THOUGHT_END,
    PLAN_START,
    PLAN_END,
]

SPECIAL_TOKENS_DICT: Dict[str, str] = {token: token for token in SPECIAL_TOKENS_LIST}
