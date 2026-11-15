"""Policies: things that render a conversation to text, tokenize text, and
map prompt tokens to sampled completions (given).

`TinkerPolicy` wraps a Tinker sampling client (used for real training runs).
`CompletionsServerPolicy` samples from an OpenAI-compatible `/v1/completions`
server such as vLLM or SGLang (used to evaluate an exported adapter).
`MockPolicy` is a deterministic scripted policy used by the unit tests and
handy for debugging the data pipeline without an API key.

A policy does not own the conversation. The agent keeps the list of chat
messages and the running token context; it asks the policy to `render` the
messages through the model's chat template, to `encode` text it has never
seen tokenized, and to `complete` a token prompt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from scottyrl.config import resolve_chat_template


@dataclass
class Completion:
    text: str
    tokens: list[int]
    logprobs: list[float]


@runtime_checkable
class Policy(Protocol):
    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """Render `[{"role": ..., "content": ...}, ...]` to prompt text."""
        ...

    def encode(self, text: str) -> list[int]: ...

    async def complete(self, prompt_tokens: list[int]) -> Completion: ...


def load_chat_template(path: str | Path) -> str:
    """Read a chat template file (relative paths are taken from the repo root)."""
    return resolve_chat_template(path).read_text()


class MockPolicy:
    """Scripted policy for tests. Tokens are UTF-8 bytes of the text.

    `script` is a list of completion strings returned in order; the last one
    repeats once the script is exhausted. `render` uses a minimal
    `<role>\\ncontent\\n` layout so tests need no tokenizer or template.
    """

    def __init__(self, script: Sequence[str], logprob: float = -0.5):
        if not script:
            raise ValueError("script must be non-empty")
        self.script = list(script)
        self.logprob = logprob
        self._i = 0

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        text = "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>\n"
        return text

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    async def complete(self, prompt_tokens: list[int]) -> Completion:
        text = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        tokens = self.encode(text)
        return Completion(text=text, tokens=tokens, logprobs=[self.logprob] * len(tokens))


class ChatTemplatePolicy:
    """What the two real policies share: a HuggingFace tokenizer, the chat
    template and its knobs, and the stop token ids.

    `chat_template` is the template text (see `load_chat_template`), not a
    path. `enable_thinking` and `preserve_thinking_history` are forwarded to
    the template on every `render`; see `ModelConfig` for what they do.
    `stop_sequences` are token names; sampling stops when the model emits one
    of them, and the stop token stays in the returned `Completion.text` so the
    agent can see where the model ended its turn.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens: int,
        temperature: float,
        chat_template: str,
        enable_thinking: bool = False,
        preserve_thinking_history: bool = True,
        stop_sequences: Sequence[str] = ("<|im_end|>", "<|endoftext|>"),
    ):
        if not chat_template:
            raise ValueError("chat_template must be the template text (see load_chat_template)")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
        self.preserve_thinking_history = preserve_thinking_history
        self.stop_sequences = list(stop_sequences)
        self.stop_token_ids = self._stop_token_ids(tokenizer, self.stop_sequences)

    @staticmethod
    def _stop_token_ids(tokenizer, stop_sequences: list[str]) -> list[int]:
        """Token ids of the stop sequences, skipping names the tokenizer does
        not know (it returns None or its unk id for those)."""
        unk = getattr(tokenizer, "unk_token_id", None)
        ids: list[int] = []
        for name in stop_sequences:
            token_id = tokenizer.convert_tokens_to_ids(name)
            if token_id is None or (unk is not None and token_id == unk):
                continue
            ids.append(int(token_id))
        return ids

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
            preserve_thinking_history=self.preserve_thinking_history,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class TinkerPolicy(ChatTemplatePolicy):
    """Wraps a Tinker sampling client + a HuggingFace tokenizer.

    The rendering arguments are those of `ChatTemplatePolicy`. `complete`
    returns the sampled token ids and their logprobs, which the trainers
    need for the importance ratio.
    """

    def __init__(
        self,
        sampling_client,
        tokenizer,
        max_tokens: int,
        temperature: float,
        chat_template: str,
        enable_thinking: bool = False,
        preserve_thinking_history: bool = True,
        stop_sequences: Sequence[str] = ("<|im_end|>", "<|endoftext|>"),
    ):
        super().__init__(
            tokenizer,
            max_tokens,
            temperature,
            chat_template,
            enable_thinking=enable_thinking,
            preserve_thinking_history=preserve_thinking_history,
            stop_sequences=stop_sequences,
        )
        self.sampling_client = sampling_client

    async def complete(self, prompt_tokens: list[int]) -> Completion:
        import tinker  # lazy: only needed for real runs

        result = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            # top_p and top_k are spelled out so that the sampler draws from
            # the full, untruncated distribution: the logprobs it returns are
            # then the ones the loss functions assume.
            sampling_params=tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                top_k=-1,
                stop=self.stop_token_ids or None,
            ),
        )
        seq = result.sequences[0]
        tokens = list(seq.tokens)
        logprobs = [float(lp) for lp in seq.logprobs]
        # Keep special tokens: a trailing <|im_end|> tells the agent the model
        # finished its turn (and must be stripped before the turn is re-rendered).
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return Completion(text=text, tokens=tokens, logprobs=logprobs)


class CompletionsServerPolicy(ChatTemplatePolicy):
    """Samples from an OpenAI-compatible `/v1/completions` server (vLLM, SGLang).

    This is how graders evaluate the Part 5 adapter you export to the
    Hugging Face Hub (`scripts/export_checkpoint.py`), and how you can
    evaluate it yourself before submitting. Serve the base model with the
    adapter, then run `scripts/evaluate_checkpoint.py --server --model`:

      vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 32 \
          --lora-modules a3=<hf_model_id>
      # SGLang names the adapter in the model field as base:adapter:
      python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
          --enable-lora --lora-paths a3=<hf_model_id> --max-lora-rank 32

    `base_url` is the server's API root (`http://localhost:8000/v1`) and
    `model` the name the server knows the adapter by (`a3` for the vLLM
    line above, `Qwen/Qwen3.5-4B:a3` for SGLang). `api_key`, when the
    server requires one, defaults to `OPENAI_API_KEY` or `VLLM_API_KEY`
    from the environment. The rendering arguments are those of
    `ChatTemplatePolicy`; use the same config as the training run so
    prompts render exactly as they did in training.

    Timeouts and retries (changed Nov 14 after the first Part 5 evaluation
    stalled). A sampling request takes a few seconds, so `timeout` (seconds
    to wait for the response, default 120, under the 150 s limit Modal puts
    on a web request) and `connect_timeout` (seconds to open a connection,
    default 10) are short: with the earlier single 600 s timeout, a request
    lost on the way held its episode for ten minutes, and a few percent of
    lost requests were enough to leave almost every episode waiting. A
    request is retried up to `max_retries` times, with a growing pause,
    after a timeout, after a dropped, refused or reset connection, and after
    a 408, 429 or 5xx answer; the earlier code retried timeouts only, so one
    dropped connection ended an episode. Sampling is safe to repeat: nothing
    on the server depends on a request having run once. `stats` counts what
    happened.

    `complete` posts `prompt` as the token ids (no re-tokenization of the
    rendered text), `max_tokens`, `temperature`, `top_p: 1.0`, `top_k: -1`,
    `stop_token_ids`, `skip_special_tokens: false` and
    `include_stop_str_in_output: true` (`no_stop_trim: true` for SGLang),
    so the stop token stays in the text
    like it does with `TinkerPolicy`. When a server reports `finish_reason
    "stop"` without the stop token in the text, the token it names in
    `stop_reason` (or the tokenizer's eos token) is appended. The returned
    `Completion.tokens` are `tokenizer.encode(text)`, not the server's
    sampled ids, and the logprobs are zeros: evaluation only needs the text,
    and the agent only needs tokens that re-render to it.

    Request fields verified on 2026-09-14 against vLLM main
    (`vllm/entrypoints/openai/completion/protocol.py`, commit e6b4e47d):
    `CompletionRequest.prompt` is `list[int] | list[list[int]] | str |
    list[str]`; `stop_token_ids: list[int] | None = []`;
    `skip_special_tokens: bool = True`; `include_stop_str_in_output: bool =
    False`; `return_tokens_as_token_ids` exists but only changes how logprob
    tokens are spelled, so it is not used. SGLang main
    (`python/sglang/srt/entrypoints/openai/protocol.py`): `prompt` is
    `Union[List[int], List[List[int]], str, List[str]]`, `stop_token_ids`
    and `skip_special_tokens` as above, and `no_stop_trim: bool = False`
    plays the role of `include_stop_str_in_output`. Both servers ignore
    request fields they do not define.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer,
        max_tokens: int,
        temperature: float,
        chat_template: str,
        enable_thinking: bool = False,
        preserve_thinking_history: bool = True,
        stop_sequences: Sequence[str] = ("<|im_end|>", "<|endoftext|>"),
        api_key: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        max_retries: int = 6,
    ):
        super().__init__(
            tokenizer,
            max_tokens,
            temperature,
            chat_template,
            enable_thinking=enable_thinking,
            preserve_thinking_history=preserve_thinking_history,
            stop_sequences=stop_sequences,
        )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self._client = None  # httpx.AsyncClient, created on first use
        # What the sampling requests did, for progress lines and diagnosis:
        # `requests` answered, `retries` taken, `failed` given up on,
        # `latency_s` total and `latency_max_s` of the answered ones (the
        # attempt that succeeded, without earlier attempts or backoff), and
        # `errors` counting each retried cause by name.
        self.stats: dict = {
            "requests": 0, "retries": 0, "failed": 0,
            "latency_s": 0.0, "latency_max_s": 0.0, "errors": {},
        }

    def _http(self):
        """The shared async HTTP client (one connection pool per policy)."""
        import httpx  # lazy: only needed to talk to a server

        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, follow_redirects=True,
                # A lost request must cost seconds, not minutes (see the
                # class docstring). No pool limit: the caller's concurrency
                # is the limit, and idle connections are kept for reuse so a
                # steady run does not keep opening new ones.
                timeout=httpx.Timeout(
                    self.timeout, connect=self.connect_timeout, write=30.0, pool=None
                ),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client (optional; connections close with the loop)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def request_body(self, prompt_tokens: list[int]) -> dict:
        """The JSON body `complete` posts for `prompt_tokens`."""
        return {
            "model": self.model,
            "prompt": list(prompt_tokens),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # No nucleus or top-k truncation, as in training. Set explicitly
            # because servers fall back to the model's generation_config.json
            # for the fields a request leaves out.
            "top_p": 1.0,
            "top_k": -1,
            "stop_token_ids": list(self.stop_token_ids),
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,  # vLLM
            "no_stop_trim": True,  # SGLang
        }

    # Statuses worth another try: the server or the proxy in front of it was
    # busy or restarting. Anything else (400, 404, ...) is a real error.
    RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

    async def complete(self, prompt_tokens: list[int]) -> Completion:
        import httpx

        body = self.request_body(prompt_tokens)
        log = logging.getLogger(__name__)
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = await self._http().post("/completions", json=body)
                if response.status_code not in self.RETRY_STATUSES:
                    break
                cause = f"HTTP {response.status_code}"
                error: Exception = httpx.HTTPStatusError(
                    f"{cause} from the sampling server", request=response.request,
                    response=response,
                )
            except httpx.TransportError as e:
                # Timeouts and dropped, refused or reset connections alike.
                cause, error = type(e).__name__, e
            waited = time.monotonic() - started
            self.stats["errors"][cause] = self.stats["errors"].get(cause, 0) + 1
            if attempt == attempts - 1:
                self.stats["failed"] += 1
                raise error
            self.stats["retries"] += 1
            log.warning(
                "Sampling request failed after %.1f s (%s); retrying (%d/%d)",
                waited, cause, attempt + 1, self.max_retries,
            )
            await asyncio.sleep(min(2 ** attempt, 15))
        response.raise_for_status()
        latency = time.monotonic() - started
        self.stats["requests"] += 1
        self.stats["latency_s"] += latency
        self.stats["latency_max_s"] = max(self.stats["latency_max_s"], latency)
        choice = response.json()["choices"][0]
        text = choice.get("text") or ""
        if choice.get("finish_reason") == "stop" and not self._ends_with_stop(text):
            text += self._stop_text(choice.get("stop_reason"))
        tokens = self.encode(text)
        return Completion(text=text, tokens=tokens, logprobs=[0.0] * len(tokens))

    def _ends_with_stop(self, text: str) -> bool:
        return any(text.endswith(name) for name in self.stop_sequences)

    def _stop_text(self, stop_reason) -> str:
        """The stop token a server trimmed: the id it names in `stop_reason`
        (vLLM) when that is one of ours, else the tokenizer's eos token."""
        if isinstance(stop_reason, int) and stop_reason in self.stop_token_ids:
            return self.tokenizer.decode([stop_reason], skip_special_tokens=False)
        if isinstance(stop_reason, str) and stop_reason in self.stop_sequences:
            return stop_reason
        return getattr(self.tokenizer, "eos_token", None) or ""
