"""
prompt.py
=========

Prompt formatting and history-item truncation helpers for the six generation
tasks covered by this release:

    LaMP_4                              news headline generation
    LaMP_5                              paper title generation
    LaMP_7                              tweet paraphrasing
    LongLaMP_abstract_generation_user   paper abstract generation
    LongLaMP_topic_writing_user         reddit topic writing
    LongLaMP_product_review_user        product review generation

Profile item formats produced by ``precompute.py`` (the gold target field is
always placed last so it can be preserved as a suffix when the body has to be
truncated to fit the token budget):

    LaMP_4                              "article: {text} headline: {title}"
    LaMP_5                              "abstract: {text} title: {title}"
    LaMP_7                              "{tweet}"                (no suffix)
    LongLaMP_abstract_generation_user   "abstract: {abstract} title: {title}"
    LongLaMP_topic_writing_user         "content: {content} summary: {summary}"
    LongLaMP_product_review_user        "description: {d} summary: {s} review: {r} rating: {x}"

In the prompt past items are exposed as

    ### User History:
    - <item>
    - ...
    ### User Instruction:
    <task instruction>
    <answer cue>
"""
import re
from typing import List, Optional


# ============================================================
# BM25 helpers
# For most tasks the query and the bank item share their wording, so the
# naive identity pass is enough. LongLaMP product review is the exception:
# the query carries the product description and the review summary, so both
# sides are reduced to those two fields before scoring.
# ============================================================
def extract_bm25_query(task_name: str, input_text: str) -> str:
    """Reduce the query to the fields that the bank items can match against."""
    if task_name == "LongLaMP_product_review_user":
        parts = []
        m_d = re.search(
            r'description\s*"(.+?)"\s*\.\s+The summary of the review text is',
            input_text, re.IGNORECASE | re.DOTALL,
        )
        if m_d:
            parts.append(m_d.group(1))
        m_s = re.search(
            r'summary of the review text is\s*"(.+?)"\s*\.?\s*$',
            input_text, re.IGNORECASE | re.DOTALL,
        )
        if m_s:
            parts.append(m_s.group(1))
        if parts:
            return " ".join(parts)
    return input_text


def extract_bm25_bank_text(task_name: str, item: str) -> str:
    """Reduce a bank item to the fields the query can match against."""
    if task_name == "LongLaMP_product_review_user":
        # "description: D summary: S review: R rating: X" -> description + summary
        m = re.search(
            r'\bdescription:\s*(.+?)\s+summary:\s*(.+?)\s+review:\s+',
            item, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return f"{m.group(1)} {m.group(2)}"
        # Items whose product description is empty: the saved text starts at
        # "summary:" so the pattern above does not match. 22 of 346,026 items in
        # the paper's cache. Fall back to the description-only form.
        m_old = re.search(
            r'\bdescription:\s*(.+?)\s+rating:\s*[\d.]+\s*$',
            item, re.IGNORECASE | re.DOTALL,
        )
        if m_old:
            return m_old.group(1)
    return item


# ============================================================
# Input parsing
# ============================================================
def extract_article_text(input_text: str) -> str:
    """LaMP-4: the article body out of the raw benchmark input."""
    text = str(input_text).strip()
    m = re.search(r"article\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"article\s*[:.]\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    return m2.group(1).strip() if m2 else text


def extract_abstract_text(input_text: str) -> str:
    """LaMP-5: the abstract body out of the raw benchmark input."""
    text = str(input_text).strip()
    m = re.search(r"abstract\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"paper\s*[:.]\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    return m2.group(1).strip() if m2 else text


# ============================================================
# History item helpers
# ============================================================
def _get_body_and_suffix(item: str, task_name: str):
    """Split an item into ``(body, suffix)``.

    ``suffix`` is the gold target field (plus, for product review, the review
    text and rating). It is preserved verbatim; only ``body`` is truncated.
    """
    if task_name == "LaMP_4":
        # "article: {text} headline: {title}"
        m = re.search(r"\bheadline:\s*(.+?)\s*$", item)
        if m:
            return item[:m.start()].strip(), f" headline: {m.group(1)}"
        return item, ""

    if task_name in ("LaMP_5", "LongLaMP_abstract_generation_user"):
        # "abstract: {text} title: {title}"
        m = re.search(r"\btitle:\s*(.+?)\s*$", item)
        if m:
            return item[:m.start()].strip(), f" title: {m.group(1)}"
        return item, ""

    if task_name == "LongLaMP_topic_writing_user":
        # "content: {content} summary: {summary}"
        m = re.search(r"\bsummary:\s*(.+?)\s*$", item)
        if m:
            return item[:m.start()].strip(), f" summary: {m.group(1)}"
        return item, ""

    if task_name == "LongLaMP_product_review_user":
        # "description: D summary: S review: R rating: X"
        # body = description, suffix = summary + review + rating (all preserved),
        # so truncation eats the product description and never the user's own text.
        m = re.search(
            r'^(description:\s*.+?)(\s+summary:\s+.+)$',
            item, re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip(), m.group(2)
        # Items whose product description is empty start at "summary:", so the
        # pattern above does not match: keep the trailing rating as the suffix.
        m_old = re.search(r"\brating:\s*([\d.]+)\s*$", item)
        if m_old:
            return item[:m_old.start()].strip(), f" rating: {m_old.group(1)}"
        return item, ""

    # LaMP_7: a past tweet, no gold-field suffix.
    return item, ""


def _extract_history_item(item: str, task_name: str, body_chars: int = 300,
                          max_item_chars: int = 400) -> str:
    """Char-level pre-truncation. Token-level budgeting (below) is the real
    constraint; this only kicks in for outlier items longer than
    ``max_item_chars`` characters."""
    if task_name in ("LaMP_4", "LaMP_5"):
        return item

    if task_name == "LongLaMP_abstract_generation_user":
        m = re.search(r"\btitle:\s*(.+?)\s*$", item)
        if m:
            title_str = f" title: {m.group(1)}"
            avail = max_item_chars - len(title_str) - 1
            body = item[:m.start()].strip()[:max(0, min(body_chars, avail))]
            return f"{body}{title_str}".strip()
        return item[:max_item_chars]

    if task_name == "LongLaMP_topic_writing_user":
        m = re.search(r"\bsummary:\s*(.+?)\s*$", item)
        if m:
            summary_str = f" summary: {m.group(1)}"
            avail = max_item_chars - len(summary_str) - 1
            body = item[:m.start()].strip()[:max(0, min(body_chars, avail))]
            return f"{body}{summary_str}".strip()
        return item[:max_item_chars]

    if task_name == "LongLaMP_product_review_user":
        m = re.search(
            r'^(description:\s*.+?)(\s+summary:\s+.+)$',
            item, re.IGNORECASE | re.DOTALL,
        )
        if m:
            suffix_part = m.group(2)
            avail = max_item_chars - len(suffix_part) - 1
            body = m.group(1).strip()[:max(0, min(body_chars, avail))]
            return f"{body}{suffix_part}".strip()
        # Empty product description -> text starts at "summary:" (see above).
        m_old = re.search(r"\brating:\s*([\d.]+)\s*$", item)
        if m_old:
            rating_str = f" rating: {m_old.group(1)}"
            avail = max_item_chars - len(rating_str) - 1
            body = item[:m_old.start()].strip()[:max(0, min(body_chars, avail))]
            return f"{body}{rating_str}".strip()
        return item[:max_item_chars]

    # LaMP_7: pass through.
    return item


def truncate_profile_items_tok(
    profile_items: List[str],
    task_name: str,
    tok,
    history_token_budget: int,
    max_items: int,
    max_per_item_tokens: Optional[int] = None,
) -> List[str]:
    """Tokenizer-aware truncation (LaMP-style equal token budget per item).

    For each retrieved history item we:
      1. split off the gold-field suffix (preserved verbatim);
      2. truncate the remaining body to a per-item token budget;
      3. carry forward unspent tokens to subsequent items so that short items
         donate budget to long ones.
    """
    if not profile_items or history_token_budget <= 0:
        return []
    items = [str(x).strip() for x in profile_items if str(x).strip()]
    if max_items and max_items > 0:
        items = items[:max_items]
    if not items:
        return []

    n = len(items)
    per_item_budget = max(1, history_token_budget // n)
    if max_per_item_tokens is not None and max_per_item_tokens > 0:
        per_item_budget = min(per_item_budget, int(max_per_item_tokens))

    saved_tokens = 0
    result = []
    prefix_tokens = len(tok("- ", add_special_tokens=False)["input_ids"])
    for item in items:
        body_raw, suffix = _get_body_and_suffix(item, task_name)
        suffix_tokens = (len(tok(suffix, add_special_tokens=False)["input_ids"])
                         if suffix else 0)
        body_budget = max(1, per_item_budget + saved_tokens - suffix_tokens - prefix_tokens)

        original_side = tok.truncation_side
        tok.truncation_side = "right"
        body_enc = tok(body_raw, max_length=body_budget,
                       truncation=True, add_special_tokens=False)
        tok.truncation_side = original_side
        body_trunc = tok.decode(body_enc["input_ids"], skip_special_tokens=True)

        actual = len(body_enc["input_ids"]) + suffix_tokens + prefix_tokens
        saved_tokens = max(0, per_item_budget + saved_tokens - actual)
        result.append((body_trunc + suffix).strip())
    return result


# ============================================================
# Prompt builder
# ============================================================
def _char_pre_caps(task_name: str):
    """Per-task char pre-truncation caps for ``_extract_history_item``.

    LongLaMP items (and their preserved suffixes) are long, so the char cap is
    kept loose and the tokenizer budget does the real work.
    """
    if task_name.startswith("LongLaMP_"):
        return 4000, 3600      # max_item_chars, body_chars
    if task_name in ("LaMP_4", "LaMP_5"):
        return 400, 300
    return 200, 170


def build_prompt(cfg, task_name: str, profile_text: str, input_text: str,
                 profile_items: List[str], tok=None) -> str:
    """Token-based prompt construction; `tok` is required.

    Training and evaluation call this with the same tokenizer, so a user sees the same
    history in both. The query is capped so that at least `cfg.history_min_budget`
    tokens stay available for the history.
    """
    if tok is None:
        raise ValueError("build_prompt requires tok (token-based path only).")

    if len(profile_items) == 0:
        profile_items = [profile_text if profile_text else input_text]

    max_item_chars, body_chars = _char_pre_caps(task_name)
    profile_items = [_extract_history_item(x, task_name, body_chars=body_chars,
                                           max_item_chars=max_item_chars)
                     for x in profile_items]
    profile_items = [" ".join(x.split()) for x in profile_items]

    chat_overhead = 64
    template_overhead = 100
    query_max_tokens = max(
        128,
        int(cfg.max_input_len) - int(cfg.history_min_budget)
        - chat_overhead - template_overhead - int(cfg.max_target_tokens),
    )
    query_ids = tok(input_text, add_special_tokens=False)["input_ids"]
    if len(query_ids) > query_max_tokens:
        input_text = tok.decode(query_ids[:query_max_tokens], skip_special_tokens=True)

    instruction_text = _build_instruction_only(task_name, input_text)
    instruction_tokens = len(tok(instruction_text, add_special_tokens=False)["input_ids"])
    history_token_budget = max(
        32,
        cfg.max_input_len - instruction_tokens - chat_overhead - cfg.max_target_tokens,
    )
    profile_items = truncate_profile_items_tok(
        profile_items, task_name=task_name, tok=tok,
        history_token_budget=history_token_budget,
        max_items=cfg.max_history_items,
        max_per_item_tokens=cfg.prompt_max_tokens_per_item or None,
    )

    history_text = "\n".join([f"- {x}" for x in profile_items])
    return f"### User History:\n{history_text}\n" + _instruction_block(task_name, input_text)


def _build_instruction_only(task_name: str, input_text: str) -> str:
    """Token-counting helper: the same prompt with an empty history."""
    return "### User History:\n\n" + _instruction_block(task_name, input_text)


def _instruction_block(task_name: str, input_text: str) -> str:
    """The instruction half of the prompt, shared by both builders above.

    For LaMP-4/5 the instruction is rewritten around the extracted field; for
    LaMP-7 and the LongLaMP tasks the benchmark input already carries its own
    instruction, so only the answer cue is appended.
    """
    if task_name == "LaMP_4":
        article = extract_article_text(input_text)
        return ("### User Instruction:\n"
                f"Generate a headline for the following article.\nArticle: {article}\nHeadline:")

    if task_name == "LaMP_5":
        abstract = extract_abstract_text(input_text)
        return ("### User Instruction:\n"
                f"Generate a title for the following abstract of a paper.\nAbstract: {abstract}\nTitle:")

    if task_name == "LaMP_7":
        # input_text = "Paraphrase the following tweet without any explanation
        #               before or after it: <tweet>"
        return f"### User Instruction:\n{input_text}\nParaphrased tweet:"

    if task_name == "LongLaMP_abstract_generation_user":
        # input_text = "Generate an abstract for the title ... using the following items: ..."
        return f"### User Instruction:\n{input_text}\nAbstract:"

    if task_name == "LongLaMP_topic_writing_user":
        # input_text = "Generate the content for a reddit post <summary>"
        return f"### User Instruction:\n{input_text}\nPost:"

    if task_name == "LongLaMP_product_review_user":
        # input_text = "Generate the review text ... rating X ... description Y ... summary Z"
        return f"### User Instruction:\n{input_text}\nReview:"

    return f"### User Instruction:\n{input_text}\nAnswer:"
