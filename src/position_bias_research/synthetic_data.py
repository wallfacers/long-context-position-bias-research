"""Deterministic position-equivalent synthetic data generation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .tokenization import TokenCounter


SUPPORTED_TASKS = ("kv", "two_hop")
SUPPORTED_FILLERS = ("neutral", "same_format", "answer_bearing")

SYSTEM_PROMPT = (
    "Use only the supplied context. Return valid JSON with answer, evidence_ids, "
    "evidence_quotes, and confidence. Evidence quotes must be short exact spans."
)

NEUTRAL_WORDS = (
    "archive basin cedar delta ember field gallery harbor island junction kernel "
    "lagoon meadow nickel orchard prairie quartz ridge summit timber upland valley "
    "willow xenon yard zenith amber bridge circuit domain engine forest garden "
    "horizon index lattice museum network observatory pattern route station tunnel"
).split()

COLORS = (
    "amber azure bronze coral cyan emerald indigo ivory jade lilac magenta ochre "
    "olive pearl plum saffron scarlet silver teal umber violet"
).split()


@dataclass(frozen=True)
class TaskFact:
    answer: str
    query: str
    evidence_documents: tuple[dict[str, str], ...]
    evidence_ids: tuple[str, ...]
    evidence_quotes: tuple[str, ...]
    distractor_payload: dict[str, str]


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def position_label(position: float) -> str:
    return f"p{round(position * 100):03d}"


def _random_code(rng: random.Random, prefix: str, width: int = 8) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(width))


def make_task_fact(task: str, group_id: str, rng: random.Random) -> TaskFact:
    if task == "kv":
        key = _random_code(rng, "KEY", 10)
        answer = _random_code(rng, "VALUE", 8)
        evidence_id = f"evidence-{stable_hash(group_id)[:10]}"
        quote = f"The lookup value for {key} is {answer}."
        return TaskFact(
            answer=answer,
            query=f"What is the lookup value for {key}?",
            evidence_documents=(({"id": evidence_id, "text": quote}),),
            evidence_ids=(evidence_id,),
            evidence_quotes=(quote,),
            distractor_payload={"target_key": key, "target_answer": answer},
        )

    if task == "two_hop":
        entity = _random_code(rng, "ENTITY", 8)
        bridge = _random_code(rng, "BRIDGE", 8)
        answer = rng.choice(COLORS) + "-" + _random_code(rng, "SHADE", 4)
        suffix = stable_hash(group_id)[:8]
        first_id = f"evidence-a-{suffix}"
        second_id = f"evidence-b-{suffix}"
        first_quote = f"The routing code assigned to {entity} is {bridge}."
        second_quote = f"The display color mapped from routing code {bridge} is {answer}."
        return TaskFact(
            answer=answer,
            query=f"What display color is associated with {entity}?",
            evidence_documents=(
                {"id": first_id, "text": first_quote},
                {"id": second_id, "text": second_quote},
            ),
            evidence_ids=(first_id, second_id),
            evidence_quotes=(first_quote, second_quote),
            distractor_payload={
                "target_entity": entity,
                "target_bridge": bridge,
                "target_answer": answer,
            },
        )

    raise ValueError(f"Unsupported task: {task}")


def make_filler_document(
    filler_type: str,
    document_index: int,
    group_id: str,
    task: str,
    fact: TaskFact,
    rng: random.Random,
    words_per_document: int,
) -> dict[str, str]:
    document_id = f"filler-{document_index:05d}"
    if filler_type == "neutral":
        words = [rng.choice(NEUTRAL_WORDS) for _ in range(words_per_document)]
        text = (
            f"Archive note {document_index}. "
            + " ".join(words)
            + ". This note is unrelated to any lookup or routing request."
        )
    elif task == "kv":
        distractor_key = _random_code(rng, "KEY", 10)
        if filler_type == "answer_bearing" and document_index % 3 == 0:
            distractor_value = fact.answer
        else:
            distractor_value = _random_code(rng, "VALUE", 8)
        padding = " ".join(rng.choice(NEUTRAL_WORDS) for _ in range(words_per_document))
        text = (
            f"Distractor lookup record {document_index}. The lookup value for "
            f"{distractor_key} is {distractor_value}. Notes: {padding}."
        )
    else:
        distractor_entity = _random_code(rng, "ENTITY", 8)
        distractor_bridge = _random_code(rng, "BRIDGE", 8)
        if filler_type == "answer_bearing" and document_index % 3 == 0:
            distractor_answer = fact.answer
        else:
            distractor_answer = rng.choice(COLORS) + "-" + _random_code(rng, "SHADE", 4)
        padding = " ".join(rng.choice(NEUTRAL_WORDS) for _ in range(words_per_document))
        text = (
            f"Distractor routing record {document_index}. The routing code assigned to "
            f"{distractor_entity} is {distractor_bridge}. The display color mapped from "
            f"routing code {distractor_bridge} is {distractor_answer}. Notes: {padding}."
        )

    return {"id": document_id, "text": text}


def render_document(document: dict[str, str]) -> str:
    return f"[Document {document['id']}]\n{document['text']}"


def render_prompt(documents: Sequence[dict[str, str]], query: str) -> str:
    body = "\n\n".join(render_document(document) for document in documents)
    return (
        f"<context>\n{body}\n</context>\n\nQuestion: {query}\nResponse:"
    )


def render_model_input(prompt: str) -> str:
    """Approximate chat-template input without model-specific special tokens."""

    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def _insert_cluster(
    filler_documents: Sequence[dict[str, str]],
    evidence_documents: Sequence[dict[str, str]],
    insertion_index: int,
) -> list[dict[str, str]]:
    return [
        *filler_documents[:insertion_index],
        *evidence_documents,
        *filler_documents[insertion_index:],
    ]


def _nearest_insertion_index(
    filler_documents: Sequence[dict[str, str]],
    target_position: float,
    token_counter: TokenCounter,
) -> int:
    if target_position <= 0:
        return 0
    if target_position >= 1:
        return len(filler_documents)

    rendered = [render_document(document) for document in filler_documents]
    lengths = [token_counter.count(text) for text in rendered]
    total = sum(lengths)
    target = target_position * total
    cumulative = 0
    best_index = 0
    best_distance = abs(target)
    for index, length in enumerate(lengths, start=1):
        cumulative += length
        distance = abs(cumulative - target)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _actual_evidence_position(
    prompt: str,
    first_evidence_id: str,
    token_counter: TokenCounter,
) -> tuple[float, int]:
    model_input = render_model_input(prompt)
    marker = f"[Document {first_evidence_id}]"
    marker_index = model_input.index(marker)
    tokens_before = token_counter.count(model_input[:marker_index])
    total_tokens = token_counter.count_chat(SYSTEM_PROMPT, prompt)
    return tokens_before / max(total_tokens, 1), total_tokens


def _build_filler_documents(
    *,
    task: str,
    filler_type: str,
    group_id: str,
    fact: TaskFact,
    target_tokens: int,
    token_counter: TokenCounter,
    rng: random.Random,
    words_per_document: int,
) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    empty_prompt = render_prompt(list(fact.evidence_documents), fact.query)
    current_estimate = token_counter.count_chat(SYSTEM_PROMPT, empty_prompt)
    if current_estimate >= target_tokens:
        raise ValueError(f"target_tokens={target_tokens} is too small for task {task}")

    # Small documents keep placement error bounded while preserving document boundaries.
    while True:
        candidate = make_filler_document(
            filler_type,
            len(documents),
            group_id,
            task,
            fact,
            rng,
            words_per_document,
        )
        # Counting each document once avoids quadratic tokenization at 32K.
        candidate_tokens = token_counter.count("\n\n" + render_document(candidate))
        candidate_estimate = current_estimate + candidate_tokens
        if candidate_estimate > target_tokens:
            current_error = abs(current_estimate - target_tokens)
            candidate_error = abs(candidate_estimate - target_tokens)
            if candidate_error < current_error:
                documents.append(candidate)
            break
        documents.append(candidate)
        current_estimate = candidate_estimate
    return documents


def _render_position_group(
    *,
    schema_version: str,
    split: str,
    group_id: str,
    task: str,
    filler_type: str,
    target_tokens: int,
    positions: Sequence[float],
    row_seed: int,
    token_counter: TokenCounter,
    words_per_document: int,
    fact: TaskFact,
    filler_rng: random.Random,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render one fact/filler view at every requested evidence position."""

    filler_documents = _build_filler_documents(
        task=task,
        filler_type=filler_type,
        group_id=group_id,
        fact=fact,
        target_tokens=target_tokens,
        token_counter=token_counter,
        rng=filler_rng,
        words_per_document=words_per_document,
    )
    filler_fingerprint = stable_hash(
        json.dumps(filler_documents, ensure_ascii=False, sort_keys=True)
    )
    fact_fingerprint = stable_hash(
        json.dumps(
            {
                "answer": fact.answer,
                "query": fact.query,
                "evidence_documents": fact.evidence_documents,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    samples: list[dict[str, Any]] = []
    for target_position in positions:
        insertion_index = _nearest_insertion_index(
            filler_documents, target_position, token_counter
        )
        documents = _insert_cluster(
            filler_documents,
            fact.evidence_documents,
            insertion_index,
        )
        prompt = render_prompt(documents, fact.query)
        actual_position, actual_tokens = _actual_evidence_position(
            prompt, fact.evidence_ids[0], token_counter
        )
        label = position_label(target_position)
        metadata = {
            "filler_fingerprint": filler_fingerprint,
            "filler_document_count": len(filler_documents),
            "evidence_document_count": len(fact.evidence_documents),
            "evidence_clustered": True,
            "insertion_document_index": insertion_index,
        }
        if extra_metadata:
            metadata["fact_fingerprint"] = fact_fingerprint
            metadata.update(extra_metadata)
        samples.append(
            {
                "schema_version": schema_version,
                "sample_id": f"{group_id}@{label}",
                "group_id": group_id,
                "split": split,
                "task": task,
                "filler_type": filler_type,
                "query_position": "end",
                "target_tokens": target_tokens,
                "actual_tokens": actual_tokens,
                "target_position": target_position,
                "actual_position": actual_position,
                "position_label": label,
                "seed": row_seed,
                "tokenizer": token_counter.name,
                "system_prompt": SYSTEM_PROMPT,
                "prompt": prompt,
                "target": {
                    "answer": fact.answer,
                    "evidence_ids": list(fact.evidence_ids),
                    "evidence_quotes": list(fact.evidence_quotes),
                    "confidence": 1.0,
                },
                "metadata": metadata,
            }
        )
    return samples


def generate_position_equivalent_group(
    *,
    split: str,
    group_index: int,
    task: str,
    filler_type: str,
    target_tokens: int,
    positions: Sequence[float],
    seed: int,
    token_counter: TokenCounter,
    words_per_document: int = 48,
) -> list[dict[str, Any]]:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if filler_type not in SUPPORTED_FILLERS:
        raise ValueError(f"Unsupported filler: {filler_type}")
    if not positions or any(position < 0 or position > 1 for position in positions):
        raise ValueError("positions must be non-empty values in [0, 1]")

    group_id = f"{split}-{task}-{filler_type}-{target_tokens}-{group_index:06d}"
    group_seed = int(stable_hash(f"{seed}:{group_id}")[:16], 16)
    rng = random.Random(group_seed)
    fact = make_task_fact(task, group_id, rng)
    return _render_position_group(
        schema_version="position-group-v1",
        split=split,
        group_id=group_id,
        task=task,
        filler_type=filler_type,
        target_tokens=target_tokens,
        positions=positions,
        row_seed=group_seed,
        token_counter=token_counter,
        words_per_document=words_per_document,
        fact=fact,
        filler_rng=rng,
    )


def generate_position_equivalent_replica_group(
    *,
    split: str,
    fact_index: int,
    replica_index: int,
    task: str,
    filler_type: str,
    target_tokens: int,
    positions: Sequence[float],
    seed: int,
    token_counter: TokenCounter,
    words_per_document: int = 48,
) -> list[dict[str, Any]]:
    """Generate one reusable fact with an independently sampled filler view.

    Fact identity and filler identity use separate deterministic seeds.  Multiple
    replicas therefore contain exactly the same target fact but different filler
    documents.  This supports a matched design where paired and independent
    variants use the same facts, exposure counts, and filler views.
    """

    if fact_index < 0:
        raise ValueError("fact_index must be non-negative")
    if replica_index < 0:
        raise ValueError("replica_index must be non-negative")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    if filler_type not in SUPPORTED_FILLERS:
        raise ValueError(f"Unsupported filler: {filler_type}")
    if not positions or any(position < 0 or position > 1 for position in positions):
        raise ValueError("positions must be non-empty values in [0, 1]")

    fact_id = f"{split}-{task}-fact-{fact_index:06d}"
    group_id = (
        f"{fact_id}-{filler_type}-{target_tokens}-rep-{replica_index:03d}"
    )
    fact_seed = int(stable_hash(f"{seed}:fact:{fact_id}")[:16], 16)
    filler_seed = int(stable_hash(f"{seed}:filler:{group_id}")[:16], 16)
    fact = make_task_fact(task, fact_id, random.Random(fact_seed))
    return _render_position_group(
        schema_version="position-group-v2",
        split=split,
        group_id=group_id,
        task=task,
        filler_type=filler_type,
        target_tokens=target_tokens,
        positions=positions,
        row_seed=filler_seed,
        token_counter=token_counter,
        words_per_document=words_per_document,
        fact=fact,
        filler_rng=random.Random(filler_seed),
        extra_metadata={
            "training_design": "matched-position-v1",
            "fact_id": fact_id,
            "fact_index": fact_index,
            "replica_index": replica_index,
            "fact_seed": fact_seed,
            "filler_seed": filler_seed,
        },
    )


def iter_matched_training_bank(
    *,
    split: str,
    facts_per_condition: int,
    replicas_per_fact: int,
    tasks: Sequence[str],
    filler_types: Sequence[str],
    target_lengths: Sequence[int],
    positions: Sequence[float],
    seed: int,
    token_counter: TokenCounter,
    words_per_document: int = 48,
) -> Iterable[dict[str, Any]]:
    """Yield a balanced fact × filler-replica × position training bank."""

    if facts_per_condition <= 0:
        raise ValueError("facts_per_condition must be positive")
    if replicas_per_fact <= 0:
        raise ValueError("replicas_per_fact must be positive")
    for fact_index in range(facts_per_condition):
        for target_tokens in target_lengths:
            for task in tasks:
                for filler_type in filler_types:
                    for replica_index in range(replicas_per_fact):
                        yield from generate_position_equivalent_replica_group(
                            split=split,
                            fact_index=fact_index,
                            replica_index=replica_index,
                            task=task,
                            filler_type=filler_type,
                            target_tokens=target_tokens,
                            positions=positions,
                            seed=seed,
                            token_counter=token_counter,
                            words_per_document=words_per_document,
                        )


def iter_synthetic_samples(
    *,
    split: str,
    groups_per_condition: int,
    tasks: Sequence[str],
    filler_types: Sequence[str],
    target_lengths: Sequence[int],
    positions: Sequence[float],
    seed: int,
    token_counter: TokenCounter,
    words_per_document: int = 48,
) -> Iterable[dict[str, Any]]:
    if groups_per_condition <= 0:
        raise ValueError("groups_per_condition must be positive")
    # Interleave conditions so prefixes of the file remain balanced.
    for group_index in range(groups_per_condition):
        for target_tokens in target_lengths:
            for task in tasks:
                for filler_type in filler_types:
                    yield from generate_position_equivalent_group(
                        split=split,
                        group_index=group_index,
                        task=task,
                        filler_type=filler_type,
                        target_tokens=target_tokens,
                        positions=positions,
                        seed=seed,
                        token_counter=token_counter,
                        words_per_document=words_per_document,
                    )
