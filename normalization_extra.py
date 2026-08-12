"""Opt-in output normalization for extras experiments only.

It normalizes model serialization before constructing the same BIO labels that
the original pipeline passes to seqeval; it does not alter the metric.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass

from algorithms import Algorithm


TRUE_VALUES = {"true", "yes", "y", "1", "accepted", "entity"}
FALSE_VALUES = {"false", "no", "n", "0", "rejected", "non-entity", "not entity"}
ENTITY_KEYS = ("entity", "name", "text", "span")
TYPE_KEYS = ("type", "label", "category")
STATUS_KEYS = ("accepted", "is_entity", "status")


@dataclass
class PredictionRecord:
    entity: str | None
    accepted: bool | None
    entity_type: str | None
    raw_source: str
    rejection_reason: str | None = None

    def to_dict(self):
        return asdict(self)


class LabelSchema:
    """Validate output types against the active evaluation labels."""

    def __init__(self, labels, aliases=None):
        canonical = {}
        for label in labels:
            label = str(label).strip()
            if label == "O":
                continue
            if label.startswith(("B-", "I-")):
                label = label[2:]
            canonical[label.lower()] = label
        self.labels = canonical
        self.aliases = {str(k).lower(): str(v).lower() for k, v in (aliases or {}).items()}

    @classmethod
    def from_dataset(cls, dataset, aliases=None):
        return cls((label for row in dataset["exact_types"] for label in row), aliases)

    def canonicalize(self, value):
        if value is None:
            return None
        value = clean_text(value).strip("()[]{} ")
        value = re.sub(r"^(?:B|I)-", "", value, flags=re.I).replace(" ", "")
        key = self.aliases.get(value.lower(), value.lower())
        return self.labels.get(key)


def clean_text(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = (value.replace("\u2018", "'").replace("\u2019", "'")
                  .replace("\u201c", '"').replace("\u201d", '"'))
    value = re.sub(r"\*\*(.*?)\*\*|`([^`]*)`", lambda m: m.group(1) or m.group(2), value)
    return re.sub(r"\s+", " ", value).strip()


def parse_status(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = clean_text(value).strip(". ").lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return None


def type_in_text(text, schema):
    for pattern in (r"\(([^()]+)\)", r"\[([^\[\]]+)\]",
                    r"(?:type|label|category)\s*[:=]\s*([\w/-]+)"):
        for candidate in re.findall(pattern, text, flags=re.I):
            entity_type = schema.canonicalize(candidate)
            if entity_type:
                return entity_type
    return None


def remove_list_marker(line):
    return re.sub(r"^\s*(?:\d+\s*[.)\]:-]|[-*\u2022])\s*", "", line).strip()


def record_from_mapping(item, schema, raw):
    lowered = {str(key).lower(): value for key, value in item.items()}
    entity = next((lowered[key] for key in ENTITY_KEYS if key in lowered), None)
    entity_type = next((lowered[key] for key in TYPE_KEYS if key in lowered), None)
    status = next((lowered[key] for key in STATUS_KEYS if key in lowered), None)
    return PredictionRecord(clean_text(entity) if entity is not None else None,
                            parse_status(status), schema.canonicalize(entity_type), raw)


def record_from_line(line, schema):
    raw = line
    line = remove_list_marker(clean_text(line)).strip("| ")
    if not line or line.lower().rstrip(":") in {"answer", "entities", "predictions", "summary"}:
        return None
    if "|" in line:
        fields = [field.strip() for field in line.split("|") if field.strip()]
        if fields and fields[0].lower() in {"entity", "name", "text", "span"}:
            return None
        status = next((parse_status(field) for field in fields[1:] if parse_status(field) is not None), None)
        entity_type = next((type_in_text(field, schema) for field in fields[1:] if type_in_text(field, schema)), None)
        # Tables often put the label in its own column (``Alice | PER | yes``)
        # rather than wrapping it in parentheses.
        if entity_type is None:
            entity_type = next((schema.canonicalize(field) for field in fields[1:]
                                if schema.canonicalize(field)), None)
        return PredictionRecord(fields[0] if fields else None, True if status is None and entity_type else status,
                                entity_type, raw)
    match = re.match(r"^(.*?)(?:\s*[,;:\u2014-]\s*)(?:type|label|category)\s*[:=]\s*([\w/-]+)(?:\s*[,;]\s*(.*))?$", line, re.I)
    if match:
        status = parse_status(match.group(3))
        return PredictionRecord(match.group(1).strip(), True if status is None else status,
                                schema.canonicalize(match.group(2)), raw)
    entity_type = type_in_text(line, schema)
    if entity_type:
        entity = re.sub(r"\s*(?:\([^()]+\)|\[[^\[\]]+\])\s*$", "", line).strip(" :-|")
        entity = re.sub(r"\s+(?:true|false|yes|no)\s*$", "", entity, flags=re.I)
        return PredictionRecord(entity, True, entity_type, raw)
    return PredictionRecord(None, None, None, raw, "unrecognized record")


def normalize_predictions(output, schema):
    """Parse supported serializations and retain rejection diagnostics."""
    # Preserve line structure until records have been separated.  ``clean_text``
    # deliberately collapses whitespace and is applied to each record below.
    text = unicodedata.normalize("NFKC", str(output)).replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    # Completion prompt echoes are common with local models: retain final answer.
    answer = list(re.finditer(r"(?:^|\n)\s*answer\s*:\s*", text, flags=re.I))
    if answer:
        text = text[answer[-1].end():]
    records = []
    try:
        decoded = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I))
        if isinstance(decoded, dict):
            decoded = decoded.get("entities", decoded.get("predictions", [decoded]))
        if isinstance(decoded, list):
            records = [record_from_mapping(item, schema, json.dumps(item)) if isinstance(item, dict)
                       else record_from_line(str(item), schema) for item in decoded]
    except json.JSONDecodeError:
        records = [record_from_line(line, schema) for line in text.splitlines()]
    output_records = []
    for record in records:
        if record is None:
            continue
        if record.rejection_reason is None:
            if not record.entity:
                record.rejection_reason = "missing entity"
            elif record.accepted is not True:
                record.rejection_reason = "not explicitly accepted"
            elif not record.entity_type:
                record.rejection_reason = "missing or invalid entity type"
        output_records.append(record)
    return output_records


def surface_key(text):
    text = clean_text(text).lower().strip("'`\"")
    return re.sub(r"\s+([,.;:!?%\)])", r"\1", re.sub(r"([\(])\s+", r"\1", text))


class NormalizedAlgorithm(Algorithm):
    """Algorithm subclass that leaves original PromptNER code unmodified."""

    def __init__(self, *args, label_schema=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_schema = label_schema
        self.normalization_records = []

    def _normalized_result(self, output):
        records = normalize_predictions(output, self.label_schema)
        self.normalization_records = [record.to_dict() for record in records]
        accepted = [record for record in records if record.rejection_reason is None]
        return [record.entity.lower() for record in accepted], [f"({record.entity_type})" for record in accepted]

    def perform_single_query(self, verbose=True):
        if self.exemplar_task is not None:
            task = self.defn + "\n" + self.exemplar_task + f" '{self.para}' \nAnswer:"
        else:
            task = self.defn + "\n" + self.format_task + f"\nParagraph: {self.para} \nAnswer:"
        output = self.model_fn(task)
        if verbose:
            print(output)
        answers, types = self._normalized_result(output)
        return answers, types, output

    def parse_span(self, answers, typestrings, metadata, true_tokens=None):
        tokens = list(true_tokens) if true_tokens is not None else self.para.split(" ")
        prediction = ["O"] * len(tokens)
        used = {}
        for entity, type_string in zip(answers, typestrings):
            type_match = re.search(r"\(([^()]+)\)", type_string)
            if not type_match:
                continue
            wanted = surface_key(entity)
            spans = []
            for start in range(len(tokens)):
                for end in range(start + 1, len(tokens) + 1):
                    if surface_key(" ".join(tokens[start:end])) == wanted:
                        spans.append((start, end))
            occurrence = used.get(wanted, 0)
            if occurrence >= len(spans):
                continue
            used[wanted] = occurrence + 1
            start, end = spans[occurrence]
            entity_type = type_match.group(1)
            if "-" in entity_type:
                prediction[start:end] = [entity_type] * (end - start)
            else:
                prediction[start] = "B-" + entity_type
                for index in range(start + 1, end):
                    prediction[index] = "I-" + entity_type
        return prediction, metadata
