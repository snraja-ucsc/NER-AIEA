import re

with open("utils.py") as f:
    content = f.read()

helper_fn = '''
_VALID_TYPE_TOKENS = {"per", "org", "loc", "misc", "person", "organisation",
                       "location", "b-per", "i-per", "b-org", "i-org",
                       "b-loc", "i-loc", "b-misc", "i-misc"}


def extract_type_anchored(line):
    """
    FALLBACK extractor -- only used when a line has no '|' separator at all
    (i.e. the existing strict parser already gave up on it). Looks for a
    known entity-type label in parentheses (matching the paper's own
    exemplar convention, e.g. "(org)", "(B-PER)") and, if found, extracts
    the text immediately preceding it as the entity candidate. Returns
    (entity, type) or None if no confident match is found.
    """
    c = line.strip()
    c = re.sub(r"\\*\\*(.*?)\\*\\*", r"\\1", c)
    c = re.sub(r"^[-*\\u2022]\\s*", "", c)

    paren_match = re.search(r"\\(([A-Za-z\\-]+)\\)\\s*$", c)
    if not paren_match:
        paren_match = re.search(r"\\(([A-Za-z\\-]+)\\)", c)
    if not paren_match or paren_match.group(1).lower() not in _VALID_TYPE_TOKENS:
        colon_match = re.match(r"^(.+):\\s*([A-Za-z\\-]+)\\s*$", c)
        if colon_match and colon_match.group(2).lower() in _VALID_TYPE_TOKENS:
            entity = colon_match.group(1).strip(" :|-")
            if entity:
                return (entity, colon_match.group(2))
        return None

    typ = paren_match.group(1)
    before = c[:paren_match.start()].strip()
    before = re.sub(r"[:|]\\s*$", "", before).strip()
    before = re.sub(r"\\b(true|false)\\b\\s*$", "", before, flags=re.IGNORECASE).strip()
    before = before.strip(" :|-")
    if not before:
        return None
    return (before, typ)


'''

content = content.replace("class AnswerMapping:", helper_fn + "class AnswerMapping:", 1)

old_else = '''            else:
                final.append(option.strip().lower())'''

new_else = '''            else:
                extracted = extract_type_anchored(option)
                if extracted is not None:
                    entity, typ = extracted
                    final.append(entity.strip().lower())
                    typestring.append(f"({typ})")
                else:
                    pass  # no confident extraction; skip rather than risk
                          # corrupting final/typestring list alignment'''

if old_else not in content:
    print("ERROR: could not find the target else branch. No changes made.")
else:
    content = content.replace(old_else, new_else, 1)
    with open("utils.py", "w") as f:
        f.write(content)
    print("Patched utils.py successfully: added extract_type_anchored() and fixed the else branch.")
