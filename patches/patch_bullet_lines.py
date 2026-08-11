with open("utils.py") as f:
    content = f.read()

old = '''                if c.lower().strip() in ["", "answer:"]:
                    pass
                elif re.match(r"\\d+[.)]+ *", c):
                    start = 0
                    while c[start].isnumeric() or c[start] == '.':
                        start += 1
                    final.append(c[start:].strip())
                else:
                    print(f"Unable to match nonempty {c}")
                    pass'''

new = '''                if c.lower().strip() in ["", "answer:"]:
                    pass
                elif re.match(r"\\d+[.)]+ *", c):
                    start = 0
                    while c[start].isnumeric() or c[start] == '.':
                        start += 1
                    final.append(c[start:].strip())
                elif re.match(r"[-*\\u2022]\\s+\\S", c):
                    # bullet-marker line (no leading number) -- strip the
                    # bullet and pass the rest through the same way a
                    # numbered item would be, so it can still reach the
                    # existing pipe-parser or the extract_type_anchored
                    # fallback downstream.
                    final.append(re.sub(r"^[-*\\u2022]\\s+", "", c).strip())
                elif re.match(r"^#+\\s", c) or c.lower().strip() in ["summary:", "summary", "entities:", "in summary:", "in summary"]:
                    # markdown headers / bare section labels -- not a real
                    # candidate line, skip silently like "answer:" above
                    pass
                else:
                    print(f"Unable to match nonempty {c}")
                    pass'''

assert old in content, "target block not found -- no changes made"
content = content.replace(old, new, 1)
with open("utils.py", "w") as f:
    f.write(content)
print("Patched get_numbered_list_items: now accepts bullet-marker lines and skips markdown headers/section labels.")
