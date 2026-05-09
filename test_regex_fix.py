import re

# Test that the regex fix works correctly
test_cases = [
    ('```json\n{"facts": ["a"]}\n```', '{"facts": ["a"]}'),  # normal
    ('```\n{"facts": ["b"]}\n```', '{"facts": ["b"]}'),  # without json tag
    ('```json{"facts": ["c"]}```', '{"facts": ["c"]}'),  # no newlines
]

for msg, expected in test_cases:
    msg_stripped = msg.strip()
    result = msg_stripped
    if result.startswith("```"):
        result = re.sub(r"^```(?:json)?\s*", "", result)
        result = re.sub(r"\s*```$", "", result)
    print(f"Input:  {msg!r}")
    print(f"Output: {result!r}")
    print(f"Match:  {result == expected}")
    print()

# Also test that the old (broken) regex would fail
print("--- OLD BROKEN REGEX ---")
for msg, expected in test_cases:
    msg_stripped = msg.strip()
    result = msg_stripped
    if result.startswith("```"):
        # Old broken version had double backslash
        result = re.sub(r"^```(?:json)?\\s*", "", result)
        result = re.sub(r"\\s*```$", "", result)
    print(f"Input:  {msg!r}")
    print(f"Output: {result!r}")
    print(f"Match:  {result == expected}")
    print()

# Test prompt has real newlines
prompt_lines = [
    "Decompose the following text into a list of atomic, self-contained factual "
    "statements or concepts. Each statement or concept must contain enough context "
    "to be fully understood on its own.\n"
    "Do not exceed 500 characters per fact.\n"
    'Output JSON only with the shape `{"facts": ["...", "..."]}`.\n\nTEXT:\n'
    "Some sample text here"
]

prompt = "".join(prompt_lines)
print("--- PROMPT CONTAINS REAL NEWLINES ---")
has_newline = chr(10) in prompt
print(f"Contains real newlines: {has_newline}")
has_literal_bs_n = r'\n' in prompt
print(f"Contains literal backslash-n: {has_literal_bs_n}")
print(repr(prompt[:200]))