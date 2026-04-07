import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from storage.filesystem.utils import resolve_base_path

def test_resolve_base_path():
    home = Path.home()
    
    test_cases = [
        # (input_uri, expected_suffix_after_home, description)
        ("", ".librarian_mcp", "Empty string defaults to ~/.librarian_mcp"),
        ("localhost", ".librarian_mcp", "localhost defaults to ~/.librarian_mcp"),
        ("file:///tmp/librarian", "/tmp/librarian", "Absolute file URI"),
        ("file://~/librarian", str(home / "librarian"), "Home relative with file://"),
        ("file:///~/librarian", str(home / "librarian"), "Home relative with file:/// (THE BUG CASE)"),
        ("~/librarian", str(home / "librarian"), "Raw path with ~"),
        ("/tmp/librarian", "/tmp/librarian", "Raw absolute path"),
    ]

    print(f"Home directory: {home}")
    
    all_passed = True
    for uri, expected, desc in test_cases:
        try:
            resolved = resolve_base_path(uri)
            # Handle cases where expected is a full path or a suffix
            if uri in ("", "localhost"):
                expected_path = home / expected
            elif expected.startswith(str(home)):
                expected_path = Path(expected)
            else:
                expected_path = Path(expected)
            
            if resolved == expected_path:
                print(f"✅ PASS: {desc} ({uri} -> {resolved})")
            else:
                print(f"❌ FAIL: {desc}")
                print(f"   Input:    {uri}")
                print(f"   Expected: {expected_path}")
                print(f"   Got:      {resolved}")
                all_passed = False
        except Exception as e:
            print(f"💥 ERROR: {desc} ({uri})")
            print(f"   Exception: {e}")
            all_passed = False

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    test_resolve_base_path()
