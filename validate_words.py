#!/usr/bin/env python3
"""
validate_words.py

Reads a list of words from 'words_to_validate.txt', loads a pickled trie (e.g., 'trie.pkl'),
and checks each word for membership in the trie. Outputs a dictionary mapping
each word to True (found) or False (not found), with extensive debug logging.

Usage:
    python validate_words.py --trie-file /path/to/trie.pkl \
                             --validate-file /path/to/words_to_validate.txt

The script prints debug information to stderr (prefix "DEBUG:") and final results to stdout.
"""

import argparse
import pickle
import sys
import os

def debug(msg):
    """Print debug message to stderr with a DEBUG prefix."""
    print(f"DEBUG: {msg}", file=sys.stderr)

def load_trie(trie_path: str):
    """
    Load and return the root TrieNode from a pickle file.
    Emits debug information throughout and exits on failure.
    """
    debug(f"load_trie: Starting with trie_path='{trie_path}'")
    if not os.path.isfile(trie_path):
        debug(f"load_trie: File not found at '{trie_path}'")
        print(f"Error: could not open trie file at '{trie_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        size_bytes = os.path.getsize(trie_path)
        debug(f"load_trie: Trie file size is {size_bytes} bytes")
    except Exception as e:
        debug(f"load_trie: Could not get file size: {e}")

    try:
        with open(trie_path, "rb") as f:
            debug("load_trie: Opened trie file in binary mode for reading")
            root = pickle.load(f)
            debug("load_trie: Successfully unpickled trie root")
    except FileNotFoundError:
        debug("load_trie: Caught FileNotFoundError when opening the file")
        print(f"Error: could not open trie file at '{trie_path}'", file=sys.stderr)
        sys.exit(1)
    except pickle.UnpicklingError as ue:
        debug(f"load_trie: Caught UnpicklingError: {ue}")
        print(f"Error loading trie: UnpicklingError: {ue}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        debug(f"load_trie: Caught general exception when unpickling: {e}")
        print(f"Error loading trie: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify the loaded object has expected attributes
    if not hasattr(root, "children") or not hasattr(root, "is_word"):
        debug("load_trie: Unpickled object lacks 'children' or 'is_word' attributes")
        print("Error: unpickled object does not appear to be a valid TrieNode", file=sys.stderr)
        sys.exit(1)

    return root

def load_words_to_validate(validate_path: str):
    """
    Read each nonempty line from validate_path, strip whitespace, convert to lowercase,
    and return a list of pure a–z words. Emit debug logs as each line is processed.
    """
    debug(f"load_words_to_validate: Starting with validate_path='{validate_path}'")
    if not os.path.isfile(validate_path):
        debug(f"load_words_to_validate: File not found at '{validate_path}'")
        print(f"Error: could not open validate-file at '{validate_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        size_bytes = os.path.getsize(validate_path)
        debug(f"load_words_to_validate: Validate file size is {size_bytes} bytes")
    except Exception as e:
        debug(f"load_words_to_validate: Could not get file size: {e}")

    words = []
    with open(validate_path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            original = line.rstrip("\n\r")
            w = original.strip().lower()
            if not w:
                debug(f"load_words_to_validate: Line {lineno} is blank or whitespace, skipping")
                continue
            if not w.isalpha():
                debug(f"load_words_to_validate: Line {lineno} '{original}' contains non-alpha characters; skipping")
                print(f"Warning: skipping invalid word '{original}' on line {lineno}", file=sys.stderr)
                continue
            debug(f"load_words_to_validate: Line {lineno} -> valid word '{w}'")
            words.append(w)

    debug(f"load_words_to_validate: Collected {len(words)} valid words from file")
    return words

def is_in_trie(root, word: str):
    """
    Traverse the trie from 'root' following each character of 'word'.
    Returns True if, after consuming all letters, the node exists and node.is_word is True.
    Emits debug logs for each character in the traversal.
    """
    debug(f"is_in_trie: Checking '{word}'")
    node = root
    for i, ch in enumerate(word):
        if not hasattr(node, "children"):
            debug(f"is_in_trie: Node at prefix '{word[:i]}' has no 'children'; returning False")
            return False
        if ch not in node.children:
            debug(f"is_in_trie: Character '{ch}' not found under prefix '{word[:i]}'; returning False")
            return False
        node = node.children[ch]
        debug(f"is_in_trie: Moved to node for '{word[:i+1]}'")
    exists = getattr(node, "is_word", False)
    debug(f"is_in_trie: Final node for '{word}' has is_word={exists}")
    return exists

def main():
    debug("main: Parsing command-line arguments")
    parser = argparse.ArgumentParser(
        description="Validate a list of words against a pickled trie, with debug output."
    )
    parser.add_argument(
        "--trie-file", "-t", required=True,
        help="Path to the pickled trie (e.g., trie.pkl)."
    )
    parser.add_argument(
        "--validate-file", "-v", required=True,
        help="Path to the text file containing words to validate (one per line)."
    )
    args = parser.parse_args()

    # Resolve and debug-print absolute paths
    trie_path = os.path.abspath(os.path.expanduser(args.trie_file))
    validate_path = os.path.abspath(os.path.expanduser(args.validate_file))
    debug(f"main: Resolved trie-file to '{trie_path}'")
    debug(f"main: Resolved validate-file to '{validate_path}'")

    # 1) Load the trie
    debug("main: Calling load_trie()")
    trie_root = load_trie(trie_path)
    debug("main: Trie loaded successfully")

    # 2) Load words to validate
    debug("main: Calling load_words_to_validate()")
    words = load_words_to_validate(validate_path)
    if not words:
        debug("main: No valid words returned by load_words_to_validate()")
        print("No valid words found in validate-file. Exiting.", file=sys.stderr)
        sys.exit(1)
    else:
        debug(f"main: Loaded {len(words)} words to validate")

    # 3) Check each word
    debug("main: Beginning validation of each word in trie")
    results = {}
    count_true = 0
    count_false = 0
    for w in words:
        exists = is_in_trie(trie_root, w)
        results[w] = exists
        if exists:
            debug(f"main: '{w}' → True")
            count_true += 1
        else:
            debug(f"main: '{w}' → False")
            count_false += 1

    # 4) Summarize and print results
    debug(f"main: Validation complete: {count_true} found, {count_false} not found")
    for word, exists in results.items():
        print(f"{word}: {exists}")

    debug("main: Exiting normally")

if __name__ == "__main__":
    main()
