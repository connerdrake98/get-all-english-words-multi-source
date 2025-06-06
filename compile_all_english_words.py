#!/usr/bin/env python3
"""
generate_all_eng_words_list.py

Creates the union of multiple large English word‐lists (DWYL, SCOWL, WordNet, wordfreq,
Linux /usr/share/dict/words, and optionally Moby Word Lists), writing the result to one
output file (one word per line). If any source fails to load (except /usr/share/dict/words),
the script raises an exception.

Additionally, if the user supplies an omit-file via --omit-file, any word in that file
will be excluded from the final union.  Finally, this version also builds a Trie from
that final union and pickles (serializes) it to disk.

Usage:
    python generate_all_eng_words_list.py \
        --output /path/to/all_english_union.txt \
        [--tempdir /path/to/temp_workspace] \
        [--moby-file /path/to/moby-3201.txt] \
        [--omit-file /path/to/words_to_omit_from_trie.txt] \
        [--trie-output /path/to/trie.pkl]

Options:
  -h, --help            Show this help message and exit  
  --output OUTPUT, -o   Path to output text file (one word per line). Overwritten if it exists.  
  --tempdir TEMPDIR, -t Directory for intermediate downloads/clones. If omitted,  
                        a temporary directory is created and removed on exit.  
  --moby-file MOBY_FILE, -m  
                        (Optional) Path to local Moby Word Lists e-text (Project Gutenberg #3201).  
  --omit-file OMIT_FILE, -x  
                        (Optional) Path to a text file containing words to exclude (one per line).  
  --trie-output TRIE_OUTPUT, -r  
                        (Optional) Path to write the pickled trie. If omitted, defaults to
                        the `--output` filename with a `.pkl` extension.
"""

import os
import sys
import subprocess
import tarfile
import urllib.request
import tempfile
import argparse
import re
import shutil
import pickle

# Import your TrieNode class:
try:
    from trie import TrieNode
except ImportError:
    print("ERROR: Could not import TrieNode from 'trie.py'. Ensure trie.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# Third‐party imports (ensure you ran: pip install nltk wordfreq)
try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("ERROR: nltk not installed. Run `pip install nltk` first.", file=sys.stderr)
    sys.exit(1)

try:
    from wordfreq import iter_wordlist
except ImportError:
    print("ERROR: wordfreq not installed. Run `pip install wordfreq` first.", file=sys.stderr)
    sys.exit(1)


################################################################################
# UTILITY FUNCTIONS
################################################################################

def ensure_dir(path):
    """Create directory if it doesn’t exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def run_command(cmd_list, cwd=None):
    """Run a subprocess command, raising on failure."""
    subprocess.run(cmd_list, cwd=cwd, check=True)

def is_pure_alpha(word):
    """Return True if `word` consists of only lowercase a–z."""
    return bool(re.fullmatch(r"[a-z]+", word))


################################################################################
# 1. DWYL “english-words” loader (raises on failure)
################################################################################

def load_dwyl_words(tempdir):
    """
    Clone (or pull) the DWYL repository and read words_alpha.txt.
    Returns a set of all words (already lowercase a–z).
    Raises RuntimeError if anything goes wrong.
    """
    print("\n→ Loading DWYL ‘english-words’…")
    dwyl_dir = os.path.join(tempdir, "english-words")
    if not os.path.isdir(dwyl_dir):
        print(f"  • Cloning https://github.com/dwyl/english-words.git into {dwyl_dir} …")
        try:
            run_command(["git", "clone", "https://github.com/dwyl/english-words.git", dwyl_dir])
        except Exception as e:
            raise RuntimeError(f"DWYL loader error: could not clone repo: {e}")
    else:
        print(f"  • {dwyl_dir} already exists. Pulling latest changes…")
        try:
            run_command(["git", "-C", dwyl_dir, "pull"])
        except Exception as e:
            raise RuntimeError(f"DWYL loader error: could not pull repo: {e}")

    alpha_path = os.path.join(dwyl_dir, "words_alpha.txt")
    if not os.path.isfile(alpha_path):
        raise RuntimeError(f"DWYL loader error: expected file not found: {alpha_path}")

    words = set()
    with open(alpha_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w and is_pure_alpha(w):
                words.add(w)
    if not words:
        raise RuntimeError("DWYL loader error: no valid words found in words_alpha.txt")
    print(f"  → {len(words)} entries loaded from DWYL (words_alpha.txt).")
    return words


################################################################################
# 2. SCOWL size 60 loader (raises on failure)
################################################################################

def load_scowl_words(tempdir):
    """
    Download SCOWL size 60 tarball over SourceForge, extract the correct
    “english-words.60” file (wherever it lives inside the archive),
    filter to a–z, and return a set of words.
    Raises RuntimeError on any failure.
    """
    print("\n→ Loading SCOWL size 60…")

    SCOWL_VERSION = "2020.12.07"
    SCOWL_URL = (
        f"https://downloads.sourceforge.net/project/wordlist/"
        f"SCOWL/{SCOWL_VERSION}/scowl-{SCOWL_VERSION}.tar.gz"
    )
    tarball_name = os.path.join(tempdir, f"scowl-{SCOWL_VERSION}.tar.gz")

    if not os.path.isfile(tarball_name):
        print(f"  • Downloading SCOWL {SCOWL_VERSION} from {SCOWL_URL} …")
        try:
            urllib.request.urlretrieve(SCOWL_URL, tarball_name)
        except Exception as e:
            raise RuntimeError(f"SCOWL download error: {e}")
    else:
        print(f"  • Found existing SCOWL tarball at {tarball_name}, skipping download.")

    try:
        with tarfile.open(tarball_name, "r:gz") as tar:
            target_member = None
            for member in tar.getmembers():
                if os.path.basename(member.name) == "english-words.60":
                    target_member = member.name
                    break

            if target_member is None:
                raise RuntimeError("SCOWL extraction error: 'english-words.60' not found inside tarball")

            extracted_txt = os.path.join(tempdir, "scowl-en_wl_60.txt")
            if not os.path.isfile(extracted_txt):
                print(f"  • Extracting '{target_member}' from SCOWL tarball…")
                with tar.extractfile(target_member) as f_in, \
                     open(extracted_txt, "w", encoding="utf-8") as f_out:
                    count = 0
                    for raw in f_in:
                        line = raw.decode("utf-8", "ignore").strip()
                        w = line.lower()
                        if is_pure_alpha(w):
                            f_out.write(w + "\n")
                            count += 1
                if count == 0:
                    raise RuntimeError("SCOWL extraction error: no valid words found in 'english-words.60'")
                print(f"  → Extracted and filtered {count} words to {extracted_txt}.")
            else:
                print(f"  • Previously extracted SCOWL file found at {extracted_txt}, skipping extraction.")
    except (tarfile.TarError, RuntimeError) as e:
        raise RuntimeError(f"SCOWL extraction error: {e}")

    extracted_txt = os.path.join(tempdir, "scowl-en_wl_60.txt")
    words = set()
    with open(extracted_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    if not words:
        raise RuntimeError("SCOWL loader error: failed to load any words from extracted file")
    print(f"  → {len(words)} entries loaded from SCOWL size 60.")
    return words


################################################################################
# 3. WordNet (NLTK) loader (raises on failure)
################################################################################

def load_wordnet_words():
    """
    Ensure WordNet is downloaded, then collect all lemma names that are purely a–z.
    Return as a set. Raises RuntimeError on failure.
    """
    print("\n→ Loading WordNet lemmas (via NLTK)…")
    try:
        wn.ensure_loaded()
    except LookupError:
        print("  • WordNet not found locally. Downloading via nltk.download('wordnet') …")
        try:
            nltk.download('wordnet', quiet=True)
            wn.ensure_loaded()
        except Exception as e:
            raise RuntimeError(f"WordNet download error: {e}")

    lemmas = set()
    try:
        for synset in wn.all_synsets():
            for lemma in synset.lemma_names():
                w = lemma.lower()
                if is_pure_alpha(w):
                    lemmas.add(w)
    except Exception as e:
        raise RuntimeError(f"WordNet iteration error: {e}")

    if not lemmas:
        raise RuntimeError("WordNet loader error: no lemmas found")
    print(f"  → {len(lemmas)} unique WordNet lemmas (a–z only).")
    return lemmas


################################################################################
# 4. wordfreq loader (raises on failure)
################################################################################

def load_wordfreq_words():
    """
    Use wordfreq.iter_wordlist("en") to retrieve every English word in descending frequency order,
    filter to a–z only, return as a set. Raises RuntimeError on failure.
    """
    print("\n→ Loading wordfreq’s entire English list…")
    words = set()
    try:
        for w in iter_wordlist("en"):
            w_lower = w.lower()
            if is_pure_alpha(w_lower):
                words.add(w_lower)
    except Exception as e:
        raise RuntimeError(f"wordfreq loader error: {e}")

    if not words:
        raise RuntimeError("wordfreq loader error: no words retrieved")
    print(f"  → {len(words)} entries loaded from wordfreq.")
    return words


################################################################################
# 5. Linux /usr/share/dict/words loader (no longer fatal if missing)
################################################################################

def load_linux_dict_words():
    """
    Read /usr/share/dict/words, filter to a–z, return as a set.
    If the file does not exist, print a “skipping” message and return an empty set.
    """
    path = "/usr/share/dict/words"
    if not os.path.isfile(path):
        print("\n→ Linux dictionary (/usr/share/dict/words) not found → skipping.")
        return set()

    print("\n→ Loading /usr/share/dict/words…")
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w and is_pure_alpha(w):
                words.add(w)

    if not words:
        print("→ /usr/share/dict/words exists but contained no valid a–z entries → skipping.")
        return set()

    print(f"  → {len(words)} entries loaded from /usr/share/dict/words.")
    return words


################################################################################
# 6. (Optional) Moby Word Lists loader (raises on failure if path is invalid)
################################################################################

def load_moby_words(moby_file):
    """
    If the user supplies a local Moby‐WordLists e-text (e.g. PG #3201), parse its
    “1. Standard English Words” section. Return a set of lowercase a–z words.
    Raises RuntimeError if the file is provided but cannot be parsed.
    """
    if not moby_file:
        print("\n→ Moby file not provided → skipping Moby.")
        return set()
    if not os.path.isfile(moby_file):
        raise RuntimeError(f"Moby loader error: file not found: {moby_file}")

    print(f"\n→ Loading Moby Word Lists from '{moby_file}'…")
    words = set()
    start_reading = False
    section_header_pattern = re.compile(r"^\s*2\.\s+Hyphenated Words", re.IGNORECASE)

    with open(moby_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ln = line.rstrip("\r\n")
            if not start_reading:
                if ln.strip().startswith("1. Standard English Words"):
                    start_reading = True
                continue
            if not ln.strip():
                break
            if section_header_pattern.match(ln):
                break
            w = ln.strip().lower()
            if w and is_pure_alpha(w):
                words.add(w)

    if not words:
        raise RuntimeError("Moby loader error: no valid words found in '1. Standard English Words' section")
    print(f"  → {len(words)} entries loaded from Moby Standard English section.")
    return words


################################################################################
# 7. Load omit-file (words to exclude)
################################################################################

def load_omit_words(omit_path: str):
    """
    Load words from the omit-file (one word per line, lowercase a–z).
    Returns a set of words to exclude.
    Exits if file not found.
    """
    try:
        with open(omit_path, "r", encoding="utf-8", errors="ignore") as f:
            omit_set = {
                line.strip().lower()
                for line in f
                if line.strip() and is_pure_alpha(line.strip().lower())
            }
    except FileNotFoundError:
        print(f"ERROR: could not open omit-file at '{omit_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n→ Loaded {len(omit_set)} words to omit from '{omit_path}'.")
    return omit_set


################################################################################
# MAIN: Combine Everything, Write Output, Then Build & Pickle Trie
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Create the union of multiple large English word-lists, with optional exclusions, and then build & pickle a trie."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output text file (one word per line). Will be overwritten if it exists."
    )
    parser.add_argument(
        "--tempdir", "-t", default=None,
        help="Directory to use for intermediate downloads/clones. If omitted, a temporary directory is created and removed on exit."
    )
    parser.add_argument(
        "--moby-file", "-m", default=None,
        help="(Optional) Path to local Moby Word Lists e-text (Project Gutenberg #3201)."
    )
    parser.add_argument(
        "--omit-file", "-x", default=None,
        help="(Optional) Path to text file with words to omit (one word per line)."
    )
    parser.add_argument(
        "--trie-output", "-r", default=None,
        help="(Optional) Path to write the pickled trie. Defaults to <output>.pkl"
    )
    args = parser.parse_args()

    # Determine temp directory
    if args.tempdir:
        tempdir = os.path.abspath(os.path.expanduser(args.tempdir))
        ensure_dir(tempdir)
        remove_tempdir = False
    else:
        tempdir = tempfile.mkdtemp(prefix="union_wordlists_")
        remove_tempdir = True

    print(f"Temporary workspace: {tempdir}")

    # If an omit-file was provided, load its words
    omit_set = set()
    if args.omit_file:
        omit_path = os.path.abspath(os.path.expanduser(args.omit_file))
        omit_set = load_omit_words(omit_path)

    try:
        # 1. DWYL (raises on failure)
        dwyl_set = load_dwyl_words(tempdir)

        # 2. SCOWL (raises on failure)
        scowl_set = load_scowl_words(tempdir)

        # 3. WordNet (raises on failure)
        wordnet_set = load_wordnet_words()

        # 4. wordfreq (raises on failure)
        wordfreq_set = load_wordfreq_words()

        # 5. Linux dict (no longer fatal if missing)
        linux_set = load_linux_dict_words()

        # 6. Moby (optional; raises if provided but fails)
        moby_set = load_moby_words(args.moby_file)

        # Union all sources
        print("\n→ Combining all sources into one union set…")
        total_union = set()
        for s, name in [
            (dwyl_set, "DWYL"),
            (scowl_set, "SCOWL"),
            (wordnet_set, "WordNet"),
            (wordfreq_set, "wordfreq"),
            (linux_set, "/usr/share/dict/words"),
            (moby_set, "Moby"),
        ]:
            before = len(total_union)
            total_union.update(s)
            after = len(total_union)
            print(f"  • Added {len(s)} from {name} → union size now {after} (+{after - before})")

        if not total_union:
            raise RuntimeError("Final union is empty; no words were loaded from any source.")

        print(f"\nTotal unique words in union before omission: {len(total_union)}")

        # 7. Remove omit-words
        if omit_set:
            before_omit = len(total_union)
            total_union.difference_update(omit_set)
            after_omit = len(total_union)
            print(f"  → Omitted {before_omit - after_omit} words; union size now {after_omit}")

        # 8. Write to output file, sorted alphabetically
        output_path = os.path.abspath(os.path.expanduser(args.output))
        print(f"\n→ Writing union to '{output_path}' …")
        with open(output_path, "w", encoding="utf-8") as outf:
            for w in sorted(total_union):
                outf.write(w + "\n")
        print("Done writing word list.")

        # 9. Build and pickle the trie
        # Decide where to write the pickle:
        if args.trie_output:
            trie_path = os.path.abspath(os.path.expanduser(args.trie_output))
        else:
            base, _ = os.path.splitext(output_path)
            trie_path = base + ".pkl"

        print(f"\n→ Building Trie from {len(total_union)} words …")
        root = TrieNode()
        for w in total_union:
            # Only insert pure a–z words
            root.insert(w)

        print(f"→ Serializing Trie to '{trie_path}' …")
        try:
            with open(trie_path, "wb") as f:
                pickle.dump(root, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_bytes = os.path.getsize(trie_path)
            with open(trie_path, "rb") as f:
                header = f.read(2)
                f.seek(-1, os.SEEK_END)
                trailer = f.read(1)
            print(f"  • Wrote trie pickle ({size_bytes} bytes).")
            print(f"  • Header bytes: {header!r}  Trailer byte: {trailer!r}")
            if not header.startswith(b'\x80') or trailer != b'.':
                print("Warning: Pickle header/trailer do not match expected values.", file=sys.stderr)
        except Exception as e:
            print(f"Error pickling trie: {e}", file=sys.stderr)
            sys.exit(1)

    finally:
        # Clean up tmp dir if created automatically
        if remove_tempdir:
            print(f"\nRemoving temporary workspace {tempdir} …")
            shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    main()
