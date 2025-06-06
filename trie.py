class TrieNode:
    def __init__(self):
        # children: dict mapping single‐char (a–z) → TrieNode
        self.children = {}
        # is_word: True if the path from root down to here spells a valid word
        self.is_word = False

    def insert(self, word: str):
        """
        Insert `word` into this trie.  Assumes `word` is already all lowercase [a–z]+
        """
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True