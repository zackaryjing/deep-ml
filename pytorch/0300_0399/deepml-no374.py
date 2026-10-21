# problem: Character-Level Tokenizer (stoi/itos/BOS)

class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.

        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        self.stoi = {}
        sorted_text = sorted(text)
        cur_index = 2
        for s in sorted_text:
            if s not in self.stoi:
                self.stoi[s] = cur_index
                cur_index += 1
        self.itos = {}
        for key,value in self.stoi.items():
            self.itos[value] = key
        self.vocab_size = len(self.stoi) + 2


    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.

        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        # Your code here
        return [0] + [self.stoi[c] for c in text] + [1]

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.

        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here

        return "".join([self.itos[i] for i in indices[1:-1]])



def main():
    text = 'hello'
    tok = CharTokenizer(text)
    code = tok.encode('hello')
    print(code)
    print(tok.decode(code))

if __name__ == "__main__":
    main()

#
# Created By jing At 2026-10-18 23:45:55
#
