"""GPT2 Tokenizer on Python Backend """
import json
import regex as re
import numpy as np
import os

import triton_python_backend_utils as pb_utils

class Tokenizer():
    def __init__(
        self,
        errors="replace",
        add_prefix_space=False,
        **kwargs
    ):
        vocab_file = "persistence/vocab.json"
        merges_file = "persistence/merges.txt"
        self.unk_token="<|endoftext|>"
        self.pad_token = self.unk_token

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, inputs):
        encoded_inputs={}
        texts=[]
        input_ids = []
        attention_mask = []
        max_length = 0

        if isinstance(inputs, list):
            texts = inputs
        else:
            texts.append(inputs)

        for text in texts:
            tokens = self._tokenize(text)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            if max_length < len(token_ids):
                max_length = len(token_ids)
            input_ids.append(token_ids)
            attention_mask.append([1 for _ in range(len(token_ids))])

        input_ids_padded = [ids + [self._convert_token_to_id(self.pad_token) for _ in range(max_length - len(ids))] for ids in input_ids]
        attention_mask_padded = [m + [0 for _ in range(max_length - len(m))] for m in attention_mask]

        encoded_inputs['input_ids'] = np.array(input_ids_padded, dtype=np.int32)
        encoded_inputs['attention_mask'] = np.array(attention_mask_padded, dtype=np.int32)

        return encoded_inputs

    def _tokenize(self, text):
        """Tokenize a string"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def get_pairs(self, word):
        """
        Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        """Byte Pair Encoding"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        self.max_batch_size = self.model_config['max_batch_size']

    def execute(self, requests):
            """`execute` must be implemented in every Python model. `execute`
            function receives a list of pb_utils.InferenceRequest as the only
            argument. This function is called when an inference is requested
            for this model. Depending on the batching configuration (e.g. Dynamic
            Batching) used, `requests` may contain multiple requests. Every
            Python model, must create one pb_utils.InferenceResponse for every
            pb_utils.InferenceRequest in `requests`. If there is an error, you can
            set the error argument when creating a pb_utils.InferenceResponse.
            Parameters
            ----------
            requests : list
            A list of pb_utils.InferenceRequest
            Returns
            -------
            list
            A list of pb_utils.InferenceResponse. The length of this list must
            be the same as `requests`
            """
            
            responses = []

            tokenizer = Tokenizer()
            # Every Python backend must iterate over everyone of the requests
            # and create a pb_utils.InferenceResponse for each of them.
            for request in requests:
                is_seq_start = (pb_utils.get_input_tensor_by_name(request, "START").as_numpy() == 1)
                is_seq_end = (pb_utils.get_input_tensor_by_name(request, "END").as_numpy() == 1)
                corr_id = pb_utils.get_input_tensor_by_name(request, 'CORR_ID').as_numpy()
                print(is_seq_start, is_seq_end, corr_id)
                
                # Get INPUT0
                in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_TOKENS").as_numpy().tolist()
                # decode byte-object to string
                in_0 = [i.decode("utf-8") for i in in_0]
                encoded_inputs=tokenizer.encode(in_0)

                out_0 = encoded_inputs['input_ids'] 
 
                # Create output tensors. You need pb_utils.Tensor
                # objects to create pb_utils.InferenceResponse.
                out_tensor_0 = pb_utils.Tensor("ENCODED_INPUT_TOKENS", out_0.astype(np.int64))

 
                # Create InferenceResponse. You can set an error here in case
                # there was a problem with handling this inference request.
                # Below is an example of how you can set errors in inference
                # response:
                #
                # pb_utils.InferenceResponse(
                #    output_tensors=..., TritonError("An error occured"))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor_0])
                responses.append(inference_response)

            # You should return a list of pb_utils.InferenceResponse. Length
            # of this list must match the length of `requests` list.
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')



# ## For Debug
# if __name__ == "__main__":
#     context = ["hello guys nice to see you", "hi there"]

#     tokenizer = Tokenizer()
#     output=tokenizer.encode(input_data)
#     print(output)