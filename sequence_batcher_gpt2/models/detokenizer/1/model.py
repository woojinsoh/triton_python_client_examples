"""GPT2 Tokenizer on Python Backend """
import json
import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils

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
        self.parameters = self.model_config['parameters']

        # # Get OUTPUT0 configuration
        self.output0_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT_TOKENS")
        
        # # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(self.output0_config['data_type'])

        self.vocab_file = self.parameters['vocab_path']['string_value']
        with open(self.vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        
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
            decoded_next_token = []
            # Every Python backend must iterate over everyone of the requests
            # and create a pb_utils.InferenceResponse for each of them.
            for request in requests:
                # Get INPUT0
                in_0 = torch.tensor(pb_utils.get_input_tensor_by_name(request, "INPUT_LOGITS").as_numpy())
                next_token_logits = in_0[:,-1,:]
                next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

                next_token = [self.decoder[int(t)] for t in next_token_id]
                for token in next_token:
                    text = bytearray([self.byte_decoder[c] for c in token]).decode("utf-8", errors="replaces")
                    decoded_next_token.append(text)
                
                out_0 = np.array(decoded_next_token)
                
                # Create output tensors. You need pb_utils.Tensor
                # objects to create pb_utils.InferenceResponse.
                out_tensor_0 = pb_utils.Tensor("OUTPUT_TOKENS", out_0.astype(object))

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
