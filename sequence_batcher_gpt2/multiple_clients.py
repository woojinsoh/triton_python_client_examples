from multiprocessing import Process
from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np

def triton_client(string, corr_id):
    with httpclient.InferenceServerClient("localhost:8000", verbose=False) as client:
        output_length = 10
        context = [string]
        # It should expand dim to have shape like [batch_size, 1] since Triton doesn't support the shape [batch_size] as an input with dynamic shape
        # Reshape in config.pbtxt of tokenizer will change its input dimension from [batch_size, 1] to [batch_size]
        generated = np.expand_dims(np.array(context, dtype=object), axis=1)

        seq_start=True
        seq_end=False
        corr_id= corr_id

        for req_cnt in range(output_length):
            if req_cnt == (output_length - 1):
                seq_end=True

            input_data = generated
            inputs = [httpclient.InferInput("INPUT_TOKENS", input_data.shape, np_to_triton_dtype(input_data.dtype))]
            inputs[0].set_data_from_numpy(input_data)

            outputs = [httpclient.InferRequestedOutput("NEXT_TOKENS")]

            response = client.infer(
                                    model_name="ensemble_pipeline",
                                    model_version="1",
                                    inputs=inputs,
                                    outputs=outputs,
                                    sequence_id=corr_id,
                                    sequence_start=seq_start,
                                    sequence_end=seq_end)
            seq_start=False
            result = response.get_response()
            output_tokens = response.as_numpy("NEXT_TOKENS")
            next_tokens = np.expand_dims(np.array([t.decode('utf-8') for t in output_tokens.tolist()], dtype=object), axis=1)
            print("corr_id:" + str(corr_id) + "::" + str(next_tokens))

            generated = generated + next_tokens
            
        print("output:: " + generated)

if __name__ == "__main__":
    processes = []
    processes.append(
        Process(
            target=triton_client,
            args=("I have lived in London", 1001)
        ))
    processes.append(
        Process(
            target=triton_client,
            args=("I have lived in Paris", 1002)
        ))
    processes.append(
        Process(
            target=triton_client,
            args=("I have lived in Berlin", 1003)
        ))
    processes.append(
        Process(
            target=triton_client,
            args=("I have lived in Seoul", 1004)
        ))
    
for p in processes:
    p.start()
for p in processes:
    p.join()

