# Sequence Batcher with GPT2

Sequence batcher ensures that all inference requests in a sequence get routed to the same model instance so that the model can maintain state correctly. For more details, please refer to [Trtion's documentation](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#stateful-models). GPT2 is an autoregressive model based on Transformer decoder architecture, pretrained on a very large corpus of English data. In this example, [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) from Huggingface is exploited.

1. Move into the working directory.
    ```sh
    cd /workspace/sequence_batcher_gpt2
    ```
2. Install required libraries and their dependencies.
    ```sh
    pip install -r requirements
    ```
3. Get GPT2 model from Huggingface. 

    Below script includes not only getting GPT2 tokenizer/model from Huggingface, but also **converting them into Torchscript and ONNX format**. The output models are then saved at *persistence* folder by default, and `model.pt`, `model.onnx` are copied to the corresponding model repository path under *models* folder. Model configs have been prepared in advance under *model* folder as well.
    ```sh
    sh scripts/setup_serving_models.sh
    ```

4. Launch Triton Inference Server.

    Below command should be executed at the working directory: */workspace/sequence_batcher_gpt2*. The flag `--log-verbose` may slower the performance due to log printing, but it shows details like which CORRID occupies which batch slots.
    ```sh
    tritonserver --model-repository=models --log-verbose=1
    ```

5. Execute the client.

    Four multiple clients try making requests and those requests are sequence-batched. **The default sequence batch strategy configured in this example is `Direct` mode**. That means, all the requests with the same CORRID will go to the same batch slot. You can also try changing the strategy with the `Oldest` mode by swapping some code lines uncommented in *models/tokenizer/config.pbtxt*.
    ```sh
    python3 multiple_clients.py
    ```



