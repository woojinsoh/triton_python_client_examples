# Sync/Async/Streaming Requests with Jasper

Triton supports sync/async/streaming requests to serve models. This simple example implements Triton's several request methods via HTTP/gRPC protocols for online/offline inference(e.g.,speech recognition). Audio signals from LibriSpeech dataset are sent to the Triton server. Then, [NVIDIA Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) ensemble model processes those singals and gives reponses back to the client. You can see how it's implemented through `jasper_client.py`.

**Please note that**, this example just shows how the sync/async/streaming request mechanism should be implemented using Triton python client APIs(i.e.,it would not be optimized in terms of performance). In addition, all commands below should be executed inside the Triton container at `/workspace` where `<local_full_path>` is mounted in order to avoid directory hierarchy errors.

1. Move into the working directory.
    ```sh
    cd /workspace/sync_async_stream_requests_jasper
    ```

2. Install required libraries and their dependencies.

    Below script includes updating apte-get, installing ubuntu/python libraries and their dependencies.
    ```sh 
    sh scripts/install_dependencies.sh
    ````
3. Prepare for dataset.

    [Librispeech](https://www.openslr.org/12) is very popular benchmark dataset in the area of speech recognition. Here, we only take audio files(Flac Format) in `test-clean.tar.gz` for simplicity. Below script includes not only downloading the dataset but also converting Flac to wav format while creating the manifest.
    ```sh
    sh scripts/prepare_librispeech.sh
    ```

4. Deploy the model.
    
    The [ONNX](https://github.com/onnx/onnx) version of [NVIDIA Jasper](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) is used as a serving model. You can directly download this model via NGC. Below scripts include downloading the models from NGC, unzipping and arranging them to the Triton model repository. Then, Triton model management API is used to load the model.
    ```sh
    sh scripts/deploy_jasper.sh
    ```

5. Launch Triton Inference Server.
    
    **Note that** below command should be executed at the working directory(i.e.,*/workspace/sync_async_stream_requests_jasper*).
    ```sh
    tritonserver --model-repository=models #--model-control-mode=explicit --strict-model-config=false
    ```
    

6. Execute the client script.

    Once Triton Inference Server is launched, open the new terminal window and get into this container using `docker exec -it <container_id> /bin/bash` command. Then move into the working directory(i.e.,*/workspace/syunc_async_stream_requests_jasper*). The script below simply executes `jasper_client.py` with four configurable input arguments.
    
    - `--protocol` (*str*): Set request protocol. It must be *grpc* or *http*. The default value is *grpc*.
    - `--async-mode` (*bool*): Set *True* if you want to send requests **asynchronously**. The default value is *True*.
    - `--streaming` (*bool*): Set *True* if you want to send **streaming** requests **asynchronously**. The default value is *True*.
    - `--batch-size` (*int*): Set batch size. It must be **between *1* and *8*** since Jasper is limited to support the batch size up to 8. The default value is *1* (i.e., single batch).
    
    ```sh
    sh scripts/async_streaming_grpc_client.sh
    ```
    By replacing some configurable arguments, you can also try implementing synchrnous or asynchronous requests without streaming mode for the same dataset/model.
