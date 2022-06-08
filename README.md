# Triton Python Client Examples

This repo deals with several Triton python client examples for novice users as a reference. Most of the examples are simply focused on demonstrating some basics and important functionalitiy of Triton(i.e.,performance might not be severely considered). On a side note, you can also find some Trtion python client examples from the official [Triton git repo](https://github.com/triton-inference-server/client/tree/main/src/python/examples).

As a prerequisite step for the demonstaration, I recommend **cloning this repo to your `<local_full_path>`** first to be free from unexpected errors. Then, launch the Triton container from NGC using the docker command such as:
```sh
docker run --gpus all --network host -v <local_full_path>:/workspace \
nvcr.io/nvidia/tritonserver:22.03-py3 
```
`<local_full_path>` is a local directory where you have cloned this repository. This path is mounted on the Triton container as a docker volume. In addition, host mode networking is configured(see `--network host`) for simplicity. I guess the launching command would be able to be varied according to your environment(e.g.,even though Triton *v22.03* is used in this repo, you could pick a more advanced version).

## Example scripts
1. [Sync/Async/Streaming Requests with Jasper](https://github.com/woojinsoh/triton_python_client_examples/tree/master/sync_async_stream_requests_jasper)
2. [Sequence Batcher with GPT2](https://github.com/woojinsoh/triton_python_client_examples/tree/master/sequence_batcher_gpt2)