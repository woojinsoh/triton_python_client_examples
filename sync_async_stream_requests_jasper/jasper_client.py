import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import yaml, sys, queue, time, json, argparse
import numpy as np
import tritonclient.grpc, tritonclient.http

from functools import partial
from tritonclient.utils import np_to_triton_dtype, InferenceServerException
from utils.speech_utils import AudioSegment, postprocess, audio_generator_from_file_with_buffer
from utils.parse_utils import str2bool


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
        

def completion_callback(user_data, result, error):
    user_data._completed_requests.put((result, error))


def request_generator(prtc_client, batch_audio_samples, batch_max_num_audio_samples):
    inputs=[]    
    inputs.append(prtc_client.InferInput("AUDIO_SIGNAL", batch_audio_samples.shape, np_to_triton_dtype(batch_audio_samples.dtype)))
    inputs.append(prtc_client.InferInput("NUM_SAMPLES", batch_max_num_audio_samples.shape, np_to_triton_dtype(batch_max_num_audio_samples.dtype)))
    inputs[0].set_data_from_numpy(batch_audio_samples)
    inputs[1].set_data_from_numpy(batch_max_num_audio_samples)

    outputs=[]
    outputs.append(prtc_client.InferRequestedOutput("TRANSCRIPT"))
    
    return inputs, outputs


def audio_chunk_streaming_generator_from_file(prtc_client, filename, sample_rate, chunk_len, n_chunk_overlap):
    for audio_chunk in audio_generator_from_file_with_buffer(filename, sample_rate, chunk_len, n_chunk_overlap):
        #print("Chunk: shape: {}, {}, {}".format(chunk.shape, start, end))
        chunk = audio_chunk[0]
        start = audio_chunk[2]
        end = audio_chunk[3]
            
        chunk_audio_sample = np.expand_dims(chunk.astype("float16"), axis=0)
        chunk_max_num_audio_samples = np.expand_dims(np.full(batch_size, len(chunk_audio_sample[0])).astype("int32"), axis=1)

        inputs, outputs = request_generator(prtc_client, chunk_audio_sample, chunk_max_num_audio_samples)
                    
        yield inputs, outputs, start, end


def audio_batch_generator_from_file(prtc_client, batch_size, audio_idx):
    last_request = False
    batch_audio_samples=[]          # audio data for inference
    batch_num_audio_samples=[]      # the number of audio data for inference
    batch_filenames=[]              # audio data filename for inference
    batch_max_num_audio_samples = 0

    for _ in range(batch_size):
        filename = filenames[audio_idx]
        audio = AudioSegment.from_file(filename, offset=0, duration=0).samples.astype("float16")

        audio_idx = (audio_idx + 1) % len(filenames)
        if audio_idx == 0:
            last_request = True

        batch_audio_samples.append(audio) 
        batch_num_audio_samples.append(len(audio))
        batch_filenames.append(filename)
        
    max_num_samples = max([len(x) for x in batch_audio_samples])

    #Add Gauss noise to align the length of audio to the maximum
    batch_audio_samples = np.asarray(list(map(lambda x: np.concatenate((x, np.random.normal(np.mean(x), np.std(x), max_num_samples - len(x)))), batch_audio_samples))).astype("float16")  
    batch_max_num_audio_samples = np.expand_dims(np.full(batch_size, max_num_samples).astype("int32"), axis=1) 

    inputs, outputs = request_generator(prtc_client, batch_audio_samples, batch_max_num_audio_samples)

    return audio_idx, inputs, outputs, last_request


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, required=False, default=True, 
                        help='protocol (HTTP/gRPC) used to communicate with inference service. Default is HTTP')
    parser.add_argument('--batch-size', type=int, required=False, default=1, 
                        help='Batch size. it should be <= 8 for Jasper')
    parser.add_argument('--async-mode', type=str2bool, required=False, default=True, 
                        help='Enable asynchronous requests(True/False)')
    parser.add_argument('--streaming', type=str2bool, required=False, default=True, 
                        help='Enable Streaming requests(True/False). "--async-mode" should be set as True')
    
    FLAGS = parser.parse_args()
    
    # 1. Setup Client infos
    protocol = FLAGS.protocol.lower()
    async_mode = FLAGS.async_mode
    streaming = FLAGS.streaming
    batch_size = FLAGS.batch_size # must be <=8 for "jasper-onnx-ensemble"
    data_dir = "dataset/LibriSpeech"
    model_name = "jasper-onnx-ensemble"

    print("==")
    print("protocol: {}".format(protocol))
    print("async_mode: {}".format(async_mode))
    print("streaming: {}".format(streaming))
    print("batch size: {}".format(batch_size))
    
    if streaming:
        if protocol != "grpc": raise ValueError("streaming is only supported by grpc. Set '--protocol grpc'")
        if async_mode != True: raise ValueError("streaming requires asynchronous mode. Set '--async-mode' True")
        if batch_size != 1: raise ValueError("Streaming only supports single batch(i.e.,'--batch-size' must be 1)")
    if batch_size > 8: raise ValueError("batch size must be <= 8 for Jasper model")
    
    filter_speed = 1
    labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "<BLANK>"]
    if protocol == "grpc":
        prtc_client = tritonclient.grpc
        url = "localhost:8001"
    else:
        prtc_client = tritonclient.http
        url = "localhost:8000"
    try:
        triton_client = prtc_client.InferenceServerClient(url=url, verbose=False)
    except Exception as e:
        print("Failed to create the client: {}".format(e))
    print("Is the client alive? ", triton_client.is_server_live())
    print("==\n")

    # 2. Read Jasper model configs
    with open("config/jasper_10x5dr.yaml", 'r') as r:
        try:
            cfg = yaml.safe_load(r)
        except yaml.YAMLError as err:
            print(err)

    # 3. Initialize variables
    if async_mode:
        async_requests = []
        user_data = UserData()
    
    filenames = []
    transcripts = []
    manifest_file = os.path.join(data_dir, "librispeech-test-clean-wav.json")
    req_cnt = 0
    audio_idx = 0
    last_request = False
    seq_id = 1000

    # 4. Set up streaming variables 
    if streaming:
        sample_rate = cfg['model']['sample_rate']

        chunk_len = 1 # chunk size in seconds  
        n_chunk_len = int(chunk_len * sample_rate)

        chunk_overlap = 2 # duration of overlaps before and after current chunk in seconds 
        n_chunk_overlap = int(chunk_overlap * sample_rate)

        timestep_duration = cfg['model']['preprocessor']['window_stride']  
        for block in cfg['model']['encoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']

        n_timesteps_overlap = int(chunk_overlap / timestep_duration) - 2
        offset = 26 # number of symbols to drop for smooth streaming

        triton_client.start_stream(partial(completion_callback, user_data))
    
    # 5. Set up sample Dataset
    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifests = json.load(fh)
        for data in manifests:
            file_and_speeds = data["files"]
            audio_path = [x["fname"] for x in file_and_speeds if x["speed"] == filter_speed][0]
            filenames.append(os.path.join(data_dir, audio_path))
            transcript_text = data["transcript"]
            transcripts.append(transcript_text)
        print("A total of {} audio transcripts is loaded.".format(len(transcripts)))
    
    filenames = filenames[:120]
    print("{} audio files are chosen for simiplicity.".format(len(filenames)))
    print("Start inferenceing ...")
    time.sleep(3)
    
    # 6. Send requests
    while not last_request:
        # Online recognition
        if streaming:
            for filename in filenames:
                req_cnt += 1
                chunk_cnt = 0
                # Get audio chunk from the file, and send it to the server in a loop
                for inputs, outputs, start, end in audio_chunk_streaming_generator_from_file(prtc_client, filename, sample_rate, chunk_len, n_chunk_overlap):
                    try:
                        triton_client.async_stream_infer(model_name=model_name,
                                                        inputs=inputs,
                                                        request_id=str(req_cnt),
                                                        sequence_id=seq_id,
                                                        sequence_start= start,
                                                        sequence_end=end,
                                                        outputs=outputs)
                    except InferenceServerException as err:
                        print("Inference failed: " + str(err))
                    chunk_cnt += 1
                
                # As soon as the all audio chunks are sent to the server, the client side starts processing.
                prediction = None
                processed_cnt = 0
                while processed_cnt < chunk_cnt:
                    (result, error) = user_data._completed_requests.get()
                    processed_cnt += 1
                    if error is not None:
                        print(error)
                        print("Inference is failed at {}th request: ".format(processed_cnt))
                        sys.exit(1)
                    req_id = result.get_response().id
                    if prediction is None: # processing the first chunk
                        tokens = result.as_numpy("TRANSCRIPT")
                        valid_tokens = tokens[:, n_timesteps_overlap:-n_timesteps_overlap]
                        decoded = postprocess(valid_tokens[:,:len(valid_tokens[0]) - offset], labels)
                        # print(decoded) # for debug
                        prediction = decoded
                    else:
                        tokens = result.as_numpy("TRANSCRIPT")
                        if processed_cnt == chunk_cnt: # processing the last chunk
                            valid_tokens = tokens[:,n_timesteps_overlap:]
                            decoded = postprocess(valid_tokens, labels)
                        else: #processing the intermediate chunk   
                            valid_tokens = tokens[:,n_timesteps_overlap:-n_timesteps_overlap]
                            decoded = postprocess(valid_tokens[:,:len(valid_tokens[0]) - offset], labels)
                        # print(decoded) # for debug
                        prediction = np.concatenate((prediction, decoded), axis=1) # concat the decoded results for each chunk

                print("==")
                print("request_id: {}".format(req_id))
                print("batch_size: {}".format(batch_size))
                print("num_chunks_per_streaming: {}".format(chunk_cnt))
                print("transcripts: {}".format(''.join(prediction[0])))  
                last_request = True if req_cnt == len(filenames) else False
                seq_id += 1

        # Not streaming(i.e.,Offline recognition)
        else:
            # Get full audio signal as a batch from files.
            audio_idx, inputs, outputs, last_request = audio_batch_generator_from_file(prtc_client, batch_size, audio_idx)
            
            req_cnt += 1 
            stime = time.time()
            try:
                if async_mode:
                    print("async request: " + str(req_cnt))
                    if protocol == "grpc":
                        triton_client.async_infer(model_name=model_name,
                                                  inputs=inputs,
                                                  callback=partial(completion_callback, user_data),
                                                  request_id=str(req_cnt),
                                                  outputs=outputs)
                    else: # protocol == "http"
                        async_requests.append(
                            triton_client.async_infer(model_name=model_name,
                                                      inputs=inputs,
                                                      request_id=str(req_cnt),
                                                      outputs=outputs))
                              
                else: # sync_mode for http, grpc
                    result = triton_client.infer(model_name=model_name, 
                                                 inputs=inputs, 
                                                 request_id=str(req_cnt),
                                                 outputs=outputs)

                    # In a synchronous mode, the client processes the response for each request
                    etime = time.time()
                    if protocol == "grpc": 
                        req_id = result.get_response().id
                    else: 
                        req_id = result.get_response()['id']
                    prediction = result.as_numpy("TRANSCRIPT")
                    results = postprocess(prediction, labels)
                    print("==")
                    print("request_id: {}".format(req_id))
                    print("batch_size: {}".format(batch_size))
                    print("inference time: {} ms".format(etime - stime))
                    print("transcripts: {}".format(results))

            except InferenceServerException as err:
                print("Inference failed: " + str(err))

    if streaming:
        triton_client.stop_stream()    
        print('streaming is closed')   
    else:
        # In an asynchronous mode, the client processes the all responses at a time once the all requests are sent
        if protocol == "grpc":
            if async_mode:
                processed_cnt = 0
                while processed_cnt < req_cnt:
                    (result, error) = user_data._completed_requests.get()
                    processed_cnt += 1
                    if error is not None:
                        print("Inference is failed at {}th request: ".format(processed_cnt))
                        sys.exit(1)
                    req_id = result.get_response().id
                    prediction = result.as_numpy("TRANSCRIPT")
                    results = postprocess(prediction, labels)
                    print("==")
                    print("request_id: {}".format(req_id))
                    print("batch_size: {}".format(batch_size))
                    print("transcripts: {}".format(results)) 
                            
        else: # protocol == "http"
            if async_mode:
                for async_request in async_requests:
                    result = async_request.get_result()
                    req_id = result.get_response()['id']
                    prediction = result.as_numpy("TRANSCRIPT")
                    results = postprocess(prediction, labels)
                    print("==")
                    print("request_id: {}".format(req_id))
                    print("batch_size: {}".format(batch_size))
                    print("transcripts: {}".format(results)) 



    print("==")
    print("Inference is done.")

