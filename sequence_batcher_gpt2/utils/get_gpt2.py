import torch
import torch.nn.functional as F
import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed
from onnxruntime import InferenceSession


def gpt2_from_hf(path):
    """get GPT2 tokenizer and model from Huggingface"""
    # Download BPE
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(path)
    # # Download GPT2
    model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
    model.save_pretrained(path)
    return tokenizer, model

class TorchScriptWrapper(torch.nn.Module):
    """in order to convert Tuple output into Tensor"""
    def __init__(self, ts_path):
        super(TorchScriptWrapper, self).__init__()
        self.model = torch.jit.load(ts_path)
    
    def forward(self, input_ids):
        return self.model.forward(input_ids)[0]


def convert_hf_into_ts(tokenizer, model, input_ids, save_path):
    """convert huggingface GPT2 model into TorchScript. attention_mask is skipped."""
    ts_path = os.path.join(save_path,"temp.pt")
    wrapped_ts_path = os.path.join(save_path, "model.pt")
    
    traced_model = torch.jit.trace(model, input_ids)
    torch.jit.save(traced_model, ts_path)
    print(">> torchscript persisted in {}".format(ts_path))  

    wrapper = TorchScriptWrapper(ts_path)
    wrapped_model = torch.jit.trace(wrapper, input_ids)
    wrapped_model.save(wrapped_ts_path)
    print(">> torchscript with wrapper persisted in {}".format(wrapped_ts_path))  

    return wrapped_ts_path
    
    
def convert_ts_into_onnx(input_ids, ts_path, save_path):
    """convert TorchScript into ONNX"""
    loaded_model = torch.jit.load(ts_path)
    loaded_model.eval()
    onnx_path= os.path.join(save_path, "model.onnx")
    # Export the model
    torch.onnx.export(
        loaded_model,
        input_ids,
        onnx_path,
        export_params=True,
        opset_version=13,
        input_names=['INPUT_TOKEN_IDS'],
        output_names=['OUTPUT_LOGITS'],
        dynamic_axes={
                    'INPUT_TOKEN_IDS':{
                        0: 'batchsize',
                        1: 'seq_len'
                    },
                    'OUTPUT_LOGITS':{
                        0: 'batchsize',
                        1: 'seq_length',
                    }
        })
    print(">> ONNX is converted at {}".format(onnx_path))
    return onnx_path


def infer_test(tokenizer, model, dummy_input_ids):
    generated_text=[]
    output_length = 18
    context = dummy_input_ids
    generated = context
    
    with torch.no_grad():
        for _ in range(output_length):
            outputs, _ = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    for token_ids in generated:
        tokens = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        generated_text.append(''.join(tokens))
    print("=== huggingface model generated sentence ===")
    print(generated_text)


def infer_test_torchscript(tokenizer, ts_path, dummy_input_ids):
    generated_text=[]
    output_length = 18
    context = dummy_input_ids
    generated = context
    loaded_model = torch.jit.load(ts_path)
    loaded_model.eval()

    for _ in range(output_length):
        outputs = loaded_model(generated)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    for token_ids in generated:
        tokens = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        generated_text.append(''.join(tokens))
    print("=== torchscript model generated sentence ===")
    print(generated_text)


def infer_test_onnx(tokenizer, onnx_path, dummy_input_ids):
    session = InferenceSession(onnx_path)
    generated_text=[]
    output_length = 18
    context = dummy_input_ids
    generated = context
    for _ in range(output_length):    
        onnx_output = session.run(output_names=['OUTPUT_LOGITS'], input_feed={'INPUT_TOKEN_IDS':generated.tolist()})[0]
        next_token_logits = torch.tensor(onnx_output[:, -1, :])
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    for token_ids in generated:
        tokens = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        generated_text.append(''.join(tokens))
    print("=== onnx model generated sentence ===")
    print(generated_text)


save_path = "persistence"
tokenizer, model = gpt2_from_hf(save_path)
dummy_text = ["I have lived in London,", "The Whiteman is standing at"]
dummy_input = tokenizer(dummy_text, return_tensors='pt', padding=True, truncation=False)
dummy_input_ids = dummy_input['input_ids']

model.eval()
infer_test(tokenizer, model, dummy_input_ids)

ts_path = convert_hf_into_ts(tokenizer, model, dummy_input_ids, save_path)
infer_test_torchscript(tokenizer, ts_path, dummy_input_ids)

onnx_path = convert_ts_into_onnx(dummy_input_ids, ts_path, save_path)
infer_test_onnx(tokenizer, onnx_path, dummy_input_ids)

session = InferenceSession(onnx_path)
onnx_output = session.run(output_names=['OUTPUT_LOGITS'], input_feed={'INPUT_TOKEN_IDS':dummy_input_ids.tolist()})
base_output = model(dummy_input_ids)[0].detach().cpu().numpy()
np.testing.assert_allclose(base_output, onnx_output[0], rtol=1e-03, atol=1e-05)
print(">> Exported ONNX model has been tested with ONNXRuntime, and the result looks good.")
