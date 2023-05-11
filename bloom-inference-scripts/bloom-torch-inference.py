# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


import math
import time
from argparse import ArgumentParser

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


t_start = time.time()

num_tokens = 100

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument(
    "--bf16", action="store_true", help="Whether to run the model in bf16 precision."
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument(
    "--benchmark", action="store_true", help="additionally run benchmark"
)
args = parser.parse_args()


### Model loading and instantiating on GPUs


model_name = args.name
if args.bf16:
    infer_dtype = torch.bfloat16
else:
    infer_dtype = torch.float

print(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=infer_dtype)

model = model.eval().to("cuda")

# Some models like GPT2 do not have a PAD token so we have to set it if necessary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

if args.benchmark:
    t_ready = time.time()


### Generate


print(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "def print_hello_world():",
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)


print(f"Generate args {generate_kwargs}")

inputs = input_sentences[: args.batch_size]


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(
        inputs, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [
        o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
    ]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
print("*** Running generate warmup")
_ = generate()

print("*** Running generate")
t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in generated:
    print(f"{'-'*60}\nin={i}\nout={o}\n")

### Benchmark

# benchmark it!
if args.benchmark:
    print("*** Running benchmark")

    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
    torch.cuda.synchronize()
    throughput = (time.time() - t0) / (total_new_tokens_generated)
    print(
        f"""
*** Performance stats:
Throughput per token including tokenize: {throughput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
"""
    )
