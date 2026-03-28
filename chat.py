"""
Project 3: LLM Chat Interface
Interactive chat using Phi-3 Mini 3.8B with INT4 quantization.
Runs on NVIDIA GTX 1650 (4GB VRAM) using bitsandbytes NF4 quantization.
"""

import torch
import time
import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in explaining GPU computing, "
    "machine learning, and deep learning concepts. You give clear, concise answers."
)

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading Phi-3 Mini (INT4 quantized)...")
    print("This takes ~30 seconds on first load.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded! VRAM used: {vram:.2f} GB")
    return model, tokenizer


def generate_response(model, tokenizer, conversation_history, max_new_tokens=512):
    """Generate a response given the conversation history."""
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens  = output.shape[1] - input_len
    tps         = new_tokens / elapsed
    response    = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return response.strip(), tps, new_tokens


def chat():
    print("=" * 60)
    print("  Phi-3 Mini Chat — INT4 Quantized on GTX 1650")
    print("=" * 60)
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    model, tokenizer = load_model()

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nChat ready! Type your message (or 'quit' to exit, 'reset' to clear history)\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if user_input.lower() == "reset":
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  [Conversation history cleared]")
            continue
        if user_input.lower() == "stats":
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM in use: {vram:.2f} GB")
            print(f"  Messages in context: {len(conversation)}")
            continue

        conversation.append({"role": "user", "content": user_input})

        print("\nPhi-3: ", end="", flush=True)
        response, tps, ntok = generate_response(model, tokenizer, conversation)

        print(response)
        print(f"\n  [{ntok} tokens | {tps:.1f} tok/s]")

        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat()
