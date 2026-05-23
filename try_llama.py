import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_id = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )

    prompt = "Explain the concept of diffusion models in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
