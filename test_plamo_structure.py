"""
Script to inspect the internal structure and API of Plamo-2.1-2B-VL.
Run this before implementation to understand Plamo's exact API.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def tensor_to_pil(image_tensor):
    """Convert (3, H, W) float32 [0,1] tensor to PIL Image"""
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def inspect_plamo():
    print("=" * 80)
    print("Plamo-2.1-2B-VL Structure Inspection")
    print("=" * 80)

    # Load model and processor
    print("\n[1] Loading model and processor...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "pfnet/plamo-2.1-2b-vl",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    try:
        processor = AutoProcessor.from_pretrained(
            "pfnet/plamo-2.1-2b-vl",
            trust_remote_code=True,
        )
        print("✓ Processor loaded successfully")
    except Exception as e:
        print(f"✗ Processor loading failed: {e}")
        processor = None

    # Model structure
    print("\n[2] Model Structure:")
    print(f"  Model type: {type(model)}")

    # Try to get hidden size from various possible attributes
    hidden_size = None
    for attr in ["hidden_size", "d_model", "dim", "hidden_dim"]:
        if hasattr(model.config, attr):
            hidden_size = getattr(model.config, attr)
            print(f"  Hidden size ({attr}): {hidden_size}")
            break
    if hidden_size is None:
        print(f"  Hidden size: Not found in standard attributes")

    # Try to get vocab size
    vocab_size = None
    for attr in ["vocab_size", "vocab_size"]:
        if hasattr(model.config, attr):
            vocab_size = getattr(model.config, attr)
            print(f"  Vocab size: {vocab_size}")
            break
    if vocab_size is None:
        print(f"  Vocab size: Not found in standard attributes")

    # Print all config attributes for inspection
    print("\n  Available config attributes:")
    for attr in dir(model.config):
        if not attr.startswith("_"):
            try:
                val = getattr(model.config, attr)
                if not callable(val):
                    print(f"    - {attr}: {val}")
            except:
                pass

    # Main modules
    print("\n[3] Main Model Modules:")
    for name, module in model.named_children():
        print(f"  - {name}: {type(module).__name__}")

    # Vision-related modules
    print("\n[4] Vision-related Modules:")
    vision_found = False
    for name, module in model.named_modules():
        if "vision" in name.lower():
            print(f"  - {name}: {type(module).__name__}")
            vision_found = True
    if not vision_found:
        print("  ⚠ No modules containing 'vision' found in name")

    # Processor capabilities
    if processor is not None:
        print("\n[5] Processor Capabilities:")
        print(f"  Processor type: {type(processor)}")
        print(f"  Has image_processor: {hasattr(processor, 'image_processor')}")
        print(f"  Has tokenizer: {hasattr(processor, 'tokenizer')}")

        if hasattr(processor, "image_processor"):
            ip = processor.image_processor
            print(f"    - Image processor type: {type(ip).__name__}")
            if hasattr(ip, "size"):
                print(f"    - Input size: {ip.size}")

    # Test with dummy inputs
    print("\n[6] Test with Dummy Inputs:")
    try:
        dummy_images = torch.randn(1, 3, 384, 384)
        print(f"  Input image shape: {dummy_images.shape}")

        if processor is not None:
            print("  Processing images with processor...")
            # Plamo processor expects PIL Images, not tensors
            # Convert (B, 3, H, W) float32 [0,1] tensor to PIL Images
            pil_images = [tensor_to_pil(dummy_images[i]) for i in range(dummy_images.shape[0])]
            text_list = [""] * len(pil_images)  # Dummy text
            processed = processor(images=pil_images, text=text_list, return_tensors="pt")
            print(f"  ✓ Processor output keys: {list(processed.keys())}")
            for key, val in processed.items():
                if isinstance(val, torch.Tensor):
                    print(f"    - {key}: shape={val.shape}, dtype={val.dtype}")

        # Model inference
        print("  Running model inference...")
        with torch.no_grad():
            if processor is not None and "processed" in locals():
                try:
                    outputs = model(**processed)
                except Exception as e:
                    print(f"    Error with processed output: {e}")
                    outputs = None
            else:
                # Fallback: raw input_ids if processor not available
                dummy_input_ids = torch.randint(0, 1000, (1, 10))
                try:
                    outputs = model(input_ids=dummy_input_ids)
                except Exception as e:
                    print(f"    Error with input_ids: {e}")
                    outputs = None

        if outputs is None:
            print("  ✗ Model inference failed")
            return

        print(f"  ✓ Model output keys: {list(outputs.__dict__.keys())}")
        if hasattr(outputs, "logits"):
            print(f"    - logits shape: {outputs.logits.shape}")
        if hasattr(outputs, "hidden_states"):
            print(f"    - hidden_states available: Yes")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()

    # Tokenizer info
    print("\n[7] Tokenizer Information:")
    if processor is not None and hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        if hasattr(tokenizer, "eos_token_id"):
            print(f"  EOS token ID: {tokenizer.eos_token_id}")
        if hasattr(tokenizer, "pad_token_id"):
            print(f"  PAD token ID: {tokenizer.pad_token_id}")

    print("\n" + "=" * 80)
    print("Inspection Complete")
    print("=" * 80)

    # Recommended adjustments
    print("\n[8] Recommended Adjustments:")
    print(
        "  1. Adjust _combine_modalities() in model_plamo.py based on verified structure"
    )
    print("  2. Verify and adjust image preprocessing params (size, normalization)")
    print("  3. Update forward() to match processor output format")
    print("  4. Verify vision encoder freezing approach")


if __name__ == "__main__":
    inspect_plamo()
