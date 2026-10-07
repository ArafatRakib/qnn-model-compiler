import os
import sys
import qai_hub as hub

def main():
    token = os.environ.get("QAI_HUB_TOKEN")
    if not token:
        print("Error: QAI_HUB_TOKEN environment variable not found.")
        sys.exit(1)

    hub.configure(api_token=token)

    # Target Snapdragon 8s Gen 4 Hexagon NPU specifically
    try:
        device = hub.Device("Snapdragon 8s Gen 4")
    except Exception:
        # Fallback attribute search if exact string alias differs in API
        device = hub.Device(attributes="chipset:snapdragon-8s-gen-4")

    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    print(f"Submitting model to Qualcomm AI Hub for {device.name} HTP compilation...")
    
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=device,
        options="--target_runtime qnn_context_binary"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_npu.bin"
    target_model.download(output_filename)
    
    print(f"Compilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
