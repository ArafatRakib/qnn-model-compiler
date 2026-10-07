import os
import sys
import qai_hub as hub

def get_target_device():
    print("Fetching live device catalog from Qualcomm AI Hub...")
    all_devices = hub.get_devices()
    
    snapdragon_devices = []
    print("Available Snapdragon candidates in AI Hub pool:")
    for dev in all_devices:
        dev_info = f"{dev.name} {dev.attributes}".lower()
        # Filter for Snapdragon 8 series and modern flagships
        if any(k in dev_info for k in ["snapdragon", "s25", "s24", "s23", "8elite", "8gen", "sm8"]):
            snapdragon_devices.append(dev)
            print(f" - Found: '{dev.name}' | Attributes: {dev.attributes}")

    if not snapdragon_devices:
        print("Warning: No specific Snapdragon 8-series matched. Defaulting to first available device.")
        return all_devices[0]

    # Priority matching order (from newest generation down)
    priority_keywords = [
        "8 elite",
        "8elite",
        "sm8750",
        "sm8735",
        "s25",
        "s24",
        "8gen3",
        "snapdragon 8"
    ]

    for kw in priority_keywords:
        for dev in snapdragon_devices:
            dev_info = f"{dev.name} {dev.attributes}".lower()
            if kw in dev_info:
                print(f"\nSuccessfully selected target device: '{dev.name}'")
                return dev

    # Fallback to first available Snapdragon device
    selected = snapdragon_devices[0]
    print(f"\nFallback selected device: '{selected.name}'")
    return selected

def main():
    target_device = get_target_device()

    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    print(f"\nSubmitting ONNX model to Qualcomm AI Hub for {target_device.name} compilation...")
    
    # Submit compile job with 'precompiled_qnn_onnx' target runtime for ONNX graphs
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=target_device,
        options="--target_runtime precompiled_qnn_onnx"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_qnn.onnx"
    target_model.download(output_filename)
    
    print(f"\nCompilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
