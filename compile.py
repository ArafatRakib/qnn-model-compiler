import os
import sys
import qai_hub as hub

def get_target_device():
    # Target same-generation Adreno 800 / Hexagon 8-series targets (Snapdragon 8s Gen 4 / 8 Elite)
    device_candidates = [
        "Snapdragon 8s Gen 4",
        "Snapdragon 8 Elite",
        "Snapdragon 8 Gen 4"
    ]
    
    for dev_name in device_candidates:
        try:
            devices = hub.get_devices(name=dev_name)
            if devices:
                print(f"Selected AI Hub target device: {devices[0].name}")
                return devices[0]
        except Exception:
            continue

    # Fallback to same-gen SoC tags (SM8735 for 8s Gen 4, SM8750 for 8 Elite)
    chipset_tags = [
        "chipset:qualcomm-sm8735",
        "chipset:qualcomm-sm8750",
        "chipset:qualcomm-snapdragon-8-elite"
    ]
    
    for tag in chipset_tags:
        try:
            devices = hub.get_devices(attributes=[tag])
            if devices:
                print(f"Selected AI Hub device via chipset ({tag}): {devices[0].name}")
                return devices[0]
        except Exception:
            continue

    print("Error: Could not locate a valid Snapdragon 8 Elite or 8s Gen 4 target device in AI Hub.")
    sys.exit(1)

def main():
    target_device = get_target_device()

    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)

    print(f"Submitting ONNX model to Qualcomm AI Hub for {target_device.name} compilation...")
    
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=target_device,
        options="--target_runtime precompiled_qnn_onnx"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_qnn.onnx"
    target_model.download(output_filename)
    
    print(f"Compilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
