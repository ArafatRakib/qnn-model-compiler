import os
import sys
import qai_hub as hub

def get_target_device():
    # 1. Primary lookup: Snapdragon 8s Gen 4 chipset attribute
    devices = hub.get_devices(attributes=["chipset:qualcomm-snapdragon-8s-gen-4"])
    
    # 2. Secondary lookup: Qualcomm SoC part number (SM8735)
    if not devices:
        devices = hub.get_devices(attributes=["chipset:qualcomm-sm8735"])

    # 3. Fallback: Target Snapdragon 8 Gen 3 family if 8s Gen 4 hardware isn't active
    if not devices:
        print("Snapdragon 8s Gen 4 tag not active in pool. Falling back to Snapdragon 8 series...")
        devices = hub.get_devices(attributes=["chipset:qualcomm-snapdragon-8-gen-3"])

    # 4. Emergency Fallback: Select first available device from AI Hub pool
    if not devices:
        all_devices = hub.get_devices()
        if not all_devices:
            print("Error: No available devices found in Qualcomm AI Hub.")
            sys.exit(1)
        devices = [all_devices[0]]

    target = devices[0]
    print(f"Successfully selected target device: {target.name}")
    return target

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
        options="--target_runtime qnn_context_binary"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_npu.bin"
    target_model.download(output_filename)
    
    print(f"Compilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
