import os
import sys
import shutil
import qai_hub as hub
from huggingface_hub import hf_hub_download

def clean_repo_id(repo_input):
    """Sanitize full Hugging Face URLs into clean 'namespace/repo' format."""
    repo = repo_input.strip()
    for prefix in ["https://huggingface.co/", "http://huggingface.co/"]:
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
    return repo.strip("/")

def get_target_device():
    print("Fetching device catalog from Qualcomm AI Hub...")
    all_devices = hub.get_devices()
    
    # Target modern Snapdragon 8-series / S25 targets
    for dev in all_devices:
        dev_str = f"{dev.name} {dev.attributes}".lower()
        if any(k in dev_str for k in ["s25", "8 elite", "8elite", "sm8750", "sm8735"]):
            print(f"Selected target device: '{dev.name}'")
            return dev

    for dev in all_devices:
        if "snapdragon" in dev.name.lower() or "s24" in dev.name.lower():
            print(f"Fallback target device: '{dev.name}'")
            return dev

    return all_devices[0]

def main():
    raw_repo_id = os.environ.get("HF_REPO_ID", "onnx-community/Qwen2.5-Coder-3B-Instruct")
    repo_id = clean_repo_id(raw_repo_id)
    
    model_filename = os.environ.get("HF_MODEL_FILE", "onnx/model_q4.onnx")
    data_filename = os.environ.get("HF_DATA_FILE", "onnx/model_q4.onnx_data")

    # 1. Create directory ending with .onnx required by Qualcomm AI Hub for external weights
    staging_dir = "./qwen_q4.onnx"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 2. Download and copy ONNX graph file into staging directory
    print(f"Downloading '{model_filename}' from Hugging Face...")
    cached_onnx = hf_hub_download(repo_id=repo_id, filename=model_filename)
    shutil.copyfile(cached_onnx, os.path.join(staging_dir, os.path.basename(model_filename)))

    # 3. Download and copy external weights file into the SAME staging directory
    if data_filename:
        print(f"Downloading '{data_filename}' from Hugging Face...")
        cached_data = hf_hub_download(repo_id=repo_id, filename=data_filename)
        shutil.copyfile(cached_data, os.path.join(staging_dir, os.path.basename(data_filename)))

    target_device = get_target_device()

    print(f"\nSubmitting ONNX model directory '{staging_dir}' to Qualcomm AI Hub for {target_device.name} compilation...")
    
    # Pass the staging DIRECTORY path so qai_hub uploads both model_q4.onnx and model_q4.onnx_data
    compile_job = hub.submit_compile_job(
        model=staging_dir,
        device=target_device,
        options="--target_runtime precompiled_qnn_onnx"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_qnn.onnx"
    target_model.download(output_filename)
    
    print(f"\nCompilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
