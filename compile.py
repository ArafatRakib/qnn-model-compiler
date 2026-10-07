import os
import sys
import shutil
import onnx
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

    # 1. Create directory ending with .onnx required by Qualcomm AI Hub
    staging_dir = "./qwen_q4.onnx"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 2. Download ONNX graph file from Hugging Face
    print(f"Downloading '{model_filename}' from Hugging Face...")
    cached_onnx = hf_hub_download(repo_id=repo_id, filename=model_filename)
    staged_onnx_path = os.path.join(staging_dir, os.path.basename(model_filename))

    # 3. Handle external weights file and convert extension to .data
    if data_filename:
        print(f"Downloading '{data_filename}' from Hugging Face...")
        cached_data = hf_hub_download(repo_id=repo_id, filename=data_filename)
        
        old_data_name = os.path.basename(data_filename)
        # Convert extension from .onnx_data to .data for qai_hub compliance
        if old_data_name.endswith(".onnx_data"):
            new_data_name = old_data_name[:-10] + ".data"
        elif not old_data_name.endswith(".data"):
            new_data_name = old_data_name + ".data"
        else:
            new_data_name = old_data_name

        staged_data_path = os.path.join(staging_dir, new_data_name)
        shutil.copyfile(cached_data, staged_data_path)

        # Update ONNX graph protobuf to point to the renamed .data weight file
        print(f"Updating ONNX external data pointers from '{old_data_name}' to '{new_data_name}'...")
        onnx_model = onnx.load(cached_onnx, load_external_data=False)

        def fix_external_data_pointers(graph):
            for init in graph.initializer:
                for ext in init.external_data:
                    if ext.key == "location" and ext.value == old_data_name:
                        ext.value = new_data_name
            for node in graph.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        fix_external_data_pointers(attr.g)
                    for g in attr.graphs:
                        fix_external_data_pointers(g)

        fix_external_data_pointers(onnx_model.graph)
        onnx.save(onnx_model, staged_onnx_path, save_as_external_data=False)
    else:
        shutil.copyfile(cached_onnx, staged_onnx_path)

    target_device = get_target_device()

    print(f"\nSubmitting ONNX model directory '{staging_dir}' to Qualcomm AI Hub for {target_device.name} compilation...")
    
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
