import os
import sys
import shutil
import onnx
from onnx import helper, TensorProto
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
    
    # Target modern Snapdragon 8-series / S25 targets (Snapdragon 8 Elite / 8s Gen 4 generation)
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

def get_tensor_elem_type(graph, tensor_name):
    """Find data element type of a tensor from value_info, input, or output."""
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if info.name == tensor_name and info.type.HasField("tensor_type"):
            return info.type.tensor_type.elem_type
    return TensorProto.FLOAT16  # Default fallback for quantized models

def decompose_simplified_layer_norm_in_place(model):
    """Decompose com.microsoft:SimplifiedLayerNormalization and RMSNorm into standard ONNX ops in-place."""
    graph = model.graph
    ordered_nodes = []
    new_initializers = []
    counter = 0

    for node in graph.node:
        if node.op_type in ["SimplifiedLayerNormalization", "RMSNorm"]:
            x_input = node.input[0]
            w_input = node.input[1]
            y_output = node.output[0]
            
            epsilon = 1e-5
            axis = -1
            for attr in node.attribute:
                if attr.name == "epsilon":
                    epsilon = attr.f
                elif attr.name == "axis":
                    axis = attr.i
            
            elem_type = get_tensor_elem_type(graph, x_input)
            prefix = f"sln_decomp_{counter}_"
            counter += 1
            
            eps_name = prefix + "eps"
            eps_tensor = helper.make_tensor(
                name=eps_name,
                data_type=elem_type,
                dims=[],
                vals=[epsilon]
            )
            new_initializers.append(eps_tensor)
            
            x_sq = prefix + "x_sq"
            mean_sq = prefix + "mean_sq"
            mean_eps = prefix + "mean_eps"
            rms = prefix + "rms"
            norm = prefix + "norm"
            
            # Standard RMSNorm math decomposition
            node_mul_sq = helper.make_node("Mul", inputs=[x_input, x_input], outputs=[x_sq], name=prefix+"mul_sq")
            node_red = helper.make_node("ReduceMean", inputs=[x_sq], outputs=[mean_sq], axes=[axis], keepdims=1, name=prefix+"red_mean")
            node_add = helper.make_node("Add", inputs=[mean_sq, eps_name], outputs=[mean_eps], name=prefix+"add_eps")
            node_sqrt = helper.make_node("Sqrt", inputs=[mean_eps], outputs=[rms], name=prefix+"sqrt")
            node_div = helper.make_node("Div", inputs=[x_input, rms], outputs=[norm], name=prefix+"div")
            node_mul_w = helper.make_node("Mul", inputs=[norm, w_input], outputs=[y_output], name=prefix+"mul_w")
            
            ordered_nodes.extend([node_mul_sq, node_red, node_add, node_sqrt, node_div, node_mul_w])
        else:
            ordered_nodes.append(node)

    if counter > 0:
        print(f"Decomposed {counter} custom normalization nodes in-place (topological order preserved)...")
        del graph.node[:]
        graph.node.extend(ordered_nodes)
        graph.initializer.extend(new_initializers)
        
    return model

def sanitize_and_fix_shapes(model, default_seq_len=128):
    """
    Scans graph inputs/outputs and converts dynamic shape dimensions (dim_param or <=0)
    into static integer shapes, returning the modified model and qai_hub input_specs dict.
    """
    graph = model.graph
    input_specs = {}

    dtype_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.INT8: "int8",
        TensorProto.UINT8: "uint8",
    }

    for input_tensor in graph.input:
        t_type = input_tensor.type.tensor_type
        elem_type = t_type.elem_type
        dtype_str = dtype_map.get(elem_type, "int64")

        shape_dims = []
        if t_type.HasField("shape"):
            for dim in t_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape_dims.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    param = dim.dim_param.lower()
                    if any(k in param for k in ["seq", "token", "length", "pos"]):
                        val = default_seq_len
                    else:
                        val = 1
                    dim.dim_value = val
                    dim.ClearField("dim_param")
                    shape_dims.append(val)
                else:
                    dim.dim_value = 1
                    shape_dims.append(1)
        else:
            shape_dims = [1, default_seq_len]

        input_specs[input_tensor.name] = (tuple(shape_dims), dtype_str)
        print(f" - Fixed input shape '{input_tensor.name}': {shape_dims} ({dtype_str})")

    for output_tensor in graph.output:
        t_type = output_tensor.type.tensor_type
        if t_type.HasField("shape"):
            for dim in t_type.shape.dim:
                if dim.HasField("dim_param"):
                    param = dim.dim_param.lower()
                    dim.dim_value = default_seq_len if any(k in param for k in ["seq", "token", "length"]) else 1
                    dim.ClearField("dim_param")
                elif not dim.HasField("dim_value") or dim.dim_value <= 0:
                    dim.dim_value = 1

    return model, input_specs

def main():
    raw_repo_id = os.environ.get("HF_REPO_ID", "onnx-community/Qwen2.5-Coder-3B-Instruct")
    repo_id = clean_repo_id(raw_repo_id)
    
    model_filename = os.environ.get("HF_MODEL_FILE", "onnx/model_q4.onnx")
    data_filename = os.environ.get("HF_DATA_FILE", "onnx/model_q4.onnx_data")
    hf_token = os.environ.get("HF_TOKEN", None)
    seq_len = int(os.environ.get("MODEL_SEQ_LEN", "128"))

    # 1. Setup staging directory
    staging_dir = "./qwen_q4.onnx"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 2. Download ONNX graph file from Hugging Face
    print(f"Downloading '{model_filename}' from Hugging Face...")
    cached_onnx = hf_hub_download(repo_id=repo_id, filename=model_filename, token=hf_token)
    staged_onnx_path = os.path.join(staging_dir, os.path.basename(model_filename))

    # 3. Download and stage external weights
    old_data_name = ""
    new_data_name = ""
    if data_filename:
        print(f"Downloading '{data_filename}' from Hugging Face...")
        cached_data = hf_hub_download(repo_id=repo_id, filename=data_filename, token=hf_token)
        
        old_data_name = os.path.basename(data_filename)
        if old_data_name.endswith(".onnx_data"):
            new_data_name = old_data_name[:-10] + ".data"
        elif not old_data_name.endswith(".data"):
            new_data_name = old_data_name + ".data"
        else:
            new_data_name = old_data_name

        staged_data_path = os.path.join(staging_dir, new_data_name)
        shutil.copyfile(cached_data, staged_data_path)

    # 4. Load graph structure
    print("Loading ONNX model graph structure...")
    onnx_model = onnx.load(cached_onnx, load_external_data=False)

    # 5. Fix external weights pointers if renamed
    if old_data_name and new_data_name and old_data_name != new_data_name:
        print(f"Updating ONNX external weight pointers from '{old_data_name}' to '{new_data_name}'...")
        for init in onnx_model.graph.initializer:
            for ext in init.external_data:
                if ext.key == "location" and ext.value == old_data_name:
                    ext.value = new_data_name

    # 6. Decompose non-standard nodes IN-PLACE
    onnx_model = decompose_simplified_layer_norm_in_place(onnx_model)

    # 7. Convert dynamic input shapes into static shapes
    print(f"Converting dynamic ONNX shapes to static dimensions (batch_size=1, sequence_length={seq_len})...")
    onnx_model, input_specs = sanitize_and_fix_shapes(onnx_model, default_seq_len=seq_len)

    # 8. Save modified ONNX model to staging directory
    print(f"Saving static-shape ONNX model to '{staged_onnx_path}'...")
    onnx.save(onnx_model, staged_onnx_path)

    # 9. Verify graph topological order and file paths
    print("Verifying graph topological sorting and external data paths...")
    onnx.checker.check_model(staged_onnx_path, full_check=False)

    target_device = get_target_device()

    print(f"\nSubmitting static-shape ONNX model directory '{staging_dir}' to Qualcomm AI Hub for {target_device.name} compilation...")
    
    # Enable --truncate_64bit_io to handle int64 input tensors on Qualcomm Hexagon NPU
    compile_job = hub.submit_compile_job(
        model=staging_dir,
        device=target_device,
        input_specs=input_specs,
        options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io"
    )

    target_model = compile_job.get_target_model()
    output_filename = "qwen_coder_qnn.onnx"
    target_model.download(output_filename)
    
    print(f"\nCompilation successful! Saved artifact to {output_filename}")

if __name__ == "__main__":
    main()
