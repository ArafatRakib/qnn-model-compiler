import os
import sys
import shutil
import gc
import onnx
from onnx import helper, TensorProto
import qai_hub as hub
from huggingface_hub import hf_hub_download

def clean_repo_id(repo_input):
    repo = repo_input.strip()
    for prefix in ["https://huggingface.co/", "http://huggingface.co/"]:
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
    return repo.strip("/")

def get_target_device():
    print("Fetching device catalog from Qualcomm AI Hub...")
    all_devices = hub.get_devices()
    for dev in all_devices:
        dev_str = f"{dev.name} {dev.attributes}".lower()
        if any(k in dev_str for k in ["s25", "8 elite", "8elite", "sm8750", "sm8735"]):
            print(f"Selected target device: '{dev.name}'")
            return dev
    return all_devices[0]

def decompose_layer_norm(model):
    """Replaces custom Microsoft LayerNorm with standard ONNX math ops in-place."""
    graph = model.graph
    ordered_nodes = []
    new_initializers = []
    counter = 0

    for node in graph.node:
        if node.op_type in ["SimplifiedLayerNormalization", "RMSNorm"]:
            x_input, w_input, y_output = node.input[0], node.input[1], node.output[0]
            epsilon, axis = 1e-5, -1
            for attr in node.attribute:
                if attr.name == "epsilon": epsilon = attr.f
                elif attr.name == "axis": axis = attr.i
            
            prefix = f"sln_decomp_{counter}_"
            counter += 1
            eps_name = prefix + "eps"
            new_initializers.append(helper.make_tensor(eps_name, TensorProto.FLOAT16, [], [epsilon]))
            
            x_sq, mean_sq, mean_eps, rms, norm = [f"{prefix}{s}" for s in ["x_sq", "mean_sq", "mean_eps", "rms", "norm"]]
            ordered_nodes.extend([
                helper.make_node("Mul", [x_input, x_input], [x_sq]),
                helper.make_node("ReduceMean", [x_sq], [mean_sq], axes=[axis], keepdims=1),
                helper.make_node("Add", [mean_sq, eps_name], [mean_eps]),
                helper.make_node("Sqrt", [mean_eps], [rms]),
                helper.make_node("Div", [x_input, rms], [norm]),
                helper.make_node("Mul", [norm, w_input], [y_output])
            ])
        else:
            ordered_nodes.append(node)

    if counter > 0:
        del graph.node[:]
        graph.node.extend(ordered_nodes)
        graph.initializer.extend(new_initializers)
    return model

def sanitize_shapes(model, default_seq_len=128):
    """Converts dynamic ONNX shapes to static dimensions for Hexagon NPU."""
    graph = model.graph
    input_specs = {}
    dtype_map = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16", TensorProto.INT64: "int64", TensorProto.INT32: "int32"}

    for input_tensor in graph.input:
        t_type = input_tensor.type.tensor_type
        dtype_str = dtype_map.get(t_type.elem_type, "int64")
        shape_dims = []
        if t_type.HasField("shape"):
            for dim in t_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape_dims.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    param = dim.dim_param.lower()
                    val = default_seq_len if any(k in param for k in ["seq", "token", "length", "pos"]) else 1
                    dim.dim_value = val
                    dim.ClearField("dim_param")
                    shape_dims.append(val)
                else:
                    dim.dim_value = 1
                    shape_dims.append(1)
        else:
            shape_dims = [1, default_seq_len]

        input_specs[input_tensor.name] = (tuple(shape_dims), dtype_str)

    return model, input_specs

def main():
    repo_id = clean_repo_id(os.environ.get("HF_REPO_ID", "onnx-community/Qwen2.5-Coder-3B-Instruct"))
    model_filename = os.environ.get("HF_MODEL_FILE", "onnx/model_int8.onnx")
    data_filename = os.environ.get("HF_DATA_FILE", "onnx/model_int8.onnx_data")
    hf_token = os.environ.get("HF_TOKEN", None)
    seq_len = int(os.environ.get("MODEL_SEQ_LEN", "128"))

    staging_dir = "./qwen_model.onnx"
    if os.path.exists(staging_dir): shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    print(f"Downloading '{model_filename}'...")
    cached_onnx = hf_hub_download(repo_id=repo_id, filename=model_filename, token=hf_token)
    staged_onnx_path = os.path.join(staging_dir, os.path.basename(model_filename))

    old_data_name, new_data_name = "", ""
    if data_filename:
        print(f"Downloading '{data_filename}'...")
        cached_data = hf_hub_download(repo_id=repo_id, filename=data_filename, token=hf_token)
        old_data_name = os.path.basename(data_filename)
        new_data_name = old_data_name[:-10] + ".data" if old_data_name.endswith(".onnx_data") else old_data_name
        shutil.move(cached_data, os.path.join(staging_dir, new_data_name))

    onnx_model = onnx.load(cached_onnx, load_external_data=False)

    if old_data_name and new_data_name and old_data_name != new_data_name:
        for init in onnx_model.graph.initializer:
            for ext in init.external_data:
                if ext.key == "location" and ext.value == old_data_name:
                    ext.value = new_data_name

    onnx_model = decompose_layer_norm(onnx_model)
    onnx_model, input_specs = sanitize_shapes(onnx_model, default_seq_len=seq_len)

    onnx.save(onnx_model, staged_onnx_path)
    del onnx_model
    gc.collect()

    target_device = get_target_device()
    print(f"\nSubmitting INT8 job to Qualcomm AI Hub for {target_device.name}...")

    compile_job = hub.submit_compile_job(
        model=staging_dir,
        device=target_device,
        input_specs=input_specs,
        options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io"
    )

    target_model = compile_job.get_target_model()
    target_model.download("qwen_coder_qnn.onnx")
    print("Compilation successful! Saved artifact to qwen_coder_qnn.onnx")

if __name__ == "__main__":
    main()
