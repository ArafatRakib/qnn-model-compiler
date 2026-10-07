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
    
    # Galaxy S25 / Snapdragon 8 Elite target matches Snapdragon 8s Gen 4 NPU architecture (SM8750 / SM8735)
    for dev in all_devices:
        dev_str = f"{dev.name} {dev.attributes}".lower()
        if any(k in dev_str for k in ["s25", "8 elite", "8elite", "sm8750", "sm8735"]):
            print(f"Selected Qualcomm AI Hub cloud target: '{dev.name}' (Architecture match for Snapdragon 8s Gen 4)")
            return dev

    for dev in all_devices:
        if "snapdragon" in dev.name.lower() or "s24" in dev.name.lower():
            print(f"Fallback target device: '{dev.name}'")
            return dev

    return all_devices[0]

def get_tensor_elem_type(graph, tensor_name):
    """Find data element type of a tensor from initializers, value_info, inputs, or outputs."""
    for init in graph.initializer:
        if init.name == tensor_name:
            return init.data_type
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if info.name == tensor_name and info.type.HasField("tensor_type"):
            return info.type.tensor_type.elem_type
    return TensorProto.FLOAT

def get_tensor_shape(graph, tensor_name):
    """Extract dimension shape list for a given tensor."""
    for init in graph.initializer:
        if init.name == tensor_name:
            return [d for d in init.dims]
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if info.name == tensor_name and info.type.HasField("tensor_type"):
            shape = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(128)
            return shape
    return []

def decompose_custom_nodes(model):
    """
    Decomposes com.microsoft:SimplifiedLayerNormalization, com.microsoft:RMSNorm,
    and com.microsoft:RotaryEmbedding into standard ONNX operators in-place.
    """
    graph = model.graph
    ordered_nodes = []
    new_initializers = []
    norm_counter = 0
    rope_counter = 0

    for node in graph.node:
        # 1. Decompose Custom LayerNorm / RMSNorm
        if node.op_type in ["SimplifiedLayerNormalization", "RMSNorm"]:
            x_input, w_input, y_output = node.input[0], node.input[1], node.output[0]
            epsilon, axis = 1e-5, -1
            for attr in node.attribute:
                if attr.name == "epsilon": 
                    epsilon = attr.f
                elif attr.name == "axis": 
                    axis = attr.i
            
            elem_type = get_tensor_elem_type(graph, w_input)
            prefix = f"norm_decomp_{norm_counter}_"
            norm_counter += 1
            eps_name = prefix + "eps"
            
            eps_tensor = helper.make_tensor(eps_name, elem_type, [], [epsilon])
            new_initializers.append(eps_tensor)
            
            x_sq, mean_sq, mean_eps, rms, norm = [f"{prefix}{s}" for s in ["x_sq", "mean_sq", "mean_eps", "rms", "norm"]]
            ordered_nodes.extend([
                helper.make_node("Mul", [x_input, x_input], [x_sq], name=prefix+"mul_sq"),
                helper.make_node("ReduceMean", [x_sq], [mean_sq], axes=[axis], keepdims=1, name=prefix+"red_mean"),
                helper.make_node("Add", [mean_sq, eps_name], [mean_eps], name=prefix+"add_eps"),
                helper.make_node("Sqrt", [mean_eps], [rms], name=prefix+"sqrt"),
                helper.make_node("Div", [x_input, rms], [norm], name=prefix+"div"),
                helper.make_node("Mul", [norm, w_input], [y_output], name=prefix+"mul_w")
            ])

        # 2. Decompose Custom RotaryEmbedding (RoPE)
        elif node.op_type == "RotaryEmbedding":
            x_input = node.input[0]
            y_output = node.output[0]
            prefix = f"rope_decomp_{rope_counter}_"
            rope_counter += 1
            
            rope_nodes = []
            
            if len(node.input) >= 4:
                pos_ids = node.input[1]
                cos_cache = node.input[2]
                sin_cache = node.input[3]
                
                cos_gathered = prefix + "cos_g"
                sin_gathered = prefix + "sin_g"
                
                node_gather_cos = helper.make_node("Gather", inputs=[cos_cache, pos_ids], outputs=[cos_gathered], axis=0, name=prefix+"gather_cos")
                node_gather_sin = helper.make_node("Gather", inputs=[sin_cache, pos_ids], outputs=[sin_gathered], axis=0, name=prefix+"gather_sin")
                
                x_shape = get_tensor_shape(graph, x_input)
                unsq_axis = 1 if len(x_shape) == 4 and x_shape[1] < x_shape[2] else 2
                
                axes_unsq_name = prefix + "unsq_axes"
                new_initializers.append(helper.make_tensor(axes_unsq_name, TensorProto.INT64, [1], [unsq_axis]))
                
                cos_unsq = prefix + "cos_unsq"
                sin_unsq = prefix + "sin_unsq"
                node_unsq_cos = helper.make_node("Unsqueeze", inputs=[cos_gathered, axes_unsq_name], outputs=[cos_unsq], name=prefix+"unsq_cos")
                node_unsq_sin = helper.make_node("Unsqueeze", inputs=[sin_gathered, axes_unsq_name], outputs=[sin_unsq], name=prefix+"unsq_sin")
                
                rope_nodes.extend([node_gather_cos, node_gather_sin, node_unsq_cos, node_unsq_sin])
                
                cos_raw = cos_unsq
                sin_raw = sin_unsq
                cos_cache_shape = get_tensor_shape(graph, cos_cache)
                cos_last_dim = cos_cache_shape[-1] if cos_cache_shape else 64
            elif len(node.input) == 3:
                cos_raw = node.input[1]
                sin_raw = node.input[2]
                cos_raw_shape = get_tensor_shape(graph, cos_raw)
                cos_last_dim = cos_raw_shape[-1] if cos_raw_shape else 64
            else:
                ordered_nodes.append(node)
                continue

            x_shape = get_tensor_shape(graph, x_input)
            head_dim = x_shape[-1] if len(x_shape) > 0 and isinstance(x_shape[-1], int) and x_shape[-1] > 0 else 128
            half_dim = head_dim // 2

            # Expand half-dimension cos/sin (64) to full head_dim (128) if needed
            if cos_last_dim < head_dim:
                cos_input = prefix + "cos_full"
                sin_input = prefix + "sin_full"
                node_cat_cos = helper.make_node("Concat", inputs=[cos_raw, cos_raw], outputs=[cos_input], axis=-1, name=prefix+"cat_cos")
                node_cat_sin = helper.make_node("Concat", inputs=[sin_raw, sin_raw], outputs=[sin_input], axis=-1, name=prefix+"cat_sin")
                rope_nodes.extend([node_cat_cos, node_cat_sin])
            else:
                cos_input = cos_raw
                sin_input = sin_raw

            # Slicing initializers
            init_s0 = prefix + "s0"
            init_e_half = prefix + "e_half"
            init_s_half = prefix + "s_half"
            init_e_max = prefix + "e_max"
            init_axes = prefix + "axes"

            new_initializers.extend([
                helper.make_tensor(init_s0, TensorProto.INT64, [1], [0]),
                helper.make_tensor(init_e_half, TensorProto.INT64, [1], [half_dim]),
                helper.make_tensor(init_s_half, TensorProto.INT64, [1], [half_dim]),
                helper.make_tensor(init_e_max, TensorProto.INT64, [1], [2147483647]),
                helper.make_tensor(init_axes, TensorProto.INT64, [1], [-1]),
            ])

            x1, x2 = prefix + "x1", prefix + "x2"
            neg_x2, x_rot = prefix + "neg_x2", prefix + "x_rot"
            x_cos, x_rot_sin = prefix + "x_cos", prefix + "x_rot_sin"

            # Construct RoPE math: Y = (X * cos) + (Concat(-X2, X1) * sin)
            node_slice1 = helper.make_node("Slice", inputs=[x_input, init_s0, init_e_half, init_axes], outputs=[x1], name=prefix+"slice1")
            node_slice2 = helper.make_node("Slice", inputs=[x_input, init_s_half, init_e_max, init_axes], outputs=[x2], name=prefix+"slice2")
            node_neg = helper.make_node("Neg", inputs=[x2], outputs=[neg_x2], name=prefix+"neg")
            node_concat = helper.make_node("Concat", inputs=[neg_x2, x1], outputs=[x_rot], axis=-1, name=prefix+"concat")
            node_mul_cos = helper.make_node("Mul", inputs=[x_input, cos_input], outputs=[x_cos], name=prefix+"mul_cos")
            node_mul_sin = helper.make_node("Mul", inputs=[x_rot, sin_input], outputs=[x_rot_sin], name=prefix+"mul_sin")
            node_add = helper.make_node("Add", inputs=[x_cos, x_rot_sin], outputs=[y_output], name=prefix+"add")

            rope_nodes.extend([node_slice1, node_slice2, node_neg, node_concat, node_mul_cos, node_mul_sin, node_add])
            ordered_nodes.extend(rope_nodes)
        else:
            ordered_nodes.append(node)

    if norm_counter > 0 or rope_counter > 0:
        print(f"Decomposed {norm_counter} norm nodes and {rope_counter} RoPE nodes into standard ONNX operators...")
        del graph.node[:]
        graph.node.extend(ordered_nodes)
        graph.initializer.extend(new_initializers)
        
    return model

def sanitize_shapes(model, default_seq_len=128):
    """Converts dynamic ONNX shapes to static dimensions for Hexagon NPU."""
    graph = model.graph
    input_specs = {}
    dtype_map = {
        TensorProto.FLOAT: "float32", 
        TensorProto.FLOAT16: "float16", 
        TensorProto.INT64: "int64", 
        TensorProto.INT32: "int32"
    }

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
    if os.path.exists(staging_dir): 
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    print(f"Downloading '{model_filename}'...")
    cached_onnx = hf_hub_download(repo_id=repo_id, filename=model_filename, token=hf_token)
    staged_onnx_path = os.path.join(staging_dir, os.path.basename(model_filename))

    old_data_name, new_data_name = "", ""
    if data_filename:
        print(f"Downloading '{data_filename}'...")
        cached_data = hf_hub_download(repo_id=repo_id, filename=data_filename, token=hf_token)
        real_cached_data = os.path.realpath(cached_data)
        
        old_data_name = os.path.basename(data_filename)
        new_data_name = old_data_name[:-10] + ".data" if old_data_name.endswith(".onnx_data") else old_data_name
        
        staged_data_path = os.path.join(staging_dir, new_data_name)
        print(f"Copying real binary file to '{staged_data_path}'...")
        shutil.copyfile(real_cached_data, staged_data_path)

    onnx_model = onnx.load(cached_onnx, load_external_data=False)

    if old_data_name and new_data_name and old_data_name != new_data_name:
        for init in onnx_model.graph.initializer:
            for ext in init.external_data:
                if ext.key == "location" and ext.value == old_data_name:
                    ext.value = new_data_name

    # First shape inference pass to populate value_info shapes
    print("Running initial shape inference pass...")
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        print(f"Notice during initial shape inference: {e}")

    onnx_model = decompose_custom_nodes(onnx_model)
    onnx_model, input_specs = sanitize_shapes(onnx_model, default_seq_len=seq_len)

    # Final shape inference validation pass
    print("Running final shape and type inference validation...")
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        print("Final shape inference validation passed successfully!")
    except Exception as e:
        print(f"Notice during final shape inference: {e}")

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
