# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, broad-except
"""Utility to compile Marvell models"""

import os
import sys
import json
import shutil
import tempfile
import base64
import numpy
import tvm
import tvm._ffi


@tvm._ffi.register_func("tvm.mrvl.find_value_in_kv_pair")
def find_value_in_kv_pair(json_input: str, key_to_find: str) -> str:
    """This function takes the graph_json string and key to be searched in
    the json string, using json parser routine it loads the json string
    and access the value using the given key. It raises exception if the
    key is not found in the input json string.

    Parameters
    ----------
    graph_json: String
        This is the graph_json string

    Returns
        value_string: string
        This returns the value string for the given key string.
    """
    value = ""
    try:
        json_dict = json.loads(json_input)
        value = json_dict[key_to_find]
    except KeyError:
        assert False, "Marvell-Compiler-ERROR-Internal:: Could not find matching key in json"

    return value


@tvm._ffi.register_func("tvm.mrvl.GetNodesJSONString")
def get_nodes_json_string(graph_json): #, input_quant_info, quantization_type):
    """This takes the graph_json string from MrvlJSONSerializer and adds / modifies
    the json string to a form suitable for the Marvell Backend.

    Parameters
    ----------
    graph_json: String
        This is the graph_json string from the MrvlJSONSerializer
    input_quant_info: String
        Quantization info to be placed in the json file
    quantization_type: String
        Quantization type { "int8", "fp16" }

    Returns
    -------
    nodes_json_string: string
        This returns the nodes_json string which can be accepted by the Marvell backend.
    """

    dictionary = json.loads(graph_json)
    # dictionary["quantization_type"] = quantization_type  # Only needed for fsim run
    # input_dict = {}
    # if input_quant_info:
    #     input_dict = json.loads(input_quant_info.replace("'", '"'))
    # Add Marvell Index and rename "op" and "name" fields
    mrvl_idx = 1
    num_in = 0
    for iterator in dictionary["nodes"]:
        if iterator["op"] == "kernel":
            iterator["op"] = "tvm_op"
            iterator["attrs"]["mrvl_nodes_idx"] = [mrvl_idx]
            iterator["attrs"]["kernel_const"] = {}
            iterator["attrs"]["bias_const"] = {}
            iterator["attrs"]["beta_const"] = {}
            iterator["attrs"]["gamma_const"] = {}
            iterator["attrs"]["var_const"] = {}
            iterator["attrs"]["mean_const"] = {}
            iterator["attrs"]["input_const"] = {}
            iterator["attrs"]["lut_const"] = {}
            iterator["attrs"]["lut_layout"] = {}
            iterator["name"] = "tvmgen_mrvl_main" + "_" + str(mrvl_idx - 1)
            mrvl_idx = mrvl_idx + 1
        # Fix input layer attributes. Currently NC / NCHW is the layout for all inputs.
        if iterator["op"] == "input":
            iterator["attrs"]["layer_name"] = ["input"]
            iterator["inputs"] = []
            in_id = iterator["name"].split("_i")[-1]
            iterator["input_id"] = [in_id]
            iterator["attrs"]["dtype"] = iterator["attrs"]["dtype"][0]
            iterator["attrs"]["shape"] = iterator["attrs"]["shape"][0]
            # FIXME(shubhaml):
            # Currently, hardcoding the input min/max values to 0 and 1
            # This need to be fixed to populate input node min/max
            input_name = iterator["name"]
            # if input_dict and input_dict[input_name] and input_dict[input_name]["min"]:
            #     iterator["attrs"]["q_min_1"] = [input_dict[input_name]["min"]]
            #     iterator["attrs"]["q_max_1"] = [input_dict[input_name]["max"]]
            # else:
            #     iterator["attrs"]["q_min_1"] = ["0"]
            #     iterator["attrs"]["q_max_1"] = ["1"]
            if len(iterator["attrs"]["shape"][0]) == 2:
                iterator["attrs"]["data_layout"] = ["NC"]
            else:
                iterator["attrs"]["data_layout"] = ["NCHW"]
            # Infer Batch Size from the input shape
            batch_size = iterator["attrs"]["shape"][0][0]
            dictionary["batch_size"] = f"{batch_size}"
            num_in = num_in + 1

    # Create a new inputs to store only the previous node input and not the const inputs
    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            list_prev = []
            for prev in iterator["inputs"]:
                if dictionary["nodes"][prev[0]]["op"] == "tvm_op":
                    mrvl_idx_prev = dictionary["nodes"][prev[0]]["attrs"]["mrvl_nodes_idx"][0]
                    list_prev.append([mrvl_idx_prev + num_in - 1, 0, 0])
                if dictionary["nodes"][prev[0]]["op"] == "input":
                    # Handle multiple input case
                    idx_in = int(dictionary["nodes"][prev[0]]["input_id"][0])
                    list_prev.append([idx_in, 0, 0])
            iterator["node_prev"] = list_prev

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            del iterator["inputs"]

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            iterator["inputs"] = iterator["node_prev"]

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            del iterator["node_prev"]

    # Remove unneeded fields
    del dictionary["node_row_ptr"]

    # Patch up arg_nodes and heads to remove references to constant inputs
    list_nodes = dictionary["arg_nodes"]
    list_nodes_updated = []

    for iterator in list_nodes:
        if dictionary["nodes"][iterator]["op"] != "const":
            if dictionary["nodes"][iterator]["op"] == "input":
                input_name = dictionary["nodes"][iterator]["name"]
                input_num_str = input_name.split("_i", 1)[1]
                input_num = int(input_num_str)
                list_nodes_updated.append(input_num)
            else:
                list_nodes_updated.append(
                    dictionary["nodes"][iterator]["attrs"]["mrvl_nodes_idx"][0]
                )
    dictionary["arg_nodes"] = list_nodes_updated

    # Add additional data required by the runtime such as number of inputs
    # and number of outputs to the subgraph
    num_subgraph_inputs = str(len(list_nodes_updated))
    dictionary["num_subgraph_inputs"] = f"{num_subgraph_inputs}"
    list_heads = dictionary["heads"]
    list_heads_updated = []
    for iterator in list_heads:
        if dictionary["nodes"][iterator[0]]["op"] != "const":
            if iterator[0] != 0:
                get_index = dictionary["nodes"][iterator[0]]["attrs"]["mrvl_nodes_idx"][0]
                new_index = get_index + num_in - 1
                list_heads_updated.append([new_index, 0, 0])
    dictionary["heads"] = list_heads_updated

    num_subgraph_outputs = str(len(list_heads_updated))
    dictionary["num_subgraph_outputs"] = f"{num_subgraph_outputs}"

    # Delete the constant nodes, these are not required for the constants file
    dictionary["nodes"] = [
        feature for feature in dictionary["nodes"] if "const" not in feature["op"]
    ]

    # Remove un-needed array nesting
    for iterator in dictionary["nodes"]:
        if iterator["op"] not in "input":
            for it2 in iterator["attrs"]:
                if it2 not in [
                    "num_inputs",
                    "num_outputs",
                    "mrvl_nodes_idx",
                    "mean_const",
                    "var_const",
                    "beta_const",
                    "kernel_const",
                    "bias_const",
                    "gamma_const",
                    "input_const",
                    "lut_const",
                    "lut_layout",
                ]:
                    iterator["attrs"][it2] = iterator["attrs"][it2][0]

    # Now create the dltype and dlshape attributes
    dltype = ["list_str"]
    shape = ["list_shape"]
    list_types = []
    list_shapes = []
    for iterator in dictionary["nodes"]:
        list_types.extend(iterator["attrs"]["dtype"])
        list_shapes.extend(iterator["attrs"]["shape"])
    dltype.append(list_types)
    shape.append(list_shapes)
    dict_shape_type = {}
    dict_shape_type["shape"] = shape
    dict_shape_type["dltype"] = dltype
    dictionary["attrs"] = dict_shape_type

    nodes_json_string = json.dumps(dictionary)
    return nodes_json_string


def change_dw_conv_to_normal_conv(attrs, const_entry):
    """The function checks if the convolution is depthwise and changes it to normal
    convolution by stuffing zeros in the kernel coefficients.

    Parameters
    ----------
    attrs: dictionary
        layer attributes
    const_entry: json entry containing shape, data_base64
    """

    if attrs["layer_name"][0] != "Conv2D":
        return

    group = int(attrs["groups"][0])
    if group == 1:
        return

    shape = const_entry["shape"]
    filter_n = shape[0]
    filter_h = shape[1]
    filter_w = shape[2]
    filter_c = shape[3]

    # supporting only case where group == n and c == 1
    if group != filter_n or filter_c != 1:
        return

    # Determine dtype and decode kernel data
    dtype = const_entry.get("dtype", "float32")
    kernel_data = base64.b64decode(const_entry["data_base64"])
    if dtype in ("int8", "uint8"):
        kernel = numpy.frombuffer(kernel_data, dtype=numpy.int8 if dtype == "int8" else numpy.uint8)
    else:
        kernel = numpy.frombuffer(kernel_data, dtype=numpy.float32)
    kernel = kernel.reshape((filter_n, filter_h, filter_w, filter_c))

    # Create output kernel with expanded last dimension
    out_kernel = numpy.zeros((filter_n, filter_h, filter_w, filter_n), dtype=kernel.dtype)
    for n_idx in range(filter_n):
        for h_idx in range(filter_h):
            for w_idx in range(filter_w):
                out_kernel[n_idx, h_idx, w_idx, n_idx] = kernel[n_idx, h_idx, w_idx, 0]

    # Encode output kernel back to base64
    out_kernel_b64 = base64.b64encode(out_kernel.tobytes()).decode("utf-8")
    const_entry["data_base64"] = out_kernel_b64
    const_entry["shape"] = [filter_n, filter_h, filter_w, filter_n]
    attrs["groups"][0] = "1"
    return


@tvm._ffi.register_func("tvm.mrvl.ModifyConstNames")
def modify_const_names(nodes_json_str, consts_json_str):
    """This takes the graph module returned by relay.build an generates nodes and constant
       meta data suitable for compilation by the back end.

    Parameters
    ----------
    nodes_json_str: string
        The nodes json string suitable for the Marvell backend.

    consts_json_str: string
        The consts_json_string generated by the backend compiler.

    Returns
    -------
    modified_nodes_consts: string
        This returns a concatenated string of the nodes_json and modified
        consts json file, seperated by a delimiter |. The modification to the
        consts file is necessary since we have added the Merge Compiler Pass
        which names the constants in a form unsuitable for the backend.
    """

    nodes = json.loads(nodes_json_str)
    const = json.loads(consts_json_str)
    modif_const = {}

    for node in nodes["nodes"]:
        has_bias = False
        has_batchnorm = False
        for attrs in node["attrs"]:
            if attrs == "bias_const_name":
                has_bias = True
            if attrs == "gamma_const_name":
                has_batchnorm = True

        for attrs in node["attrs"]:
            if attrs == "kernel_const_name":
                old_name = node["attrs"][attrs][0]
                new_name = node["name"] + "_const_0"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                change_dw_conv_to_normal_conv(node["attrs"], modif_const[new_name])
                map_kernel = {}
                map_kernel["shape"] = modif_const[new_name]["shape"]
                map_kernel["dtype"] = modif_const[new_name]["dtype"]
                map_kernel["min"] = modif_const[new_name]["min"]
                map_kernel["max"] = modif_const[new_name]["max"]
                map_kernel["name"] = new_name
                node["attrs"]["kernel_const"] = map_kernel
            if attrs == "bias_const_name":
                old_name = node["attrs"][attrs][0]
                new_name = node["name"] + "_const_1"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                bias_map = {}
                bias_map["shape"] = modif_const[new_name]["shape"]
                bias_map["dtype"] = modif_const[new_name]["dtype"]
                bias_map["min"] = modif_const[new_name]["min"]
                bias_map["max"] = modif_const[new_name]["max"]
                bias_map["name"] = new_name
                node["attrs"]["bias_const"] = bias_map
            if attrs == "gamma_const_name":
                old_name = node["attrs"][attrs][0]
                if has_bias:
                    new_name = node["name"] + "_const_2"
                else:
                    new_name = node["name"] + "_const_1"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                gamma_map = {}
                gamma_map["shape"] = modif_const[new_name]["shape"]
                gamma_map["dtype"] = modif_const[new_name]["dtype"]
                gamma_map["name"] = new_name
                node["attrs"]["gamma_const"] = gamma_map
            if attrs == "beta_const_name":
                old_name = node["attrs"][attrs][0]
                if has_bias:
                    new_name = node["name"] + "_const_3"
                else:
                    new_name = node["name"] + "_const_2"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                beta_map = {}
                beta_map["shape"] = modif_const[new_name]["shape"]
                beta_map["dtype"] = modif_const[new_name]["dtype"]
                beta_map["name"] = new_name
                node["attrs"]["beta_const"] = beta_map
            if attrs == "mean_const_name":
                old_name = node["attrs"][attrs][0]
                if has_bias:
                    new_name = node["name"] + "_const_4"
                else:
                    new_name = node["name"] + "_const_3"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                mean_map = {}
                mean_map["shape"] = modif_const[new_name]["shape"]
                mean_map["dtype"] = modif_const[new_name]["dtype"]
                mean_map["name"] = new_name
                node["attrs"]["mean_const"] = mean_map
            if attrs == "var_const_name":
                old_name = node["attrs"][attrs][0]
                if has_bias:
                    new_name = node["name"] + "_const_5"
                else:
                    new_name = node["name"] + "_const_4"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                var_map = {}
                var_map["shape"] = modif_const[new_name]["shape"]
                var_map["dtype"] = modif_const[new_name]["dtype"]
                var_map["name"] = new_name
                node["attrs"]["var_const"] = var_map
            if attrs == "input_const_name":
                old_name = node["attrs"][attrs][0].split("-")[-1]
                new_name = node["name"] + "_const_0"
                modif_const[new_name] = const[old_name]
                modif_const[new_name]["shape"] = list(map(int, node["attrs"]["input_const_shape"]))
                node["attrs"][attrs][0] = new_name
                map_const = {}
                map_const["shape"] = modif_const[new_name]["shape"]
                map_const["dtype"] = modif_const[new_name]["dtype"]
                map_const["min"] = modif_const[new_name]["min"]
                map_const["max"] = modif_const[new_name]["max"]
                map_const["name"] = new_name
                node["attrs"]["input_const"] = map_const
            if attrs == "lut_const_name":
                old_name = node["attrs"][attrs][0]
                if has_batchnorm:
                    new_name = node["name"] + "_const_6"
                else:
                    new_name = node["name"] + "_const_2"
                modif_const[new_name] = const[old_name]
                node["attrs"][attrs][0] = new_name
                lut_map = {}
                lut_map["shape"] = modif_const[new_name]["shape"]
                lut_map["dtype"] = modif_const[new_name]["dtype"]
                lut_map["name"] = new_name
                node["attrs"]["lut_const"] = lut_map
                node["attrs"]["lut_layout"] = ["NC"]

    nodes_mod_str = json.dumps(nodes, indent=2)
    const_mod_str = json.dumps(modif_const, indent=2)
    return nodes_mod_str + "|" + const_mod_str


def get_working_dir():
    """Obtain the current working directory from where tvm is invoked"""
    return os.getcwd()


def delete_temp_files(symbol_name):
    nodes_json_file = f"{symbol_name}-nodes.json"
    consts_json_file = f"{symbol_name}-consts.json"
    bin_folder = os.path.join(get_working_dir(), "bin_" + symbol_name)
    os.system("rm -rf" + " " + nodes_json_file + " " + consts_json_file)
    if "MRVL_SAVE_MODEL_BIN" not in os.environ:
        os.system("rm -rf" + " " + bin_folder)


@tvm._ffi.register_func("tvm.mrvl.WriteJsonFile")
def write_json_file(json_string, json_filename):
    """Generate json file under working directory"""
    working_dir = get_working_dir()
    json_file = os.path.join(working_dir, json_filename)
    with open(json_file, "w") as out_file:
        out_file.write(json_string)
    return json_file


@tvm._ffi.register_func("tvm.mrvl.CompileModel")
def compile_model(
    symbol_name,
    #model_name,
    nodes_json_string,
    consts_json_string,
    #tools_bin_path,
    compiler_opts,
    clean_temp_files=True,
):
    """Compile the model using Marvell Backend compiler and return the generated binary"""
    # generate pair of json files
    nodes_json_file = write_json_file(nodes_json_string, f"{symbol_name}-nodes.json")
    consts_json_file = write_json_file(consts_json_string, f"{symbol_name}-consts.json")
    mrvl_exec = "/home/princek/new_tvm/tvm/build/marvell-18.04-mlip/bin/mrvl-tmlc"
    #mrvl_exec = os.path.join(tools_bin_path, bin_name)
    exec_on_path = shutil.which(mrvl_exec)
    if exec_on_path is None:
        error_msg = (
            "Marvell Compiler not found! Please specify the path to Marvell tools "
            "by using the mrvl-bin_dir option or add it to $PATH."
        )
        raise RuntimeError(error_msg)

    # Parse the nodes_json string for the batch size
    dictionary = json.loads(nodes_json_string)
    batch_size = dictionary["batch_size"]

    # Check Invalid batch_size based on the Mrvl Flow
    if "node_splitter" in compiler_opts and int(batch_size) > 1:
        print("mrvl-tmlc (node_splitter) doesn't support batch_size > 1")
        sys.exit(0)
    elif int(batch_size) > 8:
        print("mrvl-tmlc doesn't support batch_size > 8")
        sys.exit(0)

    # Invoke Marvell Backend with appropriate options
    compile_cmd = (
        mrvl_exec
        + " -mn "
        + symbol_name
        + " -f1 "
        + nodes_json_file
        + " -f2 "
        + consts_json_file
        + " "
        + compiler_opts
        + " -b "
        + batch_size
    )
    ret_val = os.system(compile_cmd)
    if ret_val == 0:
        working_dir = get_working_dir()
        # Read generated binary and encode in base64 format
        bin_file = os.path.join(working_dir, "bin_" + symbol_name, symbol_name + ".bin")
        with open(bin_file, "rb") as f:
            data = bytearray(f.read())
            base64_bytes = base64.b64encode(data)
            if not data:
                raise RuntimeError("Compilation ERROR: empty result is generated")
            # Cleanup Temporary Files
            if clean_temp_files:
                delete_temp_files(symbol_name)
            return base64_bytes
    else:
        error_msg = "Compilation ERROR: Error compiling Marvell region!"
        raise RuntimeError(error_msg)
