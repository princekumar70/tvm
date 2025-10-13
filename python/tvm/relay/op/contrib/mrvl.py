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
"""Marvell Library supported operators."""

import os
import numpy as np
import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import Call, Tuple, TupleGetItem
from tvm.relay.quantize.contrib.mrvl import MrvlQuantize
from tvm.contrib import mrvl as mrvl_contrib

from tvm.relay.transform import _ffi_api
from ...dataflow_pattern import (
    wildcard,
    is_op,
    is_constant,
    is_tuple,
    is_tuple_get_item,
    is_var,
    rewrite,
    DFPatternCallback,
)
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d

do_legalization = False


def partition_for_mrvl(
    mod,
    params=None,
    mod_name: str = "default",
    **kwargs,
):
    """Partition the graph greedily into Marvell graph region(s) and a LLVM region(s). The LLVM
    region will contain ops not supported by the Marvell backend.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    mod_name : module_name

    Returns
    -------
    mod_mrvl_llvm_regions : annotated & partitioned module (of Mrvl region(s) & LLVM region)
    """

    global do_legalization
    do_legalization = any(
        keyword in kwargs.get("mattr", "") for keyword in ("mlip2", "cn20ka", "cnf20ka")
    )

    # setup & register convert layout options
    convert_layout_dict = {
        "qnn.avg_pool2d": ["NHWC"],
        "nn.conv2d": ["NHWC", "OHWI"],
        "nn.upsampling": ["NHWC"],
        "qnn.conv2d": ["NHWC", "OHWI"],
        "nn.max_pool2d": ["NHWC"],
        "nn.avg_pool2d": ["NHWC"],
        "nn.global_avg_pool2d": ["NHWC"],
        "nn.global_max_pool2d": ["NHWC"],
    }
    # convert layout back to NCHW for ops in main
    desired_layouts_in_main = {
        "nn.conv2d": ["NCHW", "OIHW"],
        "qnn.conv2d": ["NCHW", "OIHW"],
        "nn.conv2d_transpose": ["NCHW", "OIHW"],
        "nn.max_pool2d": ["NCHW"],
        "nn.avg_pool2d": ["NCHW"],
        "nn.global_avg_pool2d": ["NCHW"],
    }
    mrvl_register_conv2d_attr_funcs_for_convert_layout()
    mrvl_register_max_pool2d_attr_funcs_for_convert_layout()
    mrvl_register_avg_pool2d_attr_funcs_for_convert_layout()
    mrvl_register_global_avg_pool2d_attr_funcs_for_convert_layout()
    mrvl_register_upsampling_attr_funcs_for_convert_layout()
    _register_external_op_helper("qnn.quantize")
    _register_external_op_helper("qnn.dequantize")
    _register_external_op_helper("qnn.requantize")

    convert_layout_dict["nn.conv2d_transpose"] = ["NHWC", "OHWI"]
    mrvl_register_conv2d_transpose_attr_funcs_for_convert_layout()

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    opt_level = 3
    disabled_pass_list = ["AlterOpLayout"]
    annotate_target_str = "mrvl"
    annotate_target_include_non_call_ops = True

    # Disable MrvlTopKArgmaxOutputCastPass, conversion is done in runtime
    skip_topk_argmax_output_cast = True

    seq_tvmc_pre_repartition = tvm.transform.Sequential(
        passes=[
            relay.transform.InferType(),
            relay.transform.FakeQuantizationToInteger(),
            legalize_transform(do_legalization),
            MrvlRemoveDropoutPass(),
            MrvlRemoveCopyPass(),
            relay.transform.RemoveUnusedFunctions(),
            MrvlOptimizeBatchnormPass(),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
        ]
    )

    seq_common_stage_2 = tvm.transform.Sequential(
        passes=[
            relay.transform.InferType(),
            relay.transform.ConvertLayout(convert_layout_dict),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
            _ffi_api.SimplifyTransposeSimulatedQuantize(),
            _ffi_api.SimplifyConsecutiveSimulatedQuantize(),
            SimplifyQnnLayoutTransform(),
            MrvlTopKArgmaxOutputCastPass(skip_topk_argmax_output_cast),
            MrvlMovePadAfterLayoutTransformPass(),
            relay.transform.MergeComposite(mrvl_pattern_table()),
            relay.transform.AnnotateTarget(
                annotate_target_str,
                annotate_target_include_non_call_ops,
            ),
        ]
    )
    seq_tvmc_specifics = tvm.transform.Sequential(
        passes=[
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(""),
            relay.transform.InferType(),
            relay.transform.DynamicToStatic(),
        ]
    )
    seq_tvmc_post_repartition = tvm.transform.Sequential(
        passes=[
            # Convert Layout of conv ops in main to NCHW (as expected by LLVM).
            # This pass does not change layout of ops already partitioned into
            # Marvell regions.
            relay.transform.ConvertLayout(desired_layouts_in_main),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass_list):
        tmp_mod1 = seq_tvmc_pre_repartition(mod)
        add_instrumentation = False
        data_path = ""
        calib_mode = "max_scale"
        calib_chunk_by = -1
        if "calibration_data" in kwargs:
            data_path = kwargs.get("calibration_data")
            add_instrumentation = True
        if "calibrate_mode" in kwargs:
            calib_mode = kwargs.get("calibrate_mode")
        if "calibrate_chunk_by" in kwargs:
            calib_chunk_by = kwargs.get("calibrate_chunk_by")
        if add_instrumentation:
            data_dict = np.load(data_path, allow_pickle=True)
            tmp_mod1 = instrument_ir(tmp_mod1, data_dict, calib_mode, calib_chunk_by, debug=False)
        tmp_mod1 = seq_common_stage_2(tmp_mod1)
        tmp_mod1 = seq_tvmc_specifics(tmp_mod1)
        tmp_mod1 = repartition_mrvl_subgraphs(tmp_mod1)
        tmp_mod1 = seq_tvmc_post_repartition(tmp_mod1)

        mod_mrvl_llvm_regions = add_attributes(
            tmp_mod1, annotate_target_str, add_instrumentation, **kwargs
        )
    return mod_mrvl_llvm_regions


mrvl_activations = [
    "nn.relu",
    "tanh",
    "sigmoid",
]


def is_activation(pattern, quantizable=True):
    """Checks the given pattern is part of Marvell activations"""
    for ptrn in mrvl_activations:
        pattern = pattern.optional(is_op(ptrn))
        if quantizable:
            pattern = pattern.optional(
                lambda x: (
                    is_op("relay.op.annotation.simulated_quantize")(
                        x, is_constant(), is_constant(), is_constant()
                    )
                )
            )
    return pattern


def legalize_transform(legalize_flag):
    @tvm.transform.module_pass(opt_level=0)
    def legalize_transform_inner(mod, ctx):
        if legalize_flag:
            # do_legalization is True only for MLIP2
            return relay.qnn.transform.Legalize()(mod)
        return mod

    return legalize_transform_inner


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the graph recursively and checks if it contains compute heavy ops like
    convolutions and dense.
    """

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        compute_intensive_ops = {
            "nn.conv2d",
            "qnn.conv2d",
            "nn.conv2d_transpose",
            "nn.dense",
            "qnn.dense",
        }
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op.name) in compute_intensive_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph):
        """
        This function recursively visits the graph and checks if it's compute intensive"
        """
        self.visit(subgraph)
        return self.is_compute_intensive


class IsSupportedGraph(ExprVisitor):
    """
    Visits the graph recursively and checks if function inputs feed into
    any unsupported ops.
    """

    def __init__(self, function):
        ExprVisitor.__init__(self)
        self.is_supported = True
        self.function = function
        self.input_op_list = []

    def _check_legal(self, node, parent_call):
        unsupported_ops = {}

        input_ops = {
            "mrvl.quant_mrvl",
            "mrvl.reshape",
        }

        if isinstance(node, relay.Function):
            if node.attrs["Composite"] in unsupported_ops:
                self.is_supported = False
            if node.attrs["Composite"] in input_ops:
                self.input_op_list.append(parent_call)

    def visit_call(self, call):
        for args in call.args:
            if args in self.function.params or args in self.input_op_list:
                relay.analysis.post_order_visit(
                    call, lambda expr, parent_call=call: self._check_legal(expr, parent_call)
                )

        return super().visit_call(call)

    def is_supported_subgraph(self):
        """
        This function recursively visits the graph and checks if graph is legal"
        """
        self.visit(self.function.body)
        return self.is_supported


class MrvlPartitionGraphValidate(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains only one mrvl partition
    """

    def __init__(self, function):
        ExprVisitor.__init__(self)
        self.function = function
        self.annotate_str = "mrvl"
        self.num_mrvl_partitions = 0
        self.num_llvm_partitions = 0
        self.visit(self.function["main"].body)

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.expr.GlobalVar):
            if self.annotate_str in call.op.name_hint:
                self.num_mrvl_partitions += 1
        elif isinstance(call.op, relay.expr.Call):
            self.num_llvm_partitions += 1
        elif isinstance(call.op, tvm.tir.op.Op):
            self.num_llvm_partitions += 1
        super().visit_call(call)

    def validate_partitioned_graph(self, pre_quantized_model):
        """Validate the partitioned graph"""
        if self.num_mrvl_partitions != 1 or self.num_llvm_partitions > 0:
            if pre_quantized_model:
                raise NotImplementedError(
                    "Mrvl-Compiler-ERROR: The model has been partitioned into multiple regions. "
                    "Multi partition model is not supported in pre_quantized flow yet. "
                    "Please check the Supported Ops and try again.\n"
                )

            print(
                "Mrvl-Compiler-WARNING: The model has been partitioned into multiple regions. "
                "Please ensure that the model is partitioned into a single region for the "
                "best performance. Please check the Supported Ops and try again.\n"
            )


def first_op_unsupported(function):
    return not IsSupportedGraph(function).is_supported_subgraph()


def repartition_subgraph(function, symbol_name):
    """
    Revert back to LLVM if the subgraph is not compute intensive or marked as
    force_llvm.
    """
    if not IsComputeIntensiveGraph().is_graph_compute_intensive(function.body):
        return True

    if first_op_unsupported(function):
        return True

    return False


def repartition_mrvl_subgraphs(mod):
    """
    Un-partition those partitions which:
     - are not computationally intensive subgraph
     - cannot be supported by the backend currently
    """
    global_vars_to_inline = [
        gv
        for gv in mod.get_global_vars()
        if mod[gv].attrs
        and mod[gv].attrs["Compiler"] == "mrvl"
        and repartition_subgraph(mod[gv], mod[gv].attrs["global_symbol"])
    ]
    return relay.transform.InlineCompilerFunctionsBoundTo(global_vars_to_inline)(mod)


def instrument_ir(mod, data_dict, calib_mode, calib_chunk_by, debug):
    """Instrument the IR with simulated_quantize
    annotations to collect tensor range information"""

    calibrate_mode = "mrvl_" + calib_mode
    Quantize = MrvlQuantize()
    with relay.quantize.qconfig(
        calibrate_mode=calibrate_mode,
        weight_scale="mrvl_weight",
        skip_dense_layer=False,
        skip_conv_layers=[],
        do_simulation=False,
        calibrate_chunk_by=calib_chunk_by,
    ):
        mod = Quantize.mrvl_quantize(
            mod,
            dataset=calibrate_dataset(data_dict),
            debug=debug,
        )
    return mod


def calibrate_dataset(data_dict):
    assert len(data_dict.files) == 1
    calib_file = data_dict.files[0]
    return data_dict[calib_file]


def get_input_min_max(func):
    """Get Inputs min/max values from the mrvl subgraph"""
    region_inputs = {}
    input_dict = {}
    no_ops_list = ["nn.batch_flatten", "reshape", "layout_transform"]

    def _is_simulated_quantize(node):
        if isinstance(node, relay.Call):
            if node.op == relay.op.get("relay.op.annotation.simulated_quantize"):
                quant_input = node.args[0]
                if (isinstance(quant_input, relay.expr.Var)) or (
                    isinstance(quant_input, relay.expr.Call)
                    and quant_input.op.name in no_ops_list
                    and isinstance(quant_input.args[0], relay.expr.Var)
                ):
                    _, _, min_val, max_val = node.args
                    input_dict["min"] = str(min_val.data)
                    input_dict["max"] = str(max_val.data)

    def visit_parent(node, input_name):
        if isinstance(node, relay.Call):
            for arg in node.args:
                if isinstance(arg, relay.expr.Var):
                    if input_name == arg.name_hint:
                        if isinstance(node.op, relay.Function):
                            relay.analysis.post_order_visit(node.op.body, _is_simulated_quantize)
                        if isinstance(node.op, relay.Call):
                            relay.analysis.post_order_visit(node, _is_simulated_quantize)

    for param in func.params:
        input_name = param.name_hint
        input_dict = {}
        relay.analysis.post_order_visit(
            func, lambda expr, name=input_name: visit_parent(expr, name)
        )
        region_inputs[param.name_hint] = input_dict

    return region_inputs


def add_attributes(mod, annotate_target_str, add_instrumentation, **kwargs):
    """This method iterates across all Marvell partitioned functions in the
    module and attaches attributes which are supplied by the user from the CLI.
    Use good defaults in case a particular option is not specified. These options
    are later accessed by codegen and are embedded into the runtime.

    Parameters
    ----------
    mod : Module
        The module to attach attributes to
    kwargs : Dict[str, str]
        Dictionary with command line options

    Returns
    -------
    mod : module with attributes
    """
    if "working_dir" in kwargs:
        working_dir = kwargs.get("working_dir")
    else:
        working_dir = mrvl_contrib.get_working_dir()
    if "model_name" in kwargs:
        model_name = kwargs.get("model_name")
    else:
        model_name = "model"
    sim_attr_found = False
    fsim_attr_found = False
    hw_attr_found = False
    quantization_type = ""

    if "mattr" in kwargs:
        base_opts_str = kwargs.get("mattr")

        if "fsim" in base_opts_str:
            fsim_attr_found = True
            base_opts_str = base_opts_str.replace("fsim", "")

        if "sim" in base_opts_str:
            sim_attr_found = True
            base_opts_str = base_opts_str.replace("sim", "")

        if "hw" in base_opts_str:
            hw_attr_found = True
            base_opts_str = base_opts_str.replace("hw", "")

        if hw_attr_found + sim_attr_found + fsim_attr_found > 1:
            error_msg = (
                "Marvell attributes SIM, FSIM and HW are mutually exclusive"
                "Specify any one of the attribute"
            )
            raise RuntimeError(error_msg)

        if "arch" not in base_opts_str:
            base_opts_str = f"{base_opts_str} -arch=mlip"

        if "quantize" not in base_opts_str:
            base_opts_str = f"{base_opts_str} -quantize=fp16"

        if "quantize=int8" in base_opts_str or "pre_quantized_model" in base_opts_str:
            quantization_type = "int8"
        else:
            quantization_type = "fp16"

        pre_quantized_model = "pre_quantized_model" in base_opts_str

        MrvlPartitionGraphValidate(mod).validate_partitioned_graph(pre_quantized_model)

        if "quantize=int8" in base_opts_str and "calibration_data" not in kwargs:
            print("Mrvl-Compiler-WARNING::", end=" ")
            print("Calibration data is required for int8 quantization.", end=" ")
            print("Without calibration data, scaling factors will be set to 0.0")
            print("Please provide calibration data using --target-mrvl-calibration_data option.")

    else:
        base_opts_str = "-arch=mlip -quantize=fp16"

    if "num_tiles" in kwargs:
        base_opts_str = f"{base_opts_str} -num_tiles={kwargs.get('num_tiles')}"
    elif "num_tiles" not in base_opts_str:
        base_opts_str = f"{base_opts_str} -num_tiles=4"

    mode_string = "sim"
    if sim_attr_found:
        mode_string = "sim"
    elif fsim_attr_found:
        mode_string = "fsim"
    elif hw_attr_found:
        mode_string = "hw"

    for var in mod.get_global_vars():
        func_name = var.name_hint
        func = mod[func_name]

        region_id = func_name.split("_")[-1]

        input_quant_dict = {}
        if add_instrumentation and func_name != "main":
            input_quant_dict = get_input_min_max(func)

        if annotate_target_str in func_name:
            func = func.with_attr("working_dir", working_dir)
            func = func.with_attr("compiler_opts_string", base_opts_str)
            func = func.with_attr("mode", mode_string)
            func = func.with_attr("quantization_type", quantization_type)

            func = func.with_attr("model_name", model_name + "_" + region_id)
            func = func.with_attr("input_quant_info", str(input_quant_dict))
            mod.update_func(var, func)

    return mod


def is_valid_batch_size(batch_size):
    if isinstance(batch_size, type(relay.Any())):
        return False
    elif batch_size > 8:
        return False
    else:
        return True


def mrvl_register_conv2d_attr_funcs_for_convert_layout():
    """register the conv2d attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.conv2d convert layout function
    relay.op.get("nn.conv2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.conv2d", "FTVMConvertOpLayout")
    def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.conv2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        weight_info_const = tinfos[1]
        new_attrs["channels"] = weight_info_const.shape[0]
        desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
        new_attrs["data_layout"] = desired_data_layout
        new_attrs["kernel_layout"] = desired_kernel_layout
        new_attrs["out_layout"] = desired_data_layout
        return relay.nn.conv2d(*inputs, **new_attrs)

    return convert_conv2d


def mrvl_register_conv2d_transpose_attr_funcs_for_convert_layout():
    """register the convert conv2d_transpose attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.conv2d_transpose convert layout function
    relay.op.get("nn.conv2d_transpose").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.conv2d_transpose", "FTVMConvertOpLayout")
    def convert_conv2d_transpose(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.conv2d_transpose(*inputs, **attrs)
        new_attrs = dict(attrs)
        weight_info_const = tinfos[1]
        new_attrs["channels"] = weight_info_const.shape[1]
        desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
        new_attrs["data_layout"] = desired_data_layout
        new_attrs["kernel_layout"] = desired_kernel_layout
        new_attrs["out_layout"] = desired_data_layout
        return relay.nn.conv2d_transpose(*inputs, **new_attrs)

    return convert_conv2d_transpose


def mrvl_register_max_pool2d_attr_funcs_for_convert_layout():
    """register the max_pool2d attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.max_pool2d convert layout function
    relay.op.get("nn.max_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.max_pool2d", "FTVMConvertOpLayout")
    def convert_max_pool2d(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.max_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.max_pool2d(*inputs, **new_attrs)

    return convert_max_pool2d


def mrvl_register_avg_pool2d_attr_funcs_for_convert_layout():
    """register the avg_pool2d attr func(s) to convert op layout"""
    # reset first in order to register& use a new nn.avg_pool2d convert layout function
    relay.op.get("nn.avg_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.avg_pool2d", "FTVMConvertOpLayout")
    def convert_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
        if (tinfos[0].shape[0] != 1) and not isinstance(tinfos[0].shape[0], type(relay.Any())):
            return relay.nn.avg_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.avg_pool2d(*inputs, **new_attrs)

    return convert_avg_pool2d


def mrvl_register_global_avg_pool2d_attr_funcs_for_convert_layout():
    """register the global_avg_pool2d attr func(s) to convert op layout"""
    # reset first in order to register& use a new nn.global_avg_pool2d convert layout function
    relay.op.get("nn.global_avg_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.global_avg_pool2d", "FTVMConvertOpLayout")
    def convert_global_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
        if (tinfos[0].shape[0] != 1) and not isinstance(tinfos[0].shape[0], type(relay.Any())):
            return relay.nn.global_avg_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.global_avg_pool2d(*inputs, **new_attrs)

    return convert_global_avg_pool2d


def mrvl_register_upsampling_attr_funcs_for_convert_layout():
    """register the nn.upsampling attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.upsampling convert layout function
    relay.op.get("nn.upsampling").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.upsampling", "FTVMConvertOpLayout")
    def convert_upsampling(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.upsampling(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        return relay.nn.upsampling(*inputs, **new_attrs)

    return convert_upsampling


@register_pattern_table("mrvl")
def mrvl_pattern_table():
    """Get the Mrvl pattern table."""

    def expand_dims_pattern():
        pattern = is_op("expand_dims")(wildcard())
        return pattern

    def qnn_tanh_pattern():
        pattern = is_op("qnn.tanh")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_sigmoid_pattern():
        pattern = is_op("qnn.sigmoid")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_conv_pattern():
        pattern = wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(
            lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern1 = pattern.optional(lambda x: (is_op("maximum")(x, is_constant())))
        pattern2 = pattern.optional(is_op("reinterpret"))
        pattern2 = pattern2.optional(lambda x: (is_op("take")(is_constant(), x)))
        return pattern1 | pattern2

    def qnn_sum_pattern():
        pattern = is_op("qnn.add")(
            wildcard(),
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pattern_1 = pattern.optional(lambda x: (is_op("maximum")(x, is_constant())))
        pattern_2 = pattern.optional(is_op("reinterpret"))
        pattern_2 = pattern_2.optional(lambda x: (is_op("take")(is_constant(), x)))
        return pattern_1 | pattern_2

    def qnn_fc_pattern():
        transform1 = is_op("layout_transform")(wildcard()).has_attr(
            {"src_layout": "NHWC", "dst_layout": "NCHW"}
        )
        reshape = is_op("reshape")(transform1)
        flatten = is_op("nn.batch_flatten")(transform1)
        flatten_or_reshape = reshape | flatten
        pattern = flatten_or_reshape | wildcard()
        pattern = is_op("qnn.dense")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(
            lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
        )
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern1 = pattern.optional(lambda x: (is_op("maximum")(x, is_constant())))
        pattern2 = pattern.optional(is_op("reinterpret"))
        pattern2 = pattern2.optional(lambda x: (is_op("take")(is_constant(), x)))
        return pattern1 | pattern2

    def qnn_globalavgpool2d_pattern():
        pattern = is_op("qnn.requantize")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("nn.global_avg_pool2d")(pattern)
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_averagepool2d_pattern():
        pattern = wildcard()
        pattern = is_op("qnn.avg_pool2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def simulated_quantize_pattern(pattern):
        pattern = pattern.optional(
            lambda x: (
                is_op("relay.op.annotation.simulated_quantize")(
                    x, is_constant(), is_constant(), is_constant()
                )
            )
        )
        return pattern

    def qnn_requantize_pattern(pattern):
        pattern = pattern.optional(
            lambda x: (
                is_op("qnn.requantize")(
                    x, is_constant(), is_constant(), is_constant(), is_constant()
                )
            )
        )
        return pattern

    def requantize_pattern():
        pattern = is_op("qnn.requantize")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_mul_pattern():
        pattern = is_op("qnn.mul")(
            wildcard(),
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pattern = qnn_requantize_pattern(pattern)
        pattern_1 = pattern.optional(lambda x: (is_op("maximum")(x, is_constant())))
        pattern_2 = pattern.optional(is_op("reinterpret"))
        pattern_2 = pattern_2.optional(lambda x: (is_op("take")(is_constant(), x)))
        return pattern_1 | pattern_2

    def conv2d_nhwc2nhwc_pattern():
        """Create a convolution-2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution-2d pattern.
        """

        def conv2d_base_pattern(pattern):
            pattern = is_op("nn.conv2d")(pattern, is_constant())
            pattern = simulated_quantize_pattern(pattern)
            pattern = pattern.optional(
                lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
            )
            pattern = simulated_quantize_pattern(pattern)

            def conv2d_no_batchnorm(pattern):
                # conv + [add] + [relu]
                pattern1 = is_activation(pattern)
                return pattern1

            def conv2d_batchnorm(pattern):
                pattern2 = is_op("nn.batch_norm")(
                    pattern, is_constant(), is_constant(), is_constant(), is_constant()
                )
                pattern2 = simulated_quantize_pattern(pattern2)
                pattern2 = is_tuple_get_item(pattern2, 0)
                pattern2 = simulated_quantize_pattern(pattern2)
                pattern2 = is_activation(pattern2)
                return pattern2

            pattern1 = conv2d_no_batchnorm(pattern)
            pattern2 = conv2d_batchnorm(pattern)

            return pattern1 | pattern2

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = simulated_quantize_pattern(pad)
        pad = conv2d_base_pattern(pad)
        no_pad = wildcard()
        no_pad = conv2d_base_pattern(no_pad)

        return pad | no_pad

    def sum_pattern():
        """Create a sum pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the sum pattern.
        """
        pattern = is_op("add")(wildcard(), wildcard())
        pattern = simulated_quantize_pattern(pattern)
        pattern = is_activation(pattern)
        return pattern

    def mul_pattern():
        pattern = is_op("multiply")(wildcard(), wildcard())
        pattern = simulated_quantize_pattern(pattern)
        pattern = is_activation(pattern)
        return pattern

    def subtract_pattern():
        pattern = is_op("subtract")(wildcard(), wildcard())
        return pattern

    def concat_pattern():
        """Create a concat pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the concat pattern.
        """
        pattern = is_op("concatenate")(is_tuple(None))
        pattern1 = simulated_quantize_pattern(pattern)
        pattern2 = qnn_requantize_pattern(pattern)
        pattern = pattern1 | pattern2
        return pattern

    def fc_pattern_base():
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = simulated_quantize_pattern(pattern)
        pattern = pattern.optional(
            lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
        )
        pattern = simulated_quantize_pattern(pattern)
        pattern = is_activation(pattern)
        return pattern

    def fc_pattern():
        """Create a fc (fully-connected) pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the fc pattern.
        """

        def fc_base_pattern(pattern):
            pattern = is_op("nn.dense")(pattern, is_constant())
            pattern = simulated_quantize_pattern(pattern)
            pattern = pattern.optional(
                lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
            )
            pattern = simulated_quantize_pattern(pattern)
            pattern = is_activation(pattern)

            return pattern

        transform1 = is_op("layout_transform")(wildcard()).has_attr(
            {"src_layout": "NHWC", "dst_layout": "NCHW"}
        )
        reshape = is_op("reshape")(transform1)
        flatten = is_op("nn.batch_flatten")(transform1)
        flatten = reshape | flatten
        flatten = simulated_quantize_pattern(flatten)
        flatten = fc_base_pattern(flatten)

        no_flatten = wildcard()
        no_flatten = fc_base_pattern(no_flatten)

        matmult = is_op("reshape")(flatten)
        matmult = is_op("layout_transform")(matmult).has_attr(
            {"src_layout": "NCHW", "dst_layout": "NHWC"}
        )

        # pylint: disable=E1131 The pattern below is a valid pattern
        return flatten | no_flatten | matmult

    def batch_matmul_pattern():
        layout_transform1 = is_op("layout_transform")(wildcard())
        input_1 = is_op("reshape")(layout_transform1)
        layout_transform2 = is_op("layout_transform")(wildcard())
        input_2 = is_op("transpose")(is_op("reshape")(layout_transform2))
        pattern = is_op("nn.batch_matmul")(input_1, input_2)
        pattern = is_op("reshape")(pattern)
        pattern = is_op("layout_transform")(pattern)

        matmul_with_var = is_op("nn.batch_matmul")(wildcard(), is_op("transpose")(wildcard()))
        matmul_with_const = is_op("nn.batch_matmul")(wildcard(), is_constant())

        return pattern | matmul_with_var | matmul_with_const

    def conv2d_transpose_pattern():
        def conv2d_transpose_base_pattern(pattern):
            pattern = is_op("nn.conv2d_transpose")(pattern, is_constant())
            pattern = simulated_quantize_pattern(pattern)
            pattern = pattern.optional(
                lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
            )
            pattern = simulated_quantize_pattern(pattern)

            def conv2d_trans_no_batchnorm(pattern):
                pattern1 = is_activation(pattern)
                return pattern1

            def conv2d_trans_batchnorm(pattern):
                pattern2 = is_op("nn.batch_norm")(
                    pattern, is_constant(), is_constant(), is_constant(), is_constant()
                )
                pattern2 = simulated_quantize_pattern(pattern2)
                pattern2 = is_tuple_get_item(pattern2, 0)
                pattern2 = simulated_quantize_pattern(pattern2)
                pattern2 = is_activation(pattern2)
                return pattern2

            pattern1 = conv2d_trans_no_batchnorm(pattern)
            pattern2 = conv2d_trans_batchnorm(pattern)

            return pattern1 | pattern2

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = simulated_quantize_pattern(pad)
        pad = conv2d_transpose_base_pattern(pad)

        no_pad = wildcard()
        no_pad = conv2d_transpose_base_pattern(no_pad)

        return pad | no_pad

    def maxpool2d_pattern():
        """Create a maxpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the maxpool2d pattern.
        """

        def maxpool2d_base_pattern(pattern):
            pattern = is_op("nn.max_pool2d")(pattern)
            pattern = simulated_quantize_pattern(pattern)
            return pattern

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = simulated_quantize_pattern(pad)
        pad = maxpool2d_base_pattern(pad)

        no_pad = wildcard()
        no_pad = maxpool2d_base_pattern(no_pad)

        return pad | no_pad

    def avgpool2d_pattern():
        """Create a avgpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the avgpool2d pattern.
        """

        def avgpool2d_base_pattern(pattern):
            pattern = is_op("nn.avg_pool2d")(pattern)
            pattern = simulated_quantize_pattern(pattern)

            return pattern

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = simulated_quantize_pattern(pad)
        pad = avgpool2d_base_pattern(pad)

        no_pad = wildcard()
        no_pad = avgpool2d_base_pattern(no_pad)

        return pad | no_pad

    def globalavgpool2d_pattern():
        """Create a globalavgpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the globalavgpool2d pattern.
        """
        pattern = is_op("nn.global_avg_pool2d")(wildcard())
        return pattern

    def globalmaxpool2d_pattern():
        """Create a globalmaxpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the globalmaxpool2d pattern.
        """
        pattern = is_op("nn.global_max_pool2d")(wildcard())
        return pattern

    def reducemax_pattern():
        """Create a reducemax pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the reducemax pattern.
        """
        pattern = is_op("max")(wildcard())
        return pattern

    def power_pattern():
        """Create a power (fully-connected) pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the power pattern.
        """
        pattern = is_op("power")(wildcard(), is_constant())
        pattern = simulated_quantize_pattern(pattern)

        return pattern

    def leaky_relu_pattern():
        """Create a leaky_relu pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the leaky_relu pattern.
        """
        pattern = is_op("nn.leaky_relu")(wildcard())
        return pattern

    def softmax_pattern():
        """Create a leaky_relu pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the softmax pattern.
        """
        pattern = is_op("nn.softmax")(wildcard())
        return pattern

    def argmax_pattern():
        """Create argmax pattern"""
        pattern = is_op("argmax")(wildcard())
        pattern = is_op("cast")(pattern)
        return pattern

    def topk_pattern():
        """Create topk pattern"""
        pattern = is_op("topk")(wildcard())
        pattern_L = is_tuple_get_item(pattern, 0)
        pattern_R = is_tuple_get_item(pattern, 1)
        pattern = is_tuple([pattern_L, pattern_R])
        return pattern

    def relu_pattern():
        """Create a relu pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the relu pattern.
        """
        pattern = is_op("nn.relu")(wildcard())
        return pattern

    def tanh_pattern():
        """Create a tanh pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the tanh pattern.
        """
        pattern = is_op("tanh")(wildcard())
        return pattern

    def sigmoid_pattern():
        """Create a sigmoid pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the sigmoid pattern.
        """
        pattern = is_op("sigmoid")(wildcard())
        return pattern

    def clip_pattern():
        """Create a clip pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the clip pattern.
        """
        pattern = is_op("clip")(wildcard())
        return pattern

    def split_pattern():
        """Create a split pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the split pattern.
        """
        pattern = is_op("split")(wildcard())
        return pattern

    def resize2d_pattern():
        """Create a resize2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the resize2d pattern.
        """
        resize2d_pattern = is_op("image.resize2d")(wildcard())
        upsample_pattern = is_op("nn.upsampling")(wildcard())
        pattern = resize2d_pattern | upsample_pattern
        return pattern

    def rsqrt_pattern():
        pattern = is_op("rsqrt")(wildcard())
        return pattern

    def strided_slice_pattern():
        pattern = is_op("strided_slice")(wildcard())
        pattern = simulated_quantize_pattern(pattern)
        return pattern

    def batch_norm_pattern():
        """Create a batch norm pattern."""
        pattern = is_op("nn.batch_norm")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def quant_layout_pattern():
        pattern = is_op("relay.op.annotation.simulated_quantize")(
            wildcard(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(is_op("layout_transform"))

        pattern1 = is_op("qnn.quantize")(wildcard(), is_constant(), is_constant())
        pattern1 = pattern1.optional(is_op("layout_transform"))

        pattern2 = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())

        pattern3 = is_op("layout_transform")(wildcard())
        pattern3 = is_op("qnn.dequantize")(pattern3, is_constant(), is_constant())
        dequantize = pattern3 | pattern2
        return pattern | pattern1 | dequantize

    def reshape_pattern():
        pattern = is_op("reshape")(wildcard())
        pattern = simulated_quantize_pattern(pattern)
        return pattern

    def batch_flatten_pattern():
        pattern = is_op("nn.batch_flatten")(wildcard())
        pattern = simulated_quantize_pattern(pattern)
        return pattern

    def squeeze_pattern():
        pattern = is_op("squeeze")(wildcard())
        pattern = simulated_quantize_pattern(pattern)
        return pattern

    def layout_transform_pattern():
        pattern = is_op("layout_transform")(is_var(), wildcard(), wildcard())
        return pattern

    def transpose_pattern():
        pattern = is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [0, 1, 3, 2]})
        pattern |= is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [0, 2, 1, 3]})
        pattern |= is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [0, 3, 1, 2]})
        pattern |= is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [0, 2, 1]})
        pattern |= is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [1, 0, 2]})
        pattern |= is_op("transpose")(is_var(), wildcard()).has_attr({"axes=": [2, 0, 1]})
        return pattern

    def reduce_pattern():
        pattern = is_op("mean")(wildcard())
        pattern = simulated_quantize_pattern(pattern)
        return pattern

    def check_qnn(extract):
        return True

    def check_conv2d(extract):
        """Check conv pattern is supported by Mrvl."""
        call = extract
        while isinstance(call, TupleGetItem) or (call.op.name != "nn.conv2d"):
            if isinstance(call, TupleGetItem):
                call = call.tuple_value
            else:
                call = call.args[0]
        return conv2d_nhwc2nhwc(call)

    def check_conv2d_transpose(extract):
        """Check conv pattern is supported by Mrvl."""
        call = extract
        while isinstance(call, TupleGetItem) or (call.op.name != "nn.conv2d_transpose"):
            if isinstance(call, TupleGetItem):
                call = call.tuple_value
            else:
                call = call.args[0]
        return conv2d_transpose(call)

    def check_fc(extract):
        """Check fc pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.dense":
            call = call.args[0]
        return fc_ni2no(call)

    def check_batch_matmul(extract):
        """Check batch_matmul pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.batch_matmul":
            call = call.args[0]
        return batch_matmul(call)

    def check_maxpool2d(extract):
        """Check maxpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.max_pool2d":
            call = call.args[0]
        return maxpool2d_nhwc2nhwc(call)

    def check_avgpool2d(extract):
        """Check avgpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.avg_pool2d":
            call = call.args[0]
        return avgpool2d_nhwc2nhwc(call)

    def check_globalavgpool2d(extract):
        """Check globalavgpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.global_avg_pool2d":
            call = call.args[0]
        return globalavgpool2d_nhwc2nhwc(call)

    def check_globalmaxpool2d(extract):
        """Check globalmaxpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.global_max_pool2d":
            call = call.args[0]
        return globalmaxpool2d_nhwc2nhwc(call)

    def check_leaky_relu(extract):
        """Check leaky_relu pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.leaky_relu":
            call = call.args[0]
        return leaky_relu(call)

    def check_softmax(extract):
        """Check softmax pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.softmax":
            call = call.args[0]
        return softmax(call)

    def check_argmax(extract):
        """Check argmax pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "argmax":
            call = call.args[0]
        return argmax(call)

    def check_topk(extract):
        """Check topk pattern is supported by Mrvl."""
        call = extract

        while isinstance(call, (Tuple, TupleGetItem)) or call.op.name != "topk":
            if isinstance(call, Tuple):
                call = call.fields[0]
            elif isinstance(call, TupleGetItem):
                call = call.tuple_value
            else:
                call = call.args[0]
        return topk(call)

    def check_power(extract):
        """Check power pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "power":
            call = call.args[0]
        return power(call)

    def check_relu(extract):
        """Check relu pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.relu":
            call = call.args[0]
        return relu(call)

    def check_tanh(extract):
        """Check tanh pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "tanh":
            call = call.args[0]
        return tanh(call)

    def check_sigmoid(extract):
        """Check sigmoid pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "sigmoid":
            call = call.args[0]
        return sigmoid(call)

    def check_clip(extract):
        """Check clip pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "clip":
            call = call.args[0]
        return clip(call)

    def check_split(extract):
        """Check split pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "split":
            call = call.args[0]
        return split(call)

    def check_resize2d(extract):
        """Check resize2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "image.resize2d" and call.op.name != "nn.upsampling":
            call = call.args[0]
        return resize2d(call)

    def check_rsqrt(extract):
        """Check rsqrt pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "rsqrt":
            call = call.args[0]
        return rsqrt(call)

    def check_stride_slice(extract):
        """Check strided_slice pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "strided_slice":
            call = call.args[0]
        return strided_slice(call)

    def check_batch_norm(extract):
        """Check batch_norm pattern is supported by Mrvl."""
        call = extract
        return batch_norm(call)

    def check_quant(extract):
        call = extract
        while call.op.name == "layout_transform":
            call = call.args[0]
        return quant_mrvl(call)

    def check_reshape(extract):
        call = extract
        while call.op.name != "reshape":
            call = call.args[0]
        return reshape_mrvl(call)

    def check_batch_flatten(extract):
        call = extract
        while call.op.name != "nn.batch_flatten":
            call = call.args[0]
        return batch_flatten_mrvl(call)

    def check_squeeze(extract):
        call = extract
        while call.op.name != "squeeze":
            call = call.args[0]
        return squeeze_mrvl(call)

    def check_layout_transform(extract):
        call = extract
        while call.op.name != "layout_transform":
            call = call.args[0]
        return layout_transform(call)

    def check_transpose(extract):
        call = extract
        while call.op.name != "transpose":
            call = call.args[0]
        return transpose(call)

    def check_sum(extract):
        """Check sum2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "add":
            call = call.args[0]
        return summation(call)

    def check_mul(extract):
        """Check mul pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "multiply":
            call = call.args[0]
        return mul(call)

    def check_concat(extract):
        """Check concat pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "concatenate":
            call = call.args[0]
        return concat(call)

    def check_reduce(extract):
        """Check reduce pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "mean":
            call = call.args[0]
        return reduce_mean(call)

    def check_reducemax(extract):
        """Check reducemax pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "max":
            call = call.args[0]
        return reduce_max(call)

    def check_subtract(extract):
        """Check subtract pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "subtract":
            call = call.args[0]
        return subtract(call)

    return [
        ("mrvl.conv2d_nhwc2nhwc", conv2d_nhwc2nhwc_pattern(), check_conv2d),
        ("mrvl.conv2d_transpose", conv2d_transpose_pattern(), check_conv2d_transpose),
        ("mrvl.qnn_conv2d", qnn_conv_pattern(), check_qnn),
        ("mrvl.expand_dims", expand_dims_pattern(), check_qnn),
        ("mrvl.reduce_max", reducemax_pattern(), check_reducemax),
        ("mrvl.qnn_add", qnn_sum_pattern(), check_qnn),
        ("mrvl.qnn_fc_ni2no", qnn_fc_pattern(), check_qnn),
        ("mrvl.fc_ni2no", fc_pattern(), check_fc),
        ("mrvl.fc_ni2no", fc_pattern_base(), check_fc),
        ("mrvl.batch_matmul", batch_matmul_pattern(), check_batch_matmul),
        ("mrvl.maxpool2d_nhwc2nhwc", maxpool2d_pattern(), check_maxpool2d),
        ("mrvl.avgpool2d_nhwc2nhwc", avgpool2d_pattern(), check_avgpool2d),
        ("mrvl.qnn_avg_pool2d", qnn_averagepool2d_pattern(), check_qnn),
        ("mrvl.qnn_globalavgpool2d_nhwc2nhwc", qnn_globalavgpool2d_pattern(), check_qnn),
        ("mrvl.globalavgpool2d_nhwc2nhwc", globalavgpool2d_pattern(), check_globalavgpool2d),
        ("mrvl.globalmaxpool2d_nhwc2nhwc", globalmaxpool2d_pattern(), check_globalmaxpool2d),
        ("mrvl.sum", sum_pattern(), check_sum),
        ("mrvl.mul", mul_pattern(), check_mul),
        ("mrvl.qnn_mul", qnn_mul_pattern(), check_qnn),
        ("mrvl.concat", concat_pattern(), check_concat),
        ("mrvl.transpose", transpose_pattern(), check_transpose),
        ("mrvl.layout_transform", layout_transform_pattern(), check_layout_transform),
        ("mrvl.reshape", reshape_pattern(), check_reshape),
        ("mrvl.batch_flatten", batch_flatten_pattern(), check_batch_flatten),
        ("mrvl.squeeze", squeeze_pattern(), check_squeeze),
        ("mrvl.strided_slice", strided_slice_pattern(), check_stride_slice),
        ("mrvl.quant_mrvl", quant_layout_pattern(), check_quant),
        ("mrvl.leaky_relu", leaky_relu_pattern(), check_leaky_relu),
        ("mrvl.softmax", softmax_pattern(), check_softmax),
        ("mrvl.argmax", argmax_pattern(), check_argmax),
        ("mrvl.topk", topk_pattern(), check_topk),
        ("mrvl.power", power_pattern(), check_power),
        ("mrvl.relu", relu_pattern(), check_relu),
        ("mrvl.tanh", tanh_pattern(), check_tanh),
        ("mrvl.qnn_tanh", qnn_tanh_pattern(), check_qnn),
        ("mrvl.sigmoid", sigmoid_pattern(), check_sigmoid),
        ("mrvl.qnn_sigmoid", qnn_sigmoid_pattern(), check_qnn),
        ("mrvl.clip", clip_pattern(), check_clip),
        ("mrvl.split", split_pattern(), check_split),
        ("mrvl.resize2d", resize2d_pattern(), check_resize2d),
        ("mrvl.rsqrt", rsqrt_pattern(), check_rsqrt),
        ("mrvl.batch_norm", batch_norm_pattern(), check_batch_norm),
        ("mrvl.reduce_mean", reduce_pattern(), check_reduce),
        ("mrvl.subtract", subtract_pattern(), check_subtract),
        ("mrvl.qnn_requantize", requantize_pattern(), check_qnn),
    ]


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported by Mrvl.
    Note that this function resets the attribute target.mrvl in case it already exists.

    Parameters
    ---------
    op_name : Str
        The name of operator that will be registered.
    supported: Boolean
        Whether the op can be supported by the backend or not.
    Returns
    -------
    f : callable
        A function that returns if the operator is supported by Marvell.
    """
    relay.op.get(op_name).reset_attr("target.mrvl")

    @tvm.ir.register_op_attr(op_name, "target.mrvl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.conv2d", "target.mrvl")
def conv2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for conv2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_type = args[0].checked_type
    if (
        (len(data_type.shape) != 4)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False
    if "KERNEL_WIDTH_LIMITATION" in os.environ:
        kernel_width_limitation = int(os.environ["KERNEL_WIDTH_LIMITATION"])
        if data_type.shape[2] > kernel_width_limitation:
            return False
    kernel_typ = args[1].checked_type
    if (len(kernel_typ.shape) != 4) or (kernel_typ.dtype not in ["float32"]):
        return False

    is_depthwise = is_depthwise_conv2d(
        data_type.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        # Mrvl support grouped conv only for groups == ch
        return bool(attrs.groups == kernel_typ.shape[0])
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.conv2d_transpose", "target.mrvl")
def conv2d_transpose(expr):
    """Check if the external Mrvl codegen for conv2d_transpose should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_type = args[0].checked_type

    if (
        (len(data_type.shape) != 4)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False
    kernel_typ = args[1].checked_type
    if (len(kernel_typ.shape) != 4) or (kernel_typ.dtype not in ["float32"]):
        return False
    is_depthwise = is_depthwise_conv2d(
        data_type.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        return bool(attrs.groups == kernel_typ.shape[0])
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("add", "target.mrvl")
def summation(expr):
    """Check if the external Mrvl codegen for sum should be used."""
    arg0 = expr.args[0]

    # - need to further checking if the call_func of arg0 is not nn.conv2d nor nn.dense
    if (
        isinstance(arg0, Call)
        and isinstance(arg0.op, tvm.ir.Op)
        and arg0.op.name in ["nn.conv2d_transpose", "nn.conv2d", "nn.dense"]
    ):
        return False

    # - need to further checking if dimension of input or output tensor is 4
    data_type = arg0.checked_type
    if (
        (len(data_type.shape) != 4 and len(data_type.shape) != 3 and len(data_type.shape) != 2)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("multiply", "target.mrvl")
def mul(expr):
    """Check if the external Mrvl codegen for mul should be used."""
    arg0 = expr.args[0]
    # - need to further checking if dimension of input or output tensor is 4
    data_type = arg0.checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 3 or len(data_type.shape) == 2):
        return False
    if ((data_type.shape[0] != 1) and not isinstance(data_type.shape[0], type(relay.Any()))) or (
        data_type.dtype not in ["float32"]
    ):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("concatenate", "target.mrvl")
def concat(expr):
    """Check if the external Mrvl codegen for concat should be used."""
    attrs, args = expr.attrs, expr.args
    arg0 = args[0]
    assert not isinstance(arg0, Call)

    # check data types for both inputs
    # - only support 4-dimension input tensors in NHWC
    # - only support batch size is 1
    data_type_a = arg0.checked_type.fields[0]
    data_type_b = arg0.checked_type.fields[1]
    if (
        (len(data_type_a.shape) != 4)
        or (len(data_type_b.shape) != 4)
        or (data_type_a.shape[0] != 1)
        or (data_type_b.shape[0] != 1)
        or (data_type_a.dtype not in ["float32"])
        or (data_type_b.dtype not in ["float32"])
    ):
        return False

    for data_type in arg0.checked_type.fields:
        if (
            (len(data_type.shape) != 4)
            or (data_type.shape[0] != 1)
            or (data_type.dtype not in ["float32"])
        ):
            return False

    if attrs["axis"] != 3:
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.batch_matmul", "target.mrvl")
def batch_matmul(expr):
    """Check if the external Mrvl codegen for batch_matmul should be used."""
    attrs, args = expr.attrs, expr.args
    first_data_type = args[0].checked_type
    second_data_type = args[1].checked_type

    if first_data_type.dtype not in ["float32"]:
        return False
    if second_data_type.dtype not in ["float32"]:
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.dense", "target.mrvl")
def fc_ni2no(expr):
    """Check if the external Mrvl codegen for fc_ni2no should be used."""
    attrs, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    if data_type.dtype not in ["float32"]:
        return False
    kernel_typ = args[1].checked_type
    if (len(kernel_typ.shape) != 2) or (kernel_typ.dtype not in ["float32"]):
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.max_pool2d", "target.mrvl")
def maxpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for maxpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if (
        (len(data_type.shape) != 4)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32", "int8"])
    ):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.avg_pool2d", "target.mrvl")
def avgpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for avgpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if len(data_type.shape) != 4 or (data_type.dtype not in ["float32"]):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.mrvl")
def globalavgpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for globalavgpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
        return False
    if data_type.dtype not in ["float32"]:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.mrvl")
def globalmaxpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for globalmaxpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
        return False
    if data_type.dtype not in ["float32", "int8"]:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.leaky_relu", "target.mrvl")
def leaky_relu(expr):
    """Claim leaky_relu to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.softmax", "target.mrvl")
def softmax(expr):
    """Claim softmax to be done on MRVL backend"""
    attrs, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    if not is_valid_batch_size(data_type.shape[0]):
        return False
    if len(data_type.shape) != 2 and len(data_type.shape) != 4:
        return False
    if len(data_type.shape) == 4 and attrs["axis"] != 3:
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("argmax", "target.mrvl")
def argmax(expr):
    """Claim argmax to be done on MRVL backend"""
    attrs, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    print(data_type)
    axis = attrs["axis"]
    if not isinstance(axis, int):
        assert len(axis) == 1
        axis = axis[0]
    if not is_valid_batch_size(data_type.shape[0]) or (data_type.dtype not in ["float32", "int8"]):
        return False
    if len(data_type.shape) == 4 and axis == 3:
        return bool(data_type.shape[1] == 1 and data_type.shape[2] == 1)
    if len(data_type.shape) == 2 and axis == 1:
        return True
    return False


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("topk", "target.mrvl")
def topk(expr):
    """Claim topk to be done on MRVL backend"""
    attrs, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    axis = attrs["axis"]
    k = attrs["k"]
    if k > 16:
        return False
    if not isinstance(axis, int):
        assert len(axis) == 1
        axis = axis[0]
    if not is_valid_batch_size(data_type.shape[0]) or (data_type.dtype not in ["float32", "int8"]):
        return False
    if len(data_type.shape) == 4 and axis == 3:
        return bool(data_type.shape[1] == 1 and data_type.shape[2] == 1)
    if len(data_type.shape) == 2 and axis == 1:
        return True
    return False


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.relu", "target.mrvl")
def relu(expr):
    """Claim relu to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("power", "target.mrvl")
def power(expr):
    """Check if the external Mrvl codegen for fc_ni2no should be used."""
    args = expr.args
    data_type = args[0].checked_type
    if data_type.dtype not in ["float32"]:
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 0:
        return False
    return True


@tvm.ir.register_op_attr("tanh", "target.mrvl")
def tanh(expr):
    """Claim tanh to be done on MRVL backend"""
    return True


@tvm.ir.register_op_attr("sigmoid", "target.mrvl")
def sigmoid(expr):
    """Claim sigmoid to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.batch_norm", "target.mrvl")
def batch_norm(expr):
    """Claim batch_norm to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("clip", "target.mrvl")
def clip(expr):
    """Claim clip to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("split", "target.mrvl")
def split(expr):
    """Claim split to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("image.resize2d", "target.mrvl")
@tvm.ir.register_op_attr("nn.upsampling", "target.mrvl")
def resize2d(expr):
    """Claim resize2d to be done on MRVL backend"""
    method = expr.attrs["method"]
    if method != "nearest_neighbor":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("rsqrt", "target.mrvl")
def rsqrt(expr):
    """Claim rsqrt to be done on MRVL backend"""
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("strided_slice", "target.mrvl")
def strided_slice(expr):
    """Claim strided_slice to be done on MRVL backend"""
    return True


@tvm.ir.register_op_attr("relay.op.annotation.simulated_quantize", "target.mrvl")
def quant_mrvl(expr):
    if expr.op.name == "relay.op.annotation.simulated_quantize":
        return True
    elif expr.op.name == "qnn.quantize" or expr.op.name == "qnn.dequantize":
        return True
    else:
        return False


@tvm.ir.register_op_attr("reshape", "target.mrvl")
def reshape_mrvl(expr):
    """Check if the external Mrvl codegen for reshape should be used."""
    if expr.op.name != "reshape":
        return False
    data_type = expr.checked_type
    if len(data_type.shape) > 4 or len(data_type.shape) < 2:
        return False

    args = expr.args
    data_type = args[0].checked_type
    if len(data_type.shape) > 4 or len(data_type.shape) < 2:
        return False

    return True


@tvm.ir.register_op_attr("nn.batch_flatten", "target.mrvl")
def batch_flatten_mrvl(expr):
    """Check if the external Mrvl codegen for batch_flatten should be used."""
    if expr.op.name != "nn.batch_flatten":
        return False
    return True


@tvm.ir.register_op_attr("squeeze", "target.mrvl")
def squeeze_mrvl(expr):
    """Check if the external Mrvl codegen for squeeze should be used."""
    if expr.op.name != "squeeze":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("layout_transform", "target.mrvl")
def layout_transform(expr):
    """Check wethere current expression is float32 and supported by Mrvl"""
    _, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    if data_type.dtype not in ["float32", "int8"]:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("transpose", "target.mrvl")
def transpose(expr):
    """Check if Mrvl codegen for transpose should be used."""
    attrs = expr.attrs
    if len(attrs["axes"]) < 3 or len(attrs["axes"]) > 4:
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("mean", "target.mrvl")
def reduce_mean(expr):
    """Claim reduce to be done on MRVL backend"""
    arg0 = expr.args[0]
    attrs = expr.attrs
    data_type = arg0.checked_type

    if len(attrs["axis"]) > 1:
        return False

    axis_to_check = (attrs["axis"][0] + len(data_type.shape)) % len(data_type.shape)

    if len(data_type.shape) == 4 and (axis_to_check == 1 or axis_to_check == 3):
        return True

    if len(data_type.shape) == 3 and (axis_to_check == 2):
        return True

    return False


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("max", "target.mrvl")
def reduce_max(expr):
    """Claim reduce to be done on MRVL backend"""
    arg0 = expr.args[0]
    attrs = expr.attrs
    data_type = arg0.checked_type
    axis = attrs["axis"][0]
    if len(data_type.shape) != 3 or axis not in [1, 2]:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("subtract", "target.mrvl")
def subtract(expr):
    """Check if the external Mrvl codegen for subtract should be used."""
    arg0 = expr.args[0]
    # - need to further checking if dimension of input or output tensor is 4
    data_type = arg0.checked_type
    if len(data_type.shape) > 4 or len(data_type.shape) < 2:
        return False
    if ((data_type.shape[0] != 1) and not isinstance(data_type.shape[0], type(relay.Any()))) or (
        data_type.dtype not in ["float32"]
    ):
        return False
    return True


class TopKArgmaxOutputCast(ExprMutator):
    """Cast int32 to fp32 to work around fp32 output only limitation"""

    def visit_call(self, call):
        valid_ops = ["argmax", "topk"]
        valid_ops = [relay.op.get(op) for op in valid_ops]

        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))

        if call.op in valid_ops:
            orig_dtype = None
            if orig_dtype is None:
                new_mod = IRModule.from_expr(call)
                new_mod = InferType()(new_mod)
                checked_arg = new_mod["main"].body
                if (
                    isinstance(tuple_expr, relay.Call)
                    and tuple_expr.op.name == "vision.all_class_non_max_suppression"
                ):
                    orig_dtype = checked_arg.checked_type.fields[1].dtype
                    c = Call(new_fn, args, call.attrs)
                    t = relay.Tuple(
                        [relay.TupleGetItem(c, 0), relay.cast(relay.TupleGetItem(c, 1), "float32")]
                    )
                    return relay.Tuple(
                        [relay.TupleGetItem(t, 0), relay.cast(relay.TupleGetItem(t, 1), orig_dtype)]
                    )
                orig_dtype = checked_arg.checked_type.dtype
                return relay.cast(relay.cast(Call(new_fn, args, call.attrs), "float32"), orig_dtype)
            else:
                return relay.cast(relay.cast(Call(new_fn, args, call.attrs), "float32"), orig_dtype)

        return Call(new_fn, args, call.attrs)


class MovePadAfterLayoutTransform(ExprMutator):
    """Move nn.pad node after layout_transform NCHW->NHWC node"""

    def visit_call(self, call):
        new_call = super().visit_call(call)

        if hasattr(new_call.op, "name") and new_call.op.name == "layout_transform":
            pad_call = new_call.args[0]
            if (
                isinstance(pad_call, Call)
                and hasattr(pad_call.op, "name")
                and pad_call.op.name == "nn.pad"
            ):
                pad_width = pad_call.attrs.pad_width
                pad_value = pad_call.args[1]
                input_expr = pad_call.args[0]

                src_layout = new_call.attrs.src_layout
                dst_layout = new_call.attrs.dst_layout

                if src_layout == "NCHW" and dst_layout == "NHWC":
                    new_pad_width = (
                        tuple(pad_width[0]),
                        tuple(pad_width[2]),
                        tuple(pad_width[3]),
                        tuple(pad_width[1]),
                    )

                    transformed = relay.layout_transform(
                        input_expr, src_layout="NCHW", dst_layout="NHWC"
                    )

                    padded = relay.nn.pad(transformed, new_pad_width, pad_value)
                    return padded

        return new_call


class MrvlOptimizeBatchnorm(ExprMutator):
    """
    This class implements a Relay expression mutator that performs BatchNorm folding into Dense
    and Conv2D layers, as well as handling Conv->Add->BatchNorm, Conv->BatchNorm,
    Dense->Add->BatchNorm, and Dense->BatchNorm cases. It checks for the presence of BatchNorm and
    applies the correct fusion strategy based on the preceding operation.
    """

    def visit_call(self, call) -> relay.expr.Expr:

        if hasattr(call.op, "name") and call.op.name == "nn.batch_norm":

            try:
                input_tensor = call.args[0]
                gamma = call.args[1].data.numpy()
                beta = call.args[2].data.numpy()
                mean = call.args[3].data.numpy()
                variance = call.args[4].data.numpy()
                epsilon = call.attrs.epsilon if hasattr(call.attrs, "epsilon") else 1e-5

                if isinstance(input_tensor, relay.Call) and input_tensor.op == relay.op.get(
                    "nn.bias_add"
                ):
                    bias_add_op = input_tensor
                    conv_output = bias_add_op.args[0]
                    bias = bias_add_op.args[1].data.numpy()

                    if isinstance(conv_output, relay.Call) and conv_output.op == relay.op.get(
                        "nn.conv2d"
                    ):
                        conv_op = conv_output
                        conv_input = conv_op.args[0]
                        conv_weight = conv_op.args[1].data.numpy()

                        scale = gamma / np.sqrt(variance + epsilon)
                        scale = scale.reshape(-1, 1, 1, 1)

                        new_weight = conv_weight * scale

                        new_bias = bias + beta - (gamma * mean / np.sqrt(variance + epsilon))

                        new_conv = relay.nn.conv2d(
                            conv_input,
                            relay.const(new_weight),
                            channels=conv_op.attrs.channels,
                            kernel_size=conv_op.attrs.kernel_size,
                            strides=conv_op.attrs.strides,
                            padding=conv_op.attrs.padding,
                            dilation=conv_op.attrs.dilation,
                            groups=conv_op.attrs.groups,
                        )

                        new_conv_out = relay.nn.bias_add(new_conv, relay.const(new_bias))

                        new_conv_out = relay.Tuple([new_conv_out])

                        return self.visit(new_conv_out)

                elif isinstance(input_tensor, relay.Call) and input_tensor.op == relay.op.get(
                    "nn.conv2d"
                ):
                    conv_op = input_tensor
                    conv_input = conv_op.args[0]
                    conv_weight = conv_op.args[1].data.numpy()

                    if len(conv_op.args) > 2:
                        conv_bias = conv_op.args[2].data.numpy()
                    else:
                        conv_bias = np.zeros(conv_weight.shape[0])

                    scale = gamma / np.sqrt(variance + epsilon)
                    scale = scale.reshape(-1, 1, 1, 1)

                    new_weight = conv_weight * scale

                    new_bias = conv_bias + beta - (gamma * mean / np.sqrt(variance + epsilon))

                    new_conv = relay.nn.conv2d(
                        conv_input,
                        relay.const(new_weight),
                        channels=conv_op.attrs.channels,
                        kernel_size=conv_op.attrs.kernel_size,
                        strides=conv_op.attrs.strides,
                        padding=conv_op.attrs.padding,
                        dilation=conv_op.attrs.dilation,
                        groups=conv_op.attrs.groups,
                    )

                    new_conv_out = relay.nn.bias_add(new_conv, relay.const(new_bias))

                    new_conv_out = relay.Tuple([new_conv_out])

                    return self.visit(new_conv_out)

                elif isinstance(input_tensor, relay.Call) and input_tensor.op == relay.op.get(
                    "add"
                ):
                    add_op = input_tensor
                    dense_output = add_op.args[0]
                    add_bias = add_op.args[1].data.numpy()

                    if isinstance(dense_output, relay.Call) and dense_output.op == relay.op.get(
                        "nn.dense"
                    ):
                        dense_op = dense_output
                        dense_input = dense_op.args[0]
                        weight = dense_op.args[1].data.numpy()

                        if len(dense_op.args) > 2:
                            dense_bias = dense_op.args[2].data.numpy()
                        else:
                            dense_bias = add_bias

                        scale = gamma / np.sqrt(variance + epsilon)
                        scale = scale.reshape(-1, 1)
                        new_weight = weight * scale
                        new_bias = dense_bias + beta - (gamma * mean / np.sqrt(variance + epsilon))

                        new_dense = relay.nn.dense(
                            dense_input, relay.const(new_weight), units=dense_op.attrs.units
                        )

                        new_add = relay.nn.bias_add(new_dense, relay.const(new_bias))

                        new_add = relay.Tuple([new_add])

                        return self.visit(new_add)

                elif isinstance(input_tensor, relay.Call) and input_tensor.op == relay.op.get(
                    "nn.dense"
                ):
                    dense_op = input_tensor
                    dense_input = dense_op.args[0]
                    weight = dense_op.args[1].data.numpy()

                    if len(dense_op.args) > 2:
                        dense_bias = dense_op.args[2].data.numpy()
                    else:
                        dense_bias = np.zeros(weight.shape[0])

                    scale = gamma / np.sqrt(variance + epsilon)
                    scale = scale.reshape(-1, 1)
                    new_weight = weight * scale
                    new_bias = dense_bias + beta - (gamma * mean / np.sqrt(variance + epsilon))

                    new_dense = relay.nn.dense(
                        dense_input, relay.const(new_weight), units=dense_op.attrs.units
                    )

                    new_dense_out = relay.nn.bias_add(new_dense, relay.const(new_bias))

                    new_dense_out = relay.Tuple([new_dense_out])

                    return self.visit(new_dense_out)

            except Exception as e:
                print("Error during BatchNorm fusion:", e)
                raise

        return super().visit_call(call)


@relay.transform.function_pass(opt_level=0)
class MrvlOptimizeBatchnormPass:
    """
    A function pass that optimizes BatchNorm operations in Relay graphs.
    This pass uses the MrvlOptimizeBatchnorm class to rewrite the graph.
    """

    def transform_function(self, func, mod, _):
        return MrvlOptimizeBatchnorm().visit(func)


class SimplifyQnnLayoutTransformCallback(DFPatternCallback):
    """
    A callback class to perform the rewrite for both qnn.quantize and qnn.dequantize
    surrounded by layout_transform pairs.
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = self._get_pattern()

    def _get_pattern(self):
        """
        Defines the pattern to match:
        layout_transform -> qnn.quantize|qnn.dequantize -> layout_transform
        """
        x = wildcard()
        layout_transform1 = is_op("layout_transform")(x)
        quantize = is_op("qnn.quantize")(layout_transform1, wildcard(), wildcard())
        dequantize = is_op("qnn.dequantize")(layout_transform1, wildcard(), wildcard())
        quant_or_dequant = quantize | dequantize
        layout_transform2 = is_op("layout_transform")(quant_or_dequant)
        return layout_transform2

    def callback(self, pre, post, node_map):
        """
        The callback function that gets called when the pattern is matched.
        """
        layout_transform2 = pre
        quant_or_dequant = layout_transform2.args[0]
        layout_transform1 = quant_or_dequant.args[0]
        x = layout_transform1.args[0]
        src_layout = layout_transform1.attrs.src_layout
        dst_layout = layout_transform1.attrs.dst_layout
        final_src_layout = layout_transform2.attrs.src_layout
        final_dst_layout = layout_transform2.attrs.dst_layout

        if src_layout == final_dst_layout and dst_layout == final_src_layout:
            if quant_or_dequant.op.name == "qnn.quantize":
                output_scale = quant_or_dequant.args[1]
                output_zero_point = quant_or_dequant.args[2]
                new_quantize = relay.qnn.op.quantize(
                    x,
                    output_scale=output_scale,
                    output_zero_point=output_zero_point,
                    out_dtype=quant_or_dequant.attrs.out_dtype,
                    axis=quant_or_dequant.attrs.axis,
                )
                return new_quantize
            elif quant_or_dequant.op.name == "qnn.dequantize":
                input_scale = quant_or_dequant.args[1]
                input_zero_point = quant_or_dequant.args[2]
                new_dequantize = relay.qnn.op.dequantize(
                    x,
                    input_scale=input_scale,
                    input_zero_point=input_zero_point,
                    axis=quant_or_dequant.attrs.axis,
                )
                return new_dequantize

        return post


@relay.transform.function_pass(opt_level=0)
class SimplifyQnnLayoutTransform:
    """
    The function pass that can be used in a Sequential pass.
    This pass uses the DFPatternCallback to rewrite the graph.
    """

    def transform_function(self, func, mod, ctx):
        """The entry point for the pass."""
        return rewrite(SimplifyQnnLayoutTransformCallback(), func)


@relay.transform.function_pass(opt_level=0)
class MrvlTopKArgmaxOutputCastPass:
    """Removes Dropouts."""

    def __init__(self, skip_topk_argmax_output_cast=False):
        self.skip_topk_argmax_output_cast = skip_topk_argmax_output_cast

    def transform_function(self, func, mod, _):
        """Call TopKArgmaxOutputCast func if not skipped."""
        if self.skip_topk_argmax_output_cast:
            return func
        return TopKArgmaxOutputCast().visit(func)


@relay.transform.function_pass(opt_level=0)
class MrvlNMSInferOutputPass:
    """Replaces dynamic strided_slice with static slice using NMS parameters."""

    def __init__(self, skip_nms_infer_output=False):
        self.skip_nms_infer_output = skip_nms_infer_output

    def transform_function(self, func, mod, _):
        if self.skip_nms_infer_output:
            return func
        return NMSInferOutput().visit(func)


@relay.transform.function_pass(opt_level=0)
class MrvlMovePadAfterLayoutTransformPass:
    """Moves nn.pad node after the layout_transform node for the nn.conv2d."""

    def transform_function(self, func, mod, _):
        """call MovePadAfterLayoutTransform func."""
        return MovePadAfterLayoutTransform().visit(func)


class RemoveDropout(ExprMutator):
    """Removes all nn.dropout from an expr."""

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        if visit.index != 0:
            return visit
        if (
            isinstance(visit.tuple_value, Call)
            and visit.tuple_value.op.name == "nn.dropout"
            and visit.index == 0
        ):
            # skip nn.dropout call and return arg0 instead
            return visit.tuple_value.args[0]
        return visit


@relay.transform.function_pass(opt_level=0)
class MrvlRemoveDropoutPass:
    """Removes Dropouts."""

    def transform_function(self, func, mod, _):
        """call RemoveDropout func."""
        return RemoveDropout().visit(func)


class RemoveCopy(ExprMutator):
    """
    Delete Copy expression
    """

    def visit_call(self, call):
        visit = super().visit_call(call)
        if hasattr(visit.op, "name") and visit.op.name in ["copy"]:
            return visit.args[0]
        return visit


@relay.transform.function_pass(opt_level=0)
class MrvlRemoveCopyPass:
    """Removes Copy."""

    def transform_function(self, func, mod, _):
        """call RemoveCopy func."""
        return RemoveCopy().visit(func)
