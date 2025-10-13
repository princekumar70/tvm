/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/mrvl/mrvl_sw_runtime_lib.cc
 * \brief Runtime library for Marvell Software Simulator.
 */

#include "mrvl_sw_runtime_lib.h"

#include <assert.h>
#include <sys/stat.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <fstream>
#include <vector>

#include "mrvl_base64.h"

using namespace tvm::runtime;

template <typename T>
static void NDArrayToFile(const tvm::runtime::NDArray& arr, std::ostream& os) {
  int ndim = arr->ndim;
  int tot_dim = 1;
  for (int i = 0; i < ndim; i++) {
    tot_dim *= arr->shape[i];
  }
  T* data_ptr = reinterpret_cast<T*>(arr->data);
  os << "\t\t[";
  os << std::endl;
  for (int i = 0; i < tot_dim; i++) {
    os << "\t\t\t" << std::setprecision(10) << data_ptr[i] << (i != tot_dim - 1 ? "," : "");
    os << std::endl;
  }
  os << "\t\t]";
}

static void WriteBinToDisk(const std::string& bin_file, const std::string& bin_code) {
  auto length = tvm::runtime::contrib::mrvl::b64strlen(bin_code);
  std::vector<unsigned char> byte_array(length);
  tvm::runtime::contrib::mrvl::b64decode(bin_code, byte_array.data());
  std::ofstream file_out;
  file_out.open(bin_file, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  for (auto byte : byte_array) file_out << byte;
}

static void ReadInputsAndGenerateInputBin(TVMArgs args, const std::string& input_json,
                                          const std::string& input_bin,
                                          const std::string& bin_directory, size_t num_inputs) {
  std::ofstream file_out;
  file_out.open(input_json, std::ios_base::out | std::ios_base::trunc);
  file_out << "{" << std::endl;
  file_out << R"(    "inputs": [)" << std::endl;
  for (size_t i = 0; i < num_inputs; ++i) {
    const DLTensor* tensor;
    if (args[i].IsObjectRef<NDArray>()) {
      NDArray arr = args[i];
      tensor = arr.operator->();
    } else {
      tensor = args[i].operator DLTensor*();
    }
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < tensor->ndim; i++) {
      shape.push_back(tensor->shape[i]);
    }
    NDArray arr = NDArray::Empty(shape, tensor->dtype, tensor->device);
    arr.CopyFrom(tensor);
    NDArrayToFile<float>(arr, file_out);
    if (i != num_inputs - 1) {
      file_out << std::endl << "\t," << std::endl;
    }
  }
  file_out << std::endl << "\t]" << std::endl;
  file_out << "}" << std::endl;

  const auto* json_to_bin = tvm::runtime::Registry::Get("tvm.mrvl.JsonToBin");
  (*json_to_bin)(input_json, input_bin);
}

static void RunInferenceOnMlModel(const std::string& symbol_name, const std::string& bin_directory,
                                  const std::string& bin_file, const std::string& input_bin,
                                  const std::string& out_bin_prefix) {
  auto command =
      bin_directory + "/mlModel " + "-m " + bin_file + " -d " + input_bin + " -o " + out_bin_prefix;
  std::string sim_directory = "mrvl_sw_sim_" + symbol_name;
  const auto* run_sim = tvm::runtime::Registry::Get("tvm.mrvl.RunSim");
  (*run_sim)(command, sim_directory);
}

static void RunInferenceOnFsim(const std::string& symbol_name, const std::string& bin_file,
                               const std::string& model_name, const std::string& working_directory,
                               const std::string& quantization_type, const std::string& input_bin) {
  std::string fsim_run = "fsim_" + model_name;
  mkdir(fsim_run.c_str(), 0777);
  int r = chdir(fsim_run.c_str());
  if (r < 0) {
    perror("chdir");
    exit(1);
  }

  std::string cmake_command = "cmake -DCMAKE_C_COMPILER_FORCED=1 -DCMAKE_CXX_COMPILER_WORKS=1 ";
  cmake_command += "-DCMAKE_BUILD_TYPE=Release ";
  cmake_command += "-DMAIN_BIN_FOLDER=" + working_directory + "/emu_" + model_name;
  const char* path = std::getenv("MRVL_FUNCTIONAL_SIMULATOR");
  if (path == nullptr) {
    ICHECK(false) << "Please specify the path to Marvell tools "
                     "by setting MRVL_FUNCTIONAL_SIMULATOR in the environment.";
  }
  cmake_command += " -S" + std::string(path);
  std ::cout << "cmake for fsim: " << cmake_command << "\n";
  int ret_val = system(cmake_command.c_str());
  ICHECK(ret_val == 0) << "Marvell-Compiler-ERROR-Internal::system() - cmake call failed\n";

  std::string make_command = "make -j";
  std ::cout << "make for fsim: " << make_command << "\n";
  ret_val = system(make_command.c_str());
  ICHECK(ret_val == 0) << "Marvell-Compiler-ERROR-Internal::system() - make call failed\n";

  r = chdir("../");
  if (r < 0) {
    perror("chdir");
    exit(1);
  }

  std::string fsim_command = "./" + fsim_run + "/fsim " + input_bin + " ";
  fsim_command += working_directory + "/emu_" + model_name;
  fsim_command += " " + quantization_type;
  std ::cout << "run fsim: " << fsim_command << "\n";
  ret_val = system(fsim_command.c_str());
  ICHECK(ret_val == 0) << "Marvell-Compiler-ERROR-Internal::system() - fsim call failed\n";
}

enum DTypeEnum { FLOAT32, INT64, UINT64, UNKNOWN };

DTypeEnum GetDTypeEnum(const DLDataType& dtype) {
  unsigned int bits_value = static_cast<unsigned int>(dtype.bits);
  if (static_cast<int>(dtype.code) == 2 && bits_value == 32) {
    return FLOAT32;
  } else if (static_cast<int>(dtype.code) == 0 && bits_value == 64) {
    return INT64;
  } else if (static_cast<int>(dtype.code) == 1 && bits_value == 64) {
    return UINT64;
  } else {
    return UNKNOWN;
  }
}

template <typename T>
std::vector<T> convertVector(const std::vector<float>& input) {
  std::vector<T> output;
  output.reserve(input.size());
  for (float value : input) {
    output.push_back(static_cast<T>(value));
  }
  return output;
}

template <typename T>
void ReadData(std::ifstream& fin, size_t tot_dim, NDArray* arr, const String& run_mode) {
  if (run_mode == "fsim") {
    // Fsim output is always dequantized to FP32
    std::vector<float> fsim_data(tot_dim);
    fin.read(reinterpret_cast<char*>(fsim_data.data()), tot_dim * sizeof(float));
    ICHECK(fin.gcount() == static_cast<std::streamsize>(tot_dim * sizeof(float)))
        << "Output data size mismatch";
    std::vector<T> data = convertVector<T>(fsim_data);
    arr->CopyFromBytes(data.data(), tot_dim * sizeof(T));
  } else {
    std::vector<T> data(tot_dim);
    fin.read(reinterpret_cast<char*>(data.data()), tot_dim * sizeof(T));
    ICHECK(fin.gcount() == static_cast<std::streamsize>(tot_dim * sizeof(T)))
        << "Output data size mismatch";
    arr->CopyFromBytes(data.data(), tot_dim * sizeof(T));
  }
}

static void ReadOutputsAndUpdateRuntime(TVMArgs args, size_t num_inputs,
                                        const std::string& out_bin_prefix,
                                        const String& run_mode = "sim") {
  for (int out = num_inputs; out < args.size(); out++) {
    const DLTensor* outTensor;
    if (args[out].IsObjectRef<NDArray>()) {
      NDArray arr = args[out];
      outTensor = arr.operator->();
    } else {
      outTensor = args[out].operator DLTensor*();
    }
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < outTensor->ndim; i++) {
      shape.push_back(outTensor->shape[i]);
    }
    NDArray arr = NDArray::Empty(shape, outTensor->dtype, outTensor->device);
    int ndim = arr->ndim;
    int tot_dim = 1;
    for (int i = 0; i < ndim; i++) {
      tot_dim *= arr->shape[i];
    }
    String outbin = out_bin_prefix + "-" + std::to_string(out - num_inputs) + ".bin";
    std::ifstream fin(outbin, std::ios::binary);
    ICHECK(fin.is_open()) << "Cannot open file: " << outbin;

    DTypeEnum dtypeEnum = GetDTypeEnum(arr->dtype);
    switch (dtypeEnum) {
      case FLOAT32: {
        ReadData<float>(fin, tot_dim, &arr, run_mode);
        break;
      }
      case INT64: {
        ReadData<int64_t>(fin, tot_dim, &arr, run_mode);
        break;
      }
      case UINT64: {
        ReadData<uint64_t>(fin, tot_dim, &arr, run_mode);
        break;
      }
      default: {
        throw std::runtime_error("Unsupported data type");
      }
    }
    arr.CopyTo(const_cast<DLTensor*>(outTensor));
  }
}

static void CleanUp(TVMArgs args, const std::string& bin_file, const std::string& input_json,
                    const std::string& input_bin, const std::string& out_bin_prefix,
                    size_t num_outputs) {
  const auto* clean_up = tvm::runtime::Registry::Get("tvm.mrvl.CleanUpSim");
  (*clean_up)(bin_file, input_json, input_bin, out_bin_prefix, num_outputs);
}

void tvm::runtime::contrib::mrvl::RunMarvellSimulator(TVMArgs args, const std::string& symbol_name,
                                                      const std::string& bin_code,
                                                      size_t num_inputs, size_t num_outputs) {
  // check $PATH for the presence of MRVL dependent tools/scripts
  std::string file_name("mlModel");
  const auto* search_path = tvm::runtime::Registry::Get("tvm.mrvl.SearchPath");
  std::string tools_directory = (*search_path)(file_name);
  if (tools_directory.empty()) {
    ICHECK(false) << "mlModel simulator not found! Please specify the path to Marvell "
                     "tools by adding it to $PATH.";
  }

  const auto* temp_dir = tvm::runtime::Registry::Get("tvm.mrvl.TempDir");
  std::string working_directory = (*temp_dir)();
  auto bin_file = working_directory + "/" + symbol_name + ".bin";
  auto input_json = working_directory + "/indata.json";
  auto input_bin = working_directory + "/input.bin";
  auto out_bin_prefix = working_directory + "/mrvl_sim_out";

  WriteBinToDisk(bin_file, bin_code);
  ReadInputsAndGenerateInputBin(args, input_json, input_bin, tools_directory, num_inputs);
  RunInferenceOnMlModel(symbol_name, tools_directory, bin_file, input_bin, out_bin_prefix);
  ReadOutputsAndUpdateRuntime(args, num_inputs, out_bin_prefix);
  CleanUp(args, bin_file, input_json, input_bin, out_bin_prefix, num_outputs);
}

void tvm::runtime::contrib::mrvl::RunMarvellFsim(TVMArgs args, const std::string& symbol_name,
                                                 const std::string& bin_code,
                                                 const String& model_name,
                                                 const String& working_directory,
                                                 const String& quantization_type, size_t num_inputs,
                                                 size_t num_outputs) {
  // check $PATH for the presence of MRVL dependent tools/scripts
  std::string file_name("mlModel");
  const auto* search_path = tvm::runtime::Registry::Get("tvm.mrvl.SearchPath");
  std::string tools_directory = (*search_path)(file_name);
  if (tools_directory.empty()) {
    ICHECK(false) << "mlModel simulator not found! Please specify the path to Marvell "
                     "tools by adding it to $PATH.";
  }
  auto bin_file = working_directory + "/" + symbol_name + ".bin";
  auto input_json = working_directory + "/indata.json";
  auto input_bin = working_directory + "/input.bin";
  auto out_bin_prefix = working_directory + "/out";

  WriteBinToDisk(bin_file, bin_code);
  ReadInputsAndGenerateInputBin(args, input_json, input_bin, tools_directory, num_inputs);
  RunInferenceOnFsim(symbol_name, bin_file, model_name, working_directory, quantization_type,
                     input_bin);
  ReadOutputsAndUpdateRuntime(args, num_inputs, out_bin_prefix, "fsim");
  CleanUp(args, bin_file, input_json, input_bin, out_bin_prefix, num_outputs);
}
