import subprocess
import sys
import re
import os
import ctypes
import platform

def detect_nvcc():
	try:
		output = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
		match = re.search(r"release (\d+\.\d+)", output)
		if match:
			return match.group(1).replace(".", "")
	except Exception:
		return None

def detect_libcuda():
	try:
		if platform.system() == "Windows":
			ctypes.windll.LoadLibrary("nvcuda.dll")
		else:
			ctypes.cdll.LoadLibrary("libcuda.so")
		print("libcuda found: assuming CUDA runtime is available")
		# Default to 118 if we can't detect exact version
		return "118"
	except Exception:
		return None

def detect_cuda_version(force_cuda=None):
	if force_cuda:
		return force_cuda
	return detect_nvcc() or detect_libcuda()


def pip_install(packages, index_url=None):
	cmd = [sys.executable, "-m", "pip", "install"] + packages
	if index_url:
		cmd += ["--extra-index-url", index_url]
	subprocess.check_call(cmd)

def install_all(cuda_version):
	if cuda_version == "121":
		print("Installing PyTorch 2.1.2 with CUDA 12.1...")
		pip_install(
			["torch==2.1.2"],
			index_url="https://download.pytorch.org/whl/cu121"
		)
  		# pip install "pytorch_lightning==2.1.2"
		pip_install(["pytorch_lightning==2.1.2"])
  		# pip install torch_geometric
		pip_install(
			["torch_geometric==2.4.0"],
		)
  		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
		pip_install(
			["torch_scatter"],
			index_url="https://data.pyg.org/whl/torch-2.1.2+cu121.html"
		)
  		# pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
		pip_install(
			["dgl"],
			index_url="https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html"			
		)
	elif cuda_version == "118":
		print("Installing PyTorch 2.1.2 with CUDA 11.8...")
		pip_install(
			["torch==2.1.2"],
			index_url="https://download.pytorch.org/whl/cu118"
		)
		# pip install "pytorch_lightning==2.1.2"
		pip_install(["pytorch_lightning==2.1.2"])
		# pip install torch_geometric
		pip_install(
			["torch_geometric==2.4.0"],
		)
		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
		pip_install(
			["torch_scatter"],
			index_url="https://data.pyg.org/whl/torch-2.1.2+cu118.html"
		)
		# pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
		pip_install(
			["dgl"],
			index_url="https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html"			
		)
	else:
		print("Installing CPU version of PyTorch 2.1.2...")
		pip_install(
			["torch==2.1.2"],
			index_url="https://download.pytorch.org/whl/cpu"
		)
		pip_install(["pytorch_lightning==2.1.2", "torch_geometric==2.4.0", "dgl"])
		# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
		pip_install(
			["torch_scatter"],
			index_url="https://data.pyg.org/whl/torch-2.1.2+cpu.html"
		)		

def main():
	cuda_version = detect_cuda_version()
	print(f"Detected CUDA version: {cuda_version}" if cuda_version else "CUDA not detected.")
	install_all(cuda_version)

if __name__ == "__main__":
	main()