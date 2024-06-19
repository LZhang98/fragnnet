
import argparse

from frag_gnn.utils.nn_utils import decompile_jit_ckpt

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_fp",type=str, default=None)
	parser.add_argument("--np_ckpt_fp",type=str, default=None)
	args = parser.parse_args()
	
	print(f"> input ckpt {args.ckpt_fp}")
	print(f"> output ckpt {args.np_ckpt_fp}")
	decompile_jit_ckpt(args.ckpt_fp, args.np_ckpt_fp)