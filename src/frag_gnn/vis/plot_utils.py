import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import io
from rdkit.Chem.Draw import rdMolDraw2D, MolDrawOptions, MolToImage
from rdkit.Chem.AllChem import GenerateDepictionMatching2DStructure
import base64

EPS = np.finfo(np.float32).eps

def fig_to_data(fig,**kwargs):

	buf = io.BytesIO()
	fig.savefig(buf,**kwargs)
	buf.seek(0)
	image = Image.open(buf)
	return image

def plot_histogram(values,title,log=False,bins=10):

	fig,ax = plt.subplots()
	ax.hist(values,bins=bins)
	ax.set_title(title)
	ax.set_yscale("log" if log else "linear")
	data = fig_to_data(fig)
	plt.close("all")
	return data