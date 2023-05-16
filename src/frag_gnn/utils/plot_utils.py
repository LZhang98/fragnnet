import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image
import io
from rdkit.Chem.Draw import rdMolDraw2D
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


def get_base64_mol_image(mol, highlight:list[int] = None, size:tuple(int,int) = None):
	"""draw a mol to base64 image 200x200
	Args:
		mol (_type_): rdkit mol object
		highlight (list, optional): idx list of highlighting atoms. Defaults to [].
		size tuple(int,int): size of image, default 200 by 200

	Returns:
		_type_: _description_
	"""
	if size is None:
		size = (200,200)
	if highlight is None:
		highlight = []
	drawer = rdMolDraw2D.MolDraw2DCairo(size[0],size[1])
	drawer.DrawMolecule(mol, highlightAtoms = highlight)
	drawer.FinishDrawing()
	img_str = base64.b64encode(drawer.GetDrawingText()).decode('utf8')
	return img_str