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


def get_mol_image(mol, highlight:list[int] = None, size:tuple(int,int) = None,\
		  options = None, as_base_64_str = False, mol_template_coords = None):
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

	if mol_template_coords is not None:
		try:
			GenerateDepictionMatching2DStructure(mol , mol_template_coords)
		except ValueError:
			pass

	if options == None:
		options = MolDrawOptions()
		options.bondLineWidth = 6
		options.minFontSize = 36
		options.maxFontSize = 128
		options.noCarbonSymbols = True
		options.backgroundColour = None

	if as_base_64_str:
		drawer = rdMolDraw2D.MolDraw2DCairo(size[0],size[1])
		drawer.DrawMolecule(mol, highlightAtoms = highlight)
		drawer.SetDrawOptions(options)
		drawer.FinishDrawing()
		img_str = base64.b64encode(drawer.GetDrawingText()).decode('utf8')
		return img_str
	else:
		img = MolToImage(mol, size=size, fitImage=False, options = options)
		return img

# https://stackoverflow.com/questions/22667224/matplotlib-get-text-bounding-box-independent-of-backend/22689498
def find_renderer(fig):
    if hasattr(fig.canvas, "get_renderer"):
        # Some backends, such as TkAgg, have the get_renderer method, which
        # makes this easy.
        renderer = fig.canvas.get_renderer()
    else:
        # Other backends do not have the get_renderer method, so we have a work
        # around to find the renderer.  Print the figure to a temporary file
        # object, and then grab the renderer that was used.
        # (I stole this trick from the matplotlib backend_bases.py
        # print_figure() method.)
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return (renderer)
