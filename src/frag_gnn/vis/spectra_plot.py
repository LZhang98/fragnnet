from math import ceil
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from rdkit.Chem import AllChem


from frag_gnn.utils.standardization import Standardization
from frag_gnn.spectrum import Spectrum
from frag_gnn.utils.spec_utils import create_mol_by_inchi_or_smiles
from frag_gnn.metrics import Comparators

from frag_gnn.vis.common import *
from math import ceil

from copy import deepcopy


# FuncFormatter can be used as a decorator
@mticker.FuncFormatter
def spectra_major_formatter(x, pos):
    x = -x if x < 0 else x
    return "%d" % x + '%'

class SpectraVis(object):
    _annotation_top_kws = {
        'horizontalalignment': 'right',
        'verticalalignment': 'center', 'rotation': 90,
        'rotation_mode': 'anchor'}

    _annotation_bottom_kws = {
        'horizontalalignment': 'left',
        'verticalalignment': 'center', 'rotation': 90,
        'rotation_mode': 'anchor'}

    _default_height = 130
    _default_summary_font_size = 20
    _default_info_block_font_size = 24
    _default_info_block_title_font_size = 36
    _default_title_font_size = 24
    _default_tick_font_size = 16
    _default_annotation_font_size = 24
    _default_legends_font_size = 20
    _default_info_block_margin = 600
    _default_mol_image_size = (600, 300)
    _default_ax_width = 12
    _default_ax_height = 3
    _default_info_block_height = 1
    _default_peak_width = 2.0
    _default_font_family = 'sans-serif'
    _default_fonts = ['Arial']
    _default_bar_alpha = 1.0

    _colors = {'default': '#212121', 'text': '#212121', 'metric_background': '#0072B2'}
    _peak_colors = ['#0077BB','#33BBEE','#009988','#EE7733','#CC3311','#EE3377','#BBBBBB']
                    #['#388E3C', '#0072B2', '#E69F00', '#D55E00', "#1982C4", "#FF595E", "#8AC926",
                   # "#6A4C93", "#0EAD69", "#8C8B88", "#F2BB9B"]

    _baseline_colors = ["#56B4E970", '#009E7370']
    _cfm_colors = ['#52B4FA', '#E63434']

    _indicators = {'up': '▲', 'down': '▼'}
    _default_dpi = 128

    @classmethod
    def plot_mol_from_smiles_or_inchi(cls, chem_id_str: str):
        '''
        Method draw molecule by Smiles or Inchi 
        retuin an PIL image
        '''
        fig = plt.figure(figsize=(cls._default_ax_width, cls._default_ax_height))
        ax = fig.gca()
        cls._plot_info_block_axes(ax, chem_id_str)
        return fig

    '''
    Method to plot a spectrum
    '''

    @classmethod
    def plot_mirrored_spectrum(cls, top_spectrum: Spectrum, bottom_spectrum: Spectrum, text_type: str):
        fig = plt.figure(figsize=(cls._default_ax_width, cls._default_ax_height))
        ax = fig.gca()

        cls._plot_spectrum_axes(
            ax, top_spectrum, bottom_spectrum, spectrum_plot_settings = {'text_type':text_type})
        return fig

    '''
    Method to draw peaks
    very low peak will be focus to draw with 1.0% height
    '''

    @classmethod
    def _draw_peaks(cls, ax, peaks, max_intensity, label,
                    peak_width,
                    inverted=False,
                    peak_color=None,
                    show_annotation = False):

        for idx, peak in enumerate(peaks):
            mz = peak.mz
            # doing so very low peak will show up
            display_peak_intensity = max(1.0, peak.intensity / max_intensity * 100.0)
            annotation_pos = display_peak_intensity + 5
            if inverted:
                display_peak_intensity *= -1
                annotation_pos *= -1

            if peak_color is None:
                if inverted:
                    peak_color = cls._peak_colors[1]
                else:
                    peak_color = cls._peak_colors[0]

            # Draw peak
            #peak_width = cls._default_peak_width * 
            if idx == 0:
                ax.bar(mz, display_peak_intensity, alpha = cls._default_bar_alpha, width=peak_width, color=peak_color, label=label)
            else:
                ax.bar(mz, display_peak_intensity, alpha = cls._default_bar_alpha, width=peak_width, color=peak_color)

            # Draw Annotation as Text
            
            if peak.annotation != Standardization.NA and show_annotation:
                annotation_str = str(peak.annotation)
                ax.text(mz, annotation_pos, annotation_str, 
                    fontsize = cls._default_legends_font_size, color=peak_color, 
                    horizontalalignment='center', verticalalignment='center',  weight='bold')  

    '''
    Method draw a spectrum on matplot figure axis
    '''

    @classmethod
    def _plot_spectrum_axes(cls, ax, top_spectrum: Spectrum, 
                bottom_spectrum: Optional[Spectrum] = None, 
                metrics: Optional[dict] = None, 
                spectrum_plot_settings: Optional[dict] = None,):
        
        if spectrum_plot_settings == None:
            spectrum_plot_settings = {}
        text_type = spectrum_plot_settings.get('text_type')
        top_peak_color= spectrum_plot_settings.get('top_peak_color')
        bottom_peak_color= spectrum_plot_settings.get('bottom_peak_color')
        min_mz= spectrum_plot_settings.get('min_mz')
        max_mz= spectrum_plot_settings.get('max_mz')
        show_x_label= spectrum_plot_settings.get('show_x_label', True)
        show_y_label= spectrum_plot_settings.get('show_y_label', True)
        show_legend = spectrum_plot_settings.get('show_legend', True)
        show_annotation = spectrum_plot_settings.get('show_annotation', False)
        #print(show_annotation)
        summary_font_size = spectrum_plot_settings.get('summary_font_size', cls._default_summary_font_size)

        if show_x_label:
            ax.set_xlabel('m/z', fontsize=cls._default_annotation_font_size)
        if show_y_label:
            ax.set_ylabel('Relative Intensity', fontsize=cls._default_annotation_font_size)

        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        

        # set ranges
        mz_end = top_spectrum.max_mz + 50

        y_min = 0
        y_max = 100 if not show_annotation else cls._default_height

        if bottom_spectrum is not None:
            mz_end = max(mz_end, bottom_spectrum.max_mz + 50)
            y_min = -100 if not show_annotation else -cls._default_height
            ax.xaxis.set_label_coords(1.05, -y_min / (y_max - y_min))

        if max_mz is not None:
            mz_end = max_mz
        if min_mz is not None:
            mz_start = min_mz
        else:
            mz_start = 1
        
        peak_width =  0.5 * (max_mz/300)  #cls._default_peak_width 

        cls._draw_peaks(ax, top_spectrum.peaks,
                        top_spectrum.max_intensity, top_spectrum.get_spectrum_type_summary(), peak_width, inverted=False,
                        peak_color=top_peak_color, show_annotation = show_annotation)

        if bottom_spectrum is not None:
            cls._draw_peaks(ax, bottom_spectrum.peaks,
                            bottom_spectrum.max_intensity, bottom_spectrum.get_spectrum_type_summary(),peak_width,
                            inverted=True, peak_color=bottom_peak_color, show_annotation = show_annotation)


        # print(mz_end, max_mz)
        ax.set_xlim(left=mz_start, right=mz_end)
        ax.set_ylim(ymin=y_min, ymax=y_max)

        ax.yaxis.set_major_formatter(spectra_major_formatter)

        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both', labelsize='small')

        # add scores
        if metrics is not None:
            #summary_font_size = summary_font_size if summary_font_size is not None else cls._default_summary_font_size
            metrics_text = ""
            for idx, label in enumerate(metrics):
                if idx > 0:
                    metrics_text += '\n'

                if type(metrics[label]) is float:
                    metrics_text += "{}:{:.2}".format(label, metrics[label])
                elif type(metrics[label]) is int:
                    metrics_text += "{}:{}".format(label, metrics[label])
                elif type(metrics[label]) is str:
                    metrics_text += "{}:{}".format(label, metrics[label])

            ax.text(10, y_max - 5, metrics_text,
                    horizontalalignment='left',
                    verticalalignment='top',
                    # transform=ax.transAxes,
                    fontsize = summary_font_size,
                    linespacing=1.5,
                    # family= 'monospace',
                    # color=color_predicted,
                    bbox=dict(facecolor= '0.75', #bottom_peak_color,  # cls._colors['metric_background'],
                              # edgecolor= bottom_peak_color,
                              alpha=0.3, boxstyle='round,pad=0.3'))

        #print(text_type)
        if text_type is not None:
            #font_size = cls._default_summary_font_size

            top_spectrum_text = cls._get_spectrum_display_text(
                top_spectrum, text_type)
            ax.text(3, cls._default_height - summary_font_size, top_spectrum_text,
                    color=cls._colors['default'], fontsize=summary_font_size)

        if bottom_spectrum is not None and text_type is not None:
            bottom_spectrum_text = cls._get_spectrum_display_text(
                bottom_spectrum, text_type)

            ax.text(3, -(cls._default_height - summary_font_size), bottom_spectrum_text,
                    color=cls._colors['default'], fontsize=summary_font_size)

        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks[(-100 <= y_ticks) & (100 >= y_ticks)])
        ax.tick_params(axis='both', which='major', labelsize=cls._default_tick_font_size)
        # bbox_to_anchor=(1.00, 1.1)
        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2), fontsize=cls._default_legends_font_size, shadow=False,
                    ncol=1, frameon=False)

    @classmethod
    def _get_spectrum_display_text(cls, spectrum: Spectrum, text_type: str):
        text = ""
        if spectrum is None:
            return ""
        # if text_type == "spectrum_type_summary":
        #    text = spectrum.get_spectrum_type_summary()
        if text_type == "summary":
            text = spectrum.get_summary()
        return text

    @staticmethod
    def _get_compound_info_text(compound_name: str = '<n/a>', inchikey: str = '<n/a>'):
        return "Name: {} \nInchIKey: {} \n".format(compound_name, inchikey)

    @classmethod
    def _plot_info_block_axes(cls, ax, 
                            smiles_or_inchi: str, 
                            info_block_title: str = None,
                            info_text: str = None, 
                            metrics: Dict[str,float] = None,
                            info_block_title_font_size:int = None,
                            info_block_font_size:int = None,
                            mol_image_options = None,
                            mol_template_coords = None,
                            image_width_resize_factor = None):

        if info_block_title_font_size is None:
            info_block_title_font_size =  cls._default_info_block_title_font_size
        if info_block_font_size is None:
            info_block_font_size = cls._default_info_block_font_size


        fig = plt.gcf()

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_width, ax_height = bbox.width, bbox.height
        ax_width *= fig.dpi
        ax_height *= fig.dpi

        ax.set_xlim(0, ax_width)
        ax.set_ylim(ax_height, 0)
        
        y_offset = 16
        # Title
        if info_block_title is not None:
            text = ax.text(0,  y_offset, info_block_title,
                        horizontalalignment='left',
                        verticalalignment='top',
                        color=cls._colors['default'],
                        fontsize=info_block_title_font_size)
            text_bbox = text.get_window_extent(find_renderer(fig))
            y_offset += text_bbox.height * 1.1
        # Details
        if info_text is not None:
            text = ax.text(0,  y_offset, info_text,
                        horizontalalignment='left',
                        verticalalignment='top',
                        color=cls._colors['default'],
                        fontsize=info_block_font_size)
            text_bbox = text.get_window_extent(find_renderer(fig))
            y_offset += text_bbox.height * 1.5

        # show mertics
        mertics_bar_height = info_block_font_size * 1.5
        metrics_block_height = 0 
        if metrics is not None:
            mertics_idx = 0
            for metric_label in metrics:
                metric_bar_loc =  mertics_idx * mertics_bar_height + y_offset
                ax.barh(metric_bar_loc, metrics[metric_label] * ax_width, mertics_bar_height, alpha = 0.5)
                ax.text(0, metric_bar_loc, "{}:{:.2f}".format(metric_label, metrics[metric_label]), horizontalalignment='left',
                       verticalalignment='center', fontsize = info_block_font_size)

                mertics_idx += 1
                metrics_block_height += mertics_bar_height

        if info_block_title is None and info_text is None and metrics_block_height == 0:
            y_offset = 0
        
        if image_width_resize_factor is None:
            image_width_resize_factor = 1.0

        image_width = int(ax_width/ image_width_resize_factor) * 2 
        image_height = int(ax_height - metrics_block_height - y_offset) * 2
        #print("image_width: {}".format(image_width))
        #print("image_height: {}".format(image_height))
        #print(ax_height, metrics_block_height, y_offset)

        mol_image = cls.get_mol_img_from_chemid(smiles_or_inchi,
                                                size=(image_width, image_height),
                                                options= mol_image_options,
                                                mol_template_coords = mol_template_coords)
        # extent = (left, right, bottom, top) 
        extent =  (0, image_width/2, ax_height, ax_height - image_height/2)
        ax.imshow(mol_image, extent = extent, interpolation = 'lanczos')
        ax.axis('off')


    @classmethod
    def _init_figure_and_info_block(cls, num_row, num_col, smiles_or_inchi, info_text,
                                    width_scale=1,
                                    height_scale=1,
                                    height_ratios=None):
        # print(height_ratios)
        if smiles_or_inchi is not None:
            num_row += 1

        if height_ratios is None:
            height_ratios = [2] * num_row
            if smiles_or_inchi is not None:
                height_ratios[0] = 2

        # draw figures per ev level
        fig = plt.figure(figsize=(num_col * cls._default_ax_width * width_scale,
                                  num_row * cls._default_ax_height * height_scale), dpi=cls._default_dpi)
        #print(num_col *  cls._default_ax_width * width_scale,
        #                          num_row * cls._default_ax_height * height_scale)
        # print('_init_figure_and_info_block',num_row, num_col)
        # print(num_col, num_row, height_ratios)
        gs = GridSpec(ncols=num_col, nrows=num_row, figure=fig, height_ratios=height_ratios)

        # draw mol
        cur_row = 0
        if smiles_or_inchi is not None:
            ax = fig.add_subplot(gs[cur_row, 0:])
            cls._plot_info_block_axes(ax, smiles_or_inchi, info_text = info_text)
            cur_row += 1

        return fig, gs, cur_row

    @classmethod
    def plot_spectra(cls, spectra: List[Spectrum], smiles_or_inchi=None, info_text=None, show_annotation=False):
        """Method to plot a list of spectra

        Args:
            spectra (List[Spectrum]): [description]
            smiles_or_inchi ([type], optional): [description]. Defaults to None.
            info_text ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        num_row = len(spectra)
        num_col = 1
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text)

        if smiles_or_inchi is not None:
            num_row += 1

        max_mz = 0
        for spectrum in spectra:
            max_mz = max(spectrum.max_mz, max_mz)
        # add right margin
        max_mz += 50

        for spectrum in spectra:
            ax = fig.add_subplot(gs[cur_row, 0])
            spectrum_plot_settings = {
                        'text_type': "spectrum_type_summary",
                        'max_mz' : max_mz,
                        'show_annotation':show_annotation
            }
            cls._plot_spectrum_axes( ax, spectrum, spectrum_plot_settings = spectrum_plot_settings)
            cur_row += 1

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)
        return fig


    @classmethod
    def plot_mirrored_spectra(cls, groundtruth_spectra: List[Spectrum],
                                               predicted_spectra_list: List[List[Spectrum]],
                                               predicted_spectra_smiles_or_inchi_list: List[str] = None,
                                               predicted_spectra_info_text: List[str] = None,
                                               spectra_metrics_list: List[List[Dict]] = None,
                                               smiles_or_inchi: str = None,
                                               info_text: str = None,
                                               show_mol_image_for_each_col=False,
                                               show_annotation: bool = False):

        """ Method to create figure to compare many predicted spectra vs the same ground truth side by side

        Args:
            groundtruth_spectra (List[Spectrum]): [description]
            predicted_spectra_list (List[List[Spectrum]]): [description]
            predicted_spectra_smiles_or_inchi_list (List[str], optional): [description]. Defaults to None.
            predicted_spectra_info_text (List[str], optional): [description]. Defaults to None.
            spectra_metrics_list (List[List[Dict]], optional): [description]. Defaults to None.
            smiles_or_inchi (str, optional): [description]. Defaults to None.
            info_text (str, optional): [description]. Defaults to None.
            show_mol_image_for_each_col (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        num_row = len(groundtruth_spectra)
        num_col = len(predicted_spectra_list)
        # height_ratios = [2] * num_row
        max_mz = 0
        for spectra in predicted_spectra_list + [groundtruth_spectra]:
            for spectrum in spectra:
                #print(spectrum)
                #print(spectrum.max_mz)
                max_mz = max(spectrum.max_mz, max_mz)

        max_mz += 50

        # init fig
        has_predicted_smiles = predicted_spectra_smiles_or_inchi_list is not None
        plot_predicted_mol_block = show_mol_image_for_each_col and has_predicted_smiles
        # print(num_row, num_col)
        if plot_predicted_mol_block:
            num_row += 1

        has_metrics = spectra_metrics_list is not None
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text,
                                                           height_scale=2.0)
        display_mol_template = None
        if smiles_or_inchi is not None and plot_predicted_mol_block:
            # create a 2d coords template
            display_mol_template  = create_mol_by_inchi_or_smiles(smiles_or_inchi)            
            AllChem.Compute2DCoords(display_mol_template)
        if smiles_or_inchi is None and plot_predicted_mol_block:
            # create a 2d coords template
            display_mol_template  = create_mol_by_inchi_or_smiles(predicted_spectra_smiles_or_inchi_list[0])            
            AllChem.Compute2DCoords(display_mol_template)

        for col_idx, spectra in enumerate(predicted_spectra_list):
            if plot_predicted_mol_block:
                ax = fig.add_subplot(gs[cur_row, col_idx])
                text_info = predicted_spectra_info_text[col_idx] if predicted_spectra_info_text is not None else ''
                cls._plot_info_block_axes(ax, predicted_spectra_smiles_or_inchi_list[col_idx], info_text = text_info,      
                     info_block_font_size = cls._default_info_block_font_size,
                     mol_template_coords = display_mol_template,
                )

            for gt_spectrum_idx in range(len(groundtruth_spectra)):
                row_idx = gt_spectrum_idx + cur_row
                if show_mol_image_for_each_col:
                    row_idx += 1
                ax = fig.add_subplot(gs[row_idx, col_idx])

                top_spectrum = groundtruth_spectra[gt_spectrum_idx]
                bottom_spectrum = predicted_spectra_list[col_idx][gt_spectrum_idx]

                metrics = spectra_metrics_list[col_idx][gt_spectrum_idx] if has_metrics else None

                show_y_label = col_idx == 0
                show_x_label = col_idx == len(predicted_spectra_list) - 1

                spectrum_plot_settings = {
                        'top_peak_color' : '#52B4FA',
                        'bottom_peak_color' : '#E63434',
                        'max_mz' : max_mz,
                        'show_x_label':show_x_label,
                        'show_y_label':show_y_label,
                        'show_annotation':show_annotation
                    }

                cls._plot_spectrum_axes( ax, top_spectrum, bottom_spectrum, metrics, spectrum_plot_settings)

                pad = 5
                if col_idx == 0:
                    ax.annotate(top_spectrum.get_collision_energy_string(), xy=(0, 0.5),
                                xytext=(-ax.yaxis.labelpad - pad, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points',
                                size=cls._default_title_font_size, weight='bold', ha='right', va='center')

            plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=3.0)
        return fig
    
    @classmethod
    def plot_identification_results(cls, predicted_spectra_smiles_or_inchi_list: List[str] = None,
                              predicted_spectra_info_text: List[str] = None,
                              smiles_or_inchi: str = None,
                              info_text: str = None):
        """[summary]

        Args:
            predicted_spectra_smiles_or_inchi_list (List[str], optional): [description]. Defaults to None.
            predicted_spectra_info_text (List[str], optional): [description]. Defaults to None.
            smiles_or_inchi (str, optional): [description]. Defaults to None.
            info_text (str, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_predictions = len(predicted_spectra_smiles_or_inchi_list)
        num_col = 3 if num_predictions > 3 else num_predictions

        num_row = int(ceil(num_predictions / num_col))
        # print(num_row, num_col)

        height_ratios = [1] * num_row if smiles_or_inchi is None else [1] * (num_row + 1)
        fig, gs, cur_row = cls._init_figure_and_info_block(num_row, num_col, smiles_or_inchi, info_text,
                                                           height_scale=2.0, height_ratios=height_ratios)

        for idx, smiles_or_inchi in enumerate(predicted_spectra_smiles_or_inchi_list):
            # for row_idx in range(cur_row, cur_row + num_row):
            # print(idx//num_col + cur_row, idx % num_col)
            ax = fig.add_subplot(gs[idx // num_col + cur_row, idx % num_col])
            text_info = predicted_spectra_info_text[idx] if len(predicted_spectra_info_text) == num_predictions else ''
            cls._plot_info_block_axes(ax, smiles_or_inchi, info_text = text_info, info_block_font_size = cls._default_info_block_font_size)
        return fig

    
    @classmethod
    def plot_mirror_spectra_with_structure(cls, top_spectrum: Spectrum, bottom_spectrum: Spectrum, structure_names:List[str] = None, show_annotation  = False, use_cfm_color = False, single_mol_image = False, auto_compare = True):

        info_block_font_size = 16

        # get max mz 
        max_mz = max(top_spectrum.max_mz, bottom_spectrum.max_mz) + 50

        # init figure
        fig =plt.figure(figsize=(cls._default_ax_width * 2, cls._default_ax_height * 2), constrained_layout=True, dpi = cls._default_dpi)
        
        gs = GridSpec(ncols=4, nrows=2, figure=fig)


        #top_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)
        #bottom_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)

        display_mol_template = None
        if top_spectrum.smiles_or_inchi != Standardization.NA:
            display_mol_template  = create_mol_by_inchi_or_smiles(top_spectrum.smiles_or_inchi)
        elif bottom_spectrum.smiles_or_inchi != Standardization.NA:
            display_mol_template  = create_mol_by_inchi_or_smiles(bottom_spectrum.smiles_or_inchi)
        AllChem.Compute2DCoords(display_mol_template)

        if structure_names is None:
            structure_names = [None, None]

        if not single_mol_image:
            ax = fig.add_subplot(gs[0, 3])
            cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0], mol_template_coords = display_mol_template)

            ax = fig.add_subplot(gs[1, 3])
            cls._plot_info_block_axes(ax, bottom_spectrum.smiles_or_inchi, structure_names[1], mol_template_coords= display_mol_template)
        else:
            ax = fig.add_subplot(gs[0:2, 3])
            if top_spectrum.smiles_or_inchi != Standardization.NA: 
                cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0])
            else:
                cls._plot_info_block_axes(ax, bottom_spectrum.smiles_or_inchi, structure_names[0])


        
        show_y_label = True
        show_x_label = True

        ax = fig.add_subplot(gs[0:3, 0:3])
        if use_cfm_color:
            top_peak_color= cls._cfm_colors[0]
            bottom_peak_color= cls._cfm_colors[1]
        else:
            top_peak_color= cls._peak_colors[0]
            bottom_peak_color= cls._peak_colors[1]

        metrics = None
        if auto_compare:
            metrics = Comparators.get_metrics(top_spectrum, bottom_spectrum, verbose=False)

        spectrum_plot_settings = {
                        'top_peak_color' : top_peak_color,
                        'bottom_peak_color' : bottom_peak_color,
                        'min_mz' : -10,
                        'max_mz' : max_mz,
                        'show_x_label':show_x_label,
                        'show_y_label':show_y_label,
                        'show_legend':True,
                        'show_annotation':show_annotation,
                        'summary_font_size':info_block_font_size,
                        'text_type':None,
        }

        cls._plot_spectrum_axes( ax, top_spectrum, bottom_spectrum, metrics, spectrum_plot_settings)


        return fig

    @classmethod
    def plot_mirror_spectra_list_with_structure(cls, top_spectrum: Spectrum, bottom_spectrum_list: List[Spectrum], structure_names:List[str] = None, show_annotation  = False, use_cfm_color = False, single_mol_image = False):

        info_block_font_size = 16

        # get max mz 
        max_mz = top_spectrum.max_mz
        for bottom_spectrum in bottom_spectrum_list:
            max_mz = max(max_mz, bottom_spectrum.max_mz)
        max_mz += 50

        # get number of rows and columns
        num_row = len(bottom_spectrum_list)
        if num_row == 0:
            return
        #print('num_row',num_row)
        # init figure
        fig =plt.figure(figsize=(cls._default_ax_width * 2, cls._default_ax_height * 2 * num_row), constrained_layout=True, dpi = cls._default_dpi)
        
        gs = GridSpec(ncols=4, nrows=2 *num_row, figure=fig)
        #print(gs)
        #top_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)
        #bottom_spectrum.annotate_with_mz(intensity_threshold = annotate_intensity)

        for idx,bottom_spectrum in enumerate(bottom_spectrum_list):
            display_mol_template = None
            if top_spectrum.smiles_or_inchi != Standardization.NA:
                display_mol_template  = create_mol_by_inchi_or_smiles(top_spectrum.smiles_or_inchi)
            elif bottom_spectrum.smiles_or_inchi != Standardization.NA:
                display_mol_template  = create_mol_by_inchi_or_smiles(bottom_spectrum.smiles_or_inchi)
            AllChem.Compute2DCoords(display_mol_template)

            if structure_names is None:
                structure_names = [None, None]

            if not single_mol_image:
                ax = fig.add_subplot(gs[idx * 2, 3])
                cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0], mol_template_coords = display_mol_template)

                ax = fig.add_subplot(gs[idx * 2 + 1, 3])
                cls._plot_info_block_axes(ax, bottom_spectrum.smiles_or_inchi, structure_names[1], mol_template_coords= display_mol_template)
            else:
                ax = fig.add_subplot(gs[idx * 2: idx * 2 + 2, 3])
                if top_spectrum.smiles_or_inchi != Standardization.NA: 
                    cls._plot_info_block_axes(ax, top_spectrum.smiles_or_inchi, structure_names[0])
                else:
                    cls._plot_info_block_axes(ax, bottom_spectrum.smiles_or_inchi, structure_names[0])

            
            show_y_label = True
            show_x_label = True

            ax = fig.add_subplot(gs[idx * 2: idx * 2 + 2, 0:3])
            if use_cfm_color:
                top_peak_color= cls._cfm_colors[0]
                bottom_peak_color= cls._cfm_colors[1]
            else:
                top_peak_color= cls._peak_colors[0]
                bottom_peak_color= cls._peak_colors[1]

            metrics = Comparators.get_metrics(top_spectrum, bottom_spectrum, verbose=False)

            spectrum_plot_settings = {
                            'top_peak_color' : top_peak_color,
                            'bottom_peak_color' : bottom_peak_color,
                            'min_mz' : -10,
                            'max_mz' : max_mz,
                            'show_x_label':show_x_label,
                            'show_y_label':show_y_label,
                            'show_legend':True,
                            'show_annotation':show_annotation,
                            'summary_font_size':info_block_font_size,
                            'text_type':None,
            }

            cls._plot_spectrum_axes( ax, top_spectrum, bottom_spectrum, metrics, spectrum_plot_settings)


        return fig