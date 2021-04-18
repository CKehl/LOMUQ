import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.category as _  # <-registers a category unit converter
import matplotlib.dates as _  # <-registers a date unit converter
import matplotlib.docstring as docstring
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import numpy as np

class TransparentCircles(object):
    _markerstyle = None
    _markerobj = None
    _markersizes = None
    _edgecolors = None
    _facecolors = None
    _linewidths = None
    _cmap = None
    _norm = None
    _dpi = None
    _zorder = None
    _transform = None
    _markerpath = None  # the actual glyph
    _offsets = None
    _values = None
    _collection = None
    _uniform_alpha =  None
    _npts = None


    def __init__(self,x,y, s=None, linewidths=None, values=None, cmap=None, norm=None, colors=None, edgecolors=None, alphas=1.0, zorder=None, transform=mtransforms.IdentityTransform(), dpi=72):
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert x.shape == y.shape
        self._npts = x.shape[0]
        ref_colour_nitems = None
        # == colour composition or colour mapping == #
        if values is None or cmap is None:
            # use colors as facecolors - check sizes
            if colors is not None:
                assert isinstance(colors, np.ndarray)
                if len(colors.shape) > 1:
                    assert x.shape[0] == colors.shape[0]
                color_dims = colors.shape[0] if len(colors.shape) == 1 else colors.shape[1]
                if color_dims == 3:  # RGB - check and merge with alpha(s)
                    if len(colors.shape) > 1:
                        alpha_array = alphas if isinstance(alphas, np.ndarray) else np.ones(colors.shape[0], dtype=np.float32) * alphas
                        colours = np.column_stack([colors, alpha_array])
                    else:
                        colours = np.concatenate([colors, alphas])
                else:
                    assert color_dims == 4
                    colours = colors
                ref_colour_nitems = -1
                if len(colours.shape) > 1:
                    ref_colour_nitems = colours.shape[0]
                self._facecolors = colours
            if edgecolors is not None:
                assert isinstance(edgecolors, np.ndarray)
                if len(edgecolors.shape) > 1:
                    assert x.shape[0] == edgecolors.shape[0]
                color_dims = edgecolors.shape[0] if len(edgecolors.shape) == 1 else edgecolors.shape[1]
                if color_dims == 3:  # RGB - check and merge with alpha(s)
                    if len(edgecolors.shape) > 1:
                        ecolours = np.column_stack([edgecolors, np.ones(colors.shape[0], dtype=np.float32)])
                    else:
                        ecolours = np.concatenate([edgecolors, alphas])
                else:
                    assert color_dims == 4
                    ecolours = edgecolors
                self._edgecolors = ecolours
                assert self._facecolors.shape == self._edgecolors.shape
            else:
                raise ValueError("No color information available for scatter plot circles.")
        else:  # use the colour mapping - no face or edge colors
            self._facecolors = None
            self._edgecolors = None
            self._cmap = cmap
            if norm is not None and not isinstance(norm, mcolors.Normalize):
                msg = "'norm' must be an instance of 'mcolors.Normalize'"
                raise ValueError(msg)
            self._norm = norm
            self._values = values
            assert x.shape[0] == values.shape[0]
            ref_colour_nitems = self._values.shape[0]
        if alphas not in [None, False] and type(alphas) not in [list, np.ndarray]:
            self._uniform_alpha = alphas

        self._linewidths = linewidths
        self._create_path_collection()
        if ref_colour_nitems > 0 and (isinstance(self._linewidths, np.ndarray) or type(self._linewidths) in [list, ]):
            assert self._linewidths.shape[0] == ref_colour_nitems
        else:
            assert type(self._linewidths) in [np.float, np.float32, np.float64, float]
        self._markersizes = s if s is not None else 1.0
        if ref_colour_nitems > 0 and (isinstance(self._markersizes, np.ndarray) or type(self._markersizes) in [list, ]):
            assert self._markersizes.shape[0] == ref_colour_nitems
        else:
            assert type(self._markersizes) in [np.float, np.float32, np.float64, float]
            self._markersizes = np.ones(self._npts, dtype=np.float32) * self._markersizes
        self._offsets = np.column_stack([x, y])
        self._zorder = zorder
        self._transform = transform
        self._dpi = dpi

        self._collection = mcoll.PathCollection(
            (self._markerpath,), self._markersizes,
            cmap=self._cmap,
            norm=self._norm,
            facecolors=self._facecolors,
            edgecolors=self._edgecolors,
            linewidths=self._linewidths,
            offsets=self._offsets,
            transOffset=self._transform,
            alpha=self._uniform_alpha,
            zorder=self._zorder
        )

        if self._facecolors is None and self._values is not None and self._cmap is not None and self._norm is not None:
            self._collection.set_array(np.asarray(self._values))
            # self._collection.set_cmap(self._cmap)
            # self._collection.set_norm(self._norm)

    def _create_path_collection(self):
        # load default marker from rcParams
        if self._markerstyle is None:
            self._markerstyle = mpl.rcParams['scatter.marker']

        if self._markerobj is None:
            if isinstance(self._markerstyle, mmarkers.MarkerStyle):
                self._markerobj = self._markerstyle
            else:
                self._markerobj = mmarkers.MarkerStyle(self._markerstyle)

        self.markerpath = self._markerobj.get_path().transformed(
            self._markerobj.get_transform())
        if not self._markerobj.is_filled():
            self._edgecolors = 'face'
            if self._edgecolors is None:
                self.edgecolors = 'face'
            if self._linewidths is None:
                self._linewidths = mpl.rcParams['lines.linewidth']

    def add_point(self, x, y):
        self.add_points([x,], [y,])

    def add_points(self, xs, ys):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        self.set_offsets(np.vstack((self._offsets, np.column_stack([xs, ys]))))

    def set_points(self, xs, ys):
        self._offsets = None
        self.set_offsets(np.column_stack([xs, ys]))

    def get_points(self):
        return self._offsets[:,0], self._offsets[:, 1]

    def set_offsets(self, offsets):
        self._offsets = offsets
        self._npts = offsets.shape[0]
        self._collection.set_offsets(self._offsets)

    def get_offsets(self):
        return self._offsets

    def set_sizes(self, sizes):
        self._markersizes = sizes if sizes is not None else 1.0
        if (isinstance(self._markersizes, np.ndarray) or type(self._markersizes) in [list, ]) and self._npts > 0:
            assert self._markersizes.shape[0] == self._npts
        else:
            assert type(self._markersizes) in [np.float, np.float32, np.float64, float]
            self._markersizes = np.ones(self._npts, dtype=np.float32) * self._markersizes
        self._collection.set_sizes(sizes, dpi=self._dpi)

    def get_sizes(self):
        return self._markersizes

    def set_values(self, values):
        self._values = np.asarray(values)
        assert self._values.shape[0] == self._npts

    def get_values(self):
        return self._values

    def get_collection(self):
        return self._collection

    def get_array(self):
        return self.get_values()

    def set_array(self, values):
        self.set_values(values)
