import numpy as np
from abc import ABC
from abc import abstractmethod
from matplotlib import cm
from matplotlib.colors import ListedColormap


class MultivariateColourScale(ABC):
    cmc1_bottom_rgb = tuple()
    cmc1_mid_rgb = tuple()
    cmc1_top_rgb = tuple()
    cmc1_bottom_rgba = tuple()
    cmc1_mid_rgba = tuple()
    cmc1_top_rgba = tuple()
    cm_compose1=None
    cm_compose_u=None
    cmc2_bottom_rgb = tuple()
    cmc2_mid_rgb = tuple()
    cmc2_top_rgb = tuple()
    cmc2_bottom_rgba = tuple()
    cmc2_mid_rgba = tuple()
    cmc2_top_rgba = tuple()
    cm_compose2=None
    cm_compose_v=None
    cmc3_bottom_rgb = tuple()
    cmc3_mid_rgb = tuple()
    cmc3_top_rgb = tuple()
    cmc3_bottom_rgba = tuple()
    cmc3_mid_rgba = tuple()
    cmc3_top_rgba = tuple()
    cm_compose3=None
    cm_compose_w=None
    cmc4_bottom_rgb = tuple()
    cmc4_mid_rgb = tuple()
    cmc4_top_rgb = tuple()
    cmc4_bottom_rgba = tuple()
    cmc4_mid_rgba = tuple()
    cmc4_top_rgba = tuple()
    speed_compose=None
    cm_compose_speed=None
    cm0 = None
    cm1 = None
    cm2 = None
    cm3 = None

    @abstractmethod
    def __init__(self):
        pass

    @property
    def colour_u_top_rgb(self):
        return self.cmc1_top_rgb

    @property
    def colour_u_mid_rgb(self):
        return self.cmc1_mid_rgb

    @property
    def colour_u_bottom_rgb(self):
        return self.cmc1_bottom_rgb

    @property
    def colour_v_top_rgb(self):
        return self.cmc2_top_rgb

    @property
    def colour_v_mid_rgb(self):
        return self.cmc2_mid_rgb

    @property
    def colour_v_bottom_rgb(self):
        return self.cmc2_bottom_rgb

    @property
    def colour_w_top_rgb(self):
        return self.cmc3_top_rgb

    @property
    def colour_w_mid_rgb(self):
        return self.cmc3_mid_rgb

    @property
    def colour_w_bottom_rgb(self):
        return self.cmc3_top_rgb

    @property
    def colour_velmag_top_rgb(self):
        return self.cmc4_top_rgb

    @property
    def colour_velmag_mid_rgb(self):
        return self.cmc4_mid_rgb

    @property
    def colour_velmag_bottom_rgb(self):
        return self.cmc4_bottom_rgb

    @property
    def colour_u_top_rgba(self):
        return self.cmc1_top_rgba

    @property
    def colour_u_mid_rgba(self):
        return self.cmc1_mid_rgba

    @property
    def colour_u_bottom_rgba(self):
        return self.cmc1_bottom_rgba

    @property
    def colour_v_top_rgba(self):
        return self.cmc2_top_rgba

    @property
    def colour_v_mid_rgba(self):
        return self.cmc2_mid_rgba

    @property
    def colour_v_bottom_rgba(self):
        return self.cmc2_bottom_rgba

    @property
    def colour_w_top_rgba(self):
        return self.cmc3_top_rgba

    @property
    def colour_w_mid_rgba(self):
        return self.cmc3_mid_rgba

    @property
    def colour_w_bottom_rgba(self):
        return self.cmc3_bottom_rgba

    @property
    def colour_velmag_top_rgba(self):
        return self.cmc4_top_rgba

    @property
    def colour_velmag_mid_rgba(self):
        return self.cmc4_mid_rgba

    @property
    def colour_velmag_bottom_rgba(self):
        return self.cmc4_bottom_rgba

    @property
    def colour_u_scale(self):
        return self.cm_compose_u

    @property
    def colour_v_scale(self):
        return self.cm_compose_v

    @property
    def colour_w_scale(self):
        return self.cm_compose_w

    @property
    def colour_velmag_scale(self):
        return self.cm_compose_speed

    @abstractmethod
    def to_dict(self):
        pass


class BlueQuadrant_GreyBase_AlphaGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #
        # u_img = Image.fromarray(u[0])
        # u_img.save("/var/scratch/experiments/SoA/MEDUSA/u.tif")
        # v_img = Image.fromarray(v[0])
        # v_img.save("/var/scratch/experiments/SoA/MEDUSA/v.tif")

        # cmc1_bottom = (200, 0, 255)  # high - org -> H: 287 S: 100 V: 100
        # cmc1_bottom = (168, 0, 168)  # high # H: 300 S: 100 V: 66
        self.cmc1_bottom_rgb = (129, 0, 129)  # low # H: 300 S: 100 V: 50 - pronounced difference
        # cmc1_mid = (136, 0, 178)  # mid - orig -> H: 286 S: 100 V: 70
        # cmc1_mid = (168, 84, 168)   # H: 300 S: 50 V: 66
        self.cmc1_mid_rgb = (128, 128, 128)  # H: 300 S: 0 V: 50 - pronounced difference
        # cmc1_top = (72, 0, 102)  # low - org -> H: 282 S: 100 V: 40
        # cmc1_top = (255, 128, 255)  # low # H: 300 S: 50 V: 100
        self.cmc1_top_rgb = (255, 128, 255)  # high # H: 300 S: 50 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        # cm_compose1 = np.concatenate((tstack1, mstack1, bstack1), axis=0)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')
        # print(cm_compose1)

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.9]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.9]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')
        # cm1 = colors.LinearSegmentedColormap("u", cm1_cdata, 256, 0.5)

        # cmc1_bottom = (0, 0, 255)  # high - org -> H: 240 S: 100 V: 100
        # cmc2_bottom = (0, 0, 168)  # high # H: 240 S: 100 V: 66
        self.cmc2_bottom_rgb = (0, 0, 129)  # low # H: 240 S: 100 V: 50 - pronounced difference
        # cmc2_mid = (0, 0, 178)  # mid - orig -> H: 240 S: 100 V: 70
        # cmc2_mid = (84, 84, 168)    # H: 240 S: 50 V: 66
        self.cmc2_mid_rgb = (128, 128, 128)  # H: 240 S: 0 V: 50 - pronounced difference
        # cmc2_top = (0, 0, 102)  # low - orig -> H: 240 S: 100 V: 40
        # cmc2_top = (128, 128, 255)  # low # H: 240 S: 50 V: 100
        self.cmc2_top_rgb = (128, 128, 255)  # H: 240 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        # cm_compose2 = np.concatenate((tstack2, mstack2, bstack2), axis=0)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')
        # print(cm_compose2)

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.9]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.9]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')
        # cm2 = colors.LinearSegmentedColormap("v", cm2_cdata, 256, 0.5)

        # cmc1_bottom = (0, 255, 255)  # high - org -> H: 180 S: 100 V: 100
        # cmc2_bottom = (0, 168, 168)  # high # H: 180 S: 100 V: 66
        self.cmc3_bottom_rgb = (0, 128, 128)  # low # H: 180 S: 100 V: 50 - pronounced difference
        # cmc2_mid = (0, 165, 165)  # mid - orig -> H: 180 S: 100 V: 65
        # cmc2_mid = (84, 168, 168)    # H: 180 S: 50 V: 66
        self.cmc3_mid_rgb = (128, 128, 128)  # H: 180 S: 0 V: 50 - pronounced difference
        # cmc2_top = (0, 75, 75)  # low - orig -> H: 180 S: 100 V: 29
        # cmc2_top = (128, 128, 255)  # low # H: 180 S: 50 V: 100
        self.cmc3_top_rgb = (128, 255, 255)  # H: 180 S: 50 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        # cm_compose2 = np.concatenate((tstack2, mstack2, bstack2), axis=0)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')
        # print(cm_compose2)

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.9]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.9]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')
        # cm2 = colors.LinearSegmentedColormap("v", cm2_cdata, 256, 0.5)



        # top_v = [1.0, 1.0, 1.0, 0.0]
        # mid_v = [0.0, 0.0, 0.0, 1.0]
        # low_v = [1.0, 1.0, 1.0, 0.0]

        # top_v = [1.0, 1.0, 1.0, 0.0]
        # mid_v = [0.0, 0.6, 0.6, 1.0]
        # low_v = [1.0, 1.0, 1.0, 0.0]
        # ttstack3 = np.linspace(start=top_v, stop=mid_v, num=64)
        # tstack3 = np.linspace(start=mid_v, stop=mid_v, num=64)
        # bstack3 = np.linspace(start=mid_v, stop=mid_v, num=64)
        # bbstack3 = np.linspace(start=mid_v, stop=low_v, num=64)
        # cm3 = ListedColormap(np.vstack((ttstack3, tstack3, bstack3, bbstack3)), name='s')

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [1.0, 1.0, 1.0, 0.5]
        self.cmc4_bottom_rgb = [1.0, 1.0, 1.0, 0.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        # self.cmc4_top_rgba = [1.0, 1.0, 1.0, 0.0]
        # self.cmc4_mid_rgba = [1.0, 1.0, 1.0, 0.99]
        # self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        self.cmc4_top_rgba = [1.0, 1.0, 1.0, 0.0]
        self.cmc4_mid_rgba = [0.0, 0.0, 0.0, 0.99]
        self.cmc4_bottom_rgba = [0.0, 0.0, 0.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')

    def to_dict(self):
        blue_quadrant_palette_greybase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return blue_quadrant_palette_greybase


class GreenQuadrant_GreyBase_AlphaGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #

        self.cmc1_bottom_rgb = (0, 128, 128)  # low # H: 180 S: 100 V: 50 - pronounced difference
        self.cmc1_mid_rgb = (128, 128, 128)  # mid # H: 180 S: 0 V: 50 - pronounced difference
        self.cmc1_top_rgb = (128, 255, 255)  # high # H: 180 S: 50 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.9]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.9]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')

        self.cmc2_bottom_rgb = (0, 128, 0)  # low # H: 120 S: 100 V: 50 - pronounced difference
        self.cmc2_mid_rgb = (128, 128, 128)  # mid # H: 120 S: 0 V: 50 - pronounced difference
        self.cmc2_top_rgb = (128, 255, 128)  # high # H: 120 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.9]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.9]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')

        self.cmc3_bottom_rgb = (128, 128, 0)  # low # H: 60 S: 100 V: 50 - pronounced difference
        self.cmc3_mid_rgb = (128, 128, 128)  # mid # H: 60 S: 0 V: 50 - pronounced difference
        self.cmc3_top_rgb = (255, 255, 128)  # high # H: 60 S: 50 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.9]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.9]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [1.0, 1.0, 1.0, 0.5]
        self.cmc4_bottom_rgb = [1.0, 1.0, 1.0, 0.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        self.cmc4_top_rgba = [1.0, 1.0, 1.0, 0.0]
        self.cmc4_mid_rgba = [1.0, 1.0, 1.0, 0.99]
        self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')

    def to_dict(self):
        blue_quadrant_palette_greybase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return blue_quadrant_palette_greybase


class RedQuadrant_GreyBase_AlphaGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #

        self.cmc1_bottom_rgb = (128, 128, 0)  # low # H: 60 S: 100 V: 50 - pronounced difference
        self.cmc1_mid_rgb = (128, 128, 128)  # mid # H: 60 S: 0 V: 50 - pronounced difference
        self.cmc1_top_rgb = (255, 255, 128)  # high # H: 60 S: 50 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 0.9]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 0.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 0.9]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')

        self.cmc2_bottom_rgb = (128, 0, 0)  # low # H: 0 S: 100 V: 50 - pronounced difference
        self.cmc2_mid_rgb = (128, 128, 128)  # mid # H: 0 S: 0 V: 50 - pronounced difference
        self.cmc2_top_rgb = (255, 128, 128)  # high # H: 0 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.9]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.0]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.9]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')

        self.cmc3_bottom_rgb = (129, 0, 129)  # low # H: 300 S: 100 V: 50 - pronounced difference
        self.cmc3_mid_rgb = (128, 128, 128)  # H: 300 S: 0 V: 50 - pronounced difference
        self.cmc3_top_rgb = (255, 128, 255)  # high # H: 300 S: 50 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.95], ] * 127, dtype=np.float32)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.9]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.0]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.9]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [1.0, 1.0, 1.0, 0.5]
        self.cmc4_bottom_rgb = [1.0, 1.0, 1.0, 0.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        self.cmc4_top_rgba = [1.0, 1.0, 1.0, 0.0]
        self.cmc4_mid_rgba = [1.0, 1.0, 1.0, 0.99]
        self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')

    def to_dict(self):
        blue_quadrant_palette_greybase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return blue_quadrant_palette_greybase


class BlueQuadrant_BlackBase_GreyGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #

        self.cmc1_bottom_rgb = (234, 191, 255)  # low # H: 280 S: 25 V: 100 - pronounced difference
        self.cmc1_mid_rgb = (0, 0, 0)  # H: 240 S: 0 V: 0 - pronounced difference
        self.cmc1_top_rgb = (191, 234, 255)  # high # H: 200 S: 25 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')
        self.cmc1_top_rgba[3] = 0.99
        self.cmc1_mid_rgba[3] = 0.01
        self.cmc1_bottom_rgba[3] = 0.99

        self.cmc2_bottom_rgb = (213, 128, 255)  # low # H: 280 S: 50 V: 100 - pronounced difference
        self.cmc2_mid_rgb = (0, 0, 0)  # H: 240 S: 0 V: 0 - pronounced difference
        self.cmc2_top_rgb = (128, 212, 255)  # H: 200 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.99]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.99]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.99]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')
        self.cmc2_top_rgba[3] = 0.99
        self.cmc2_mid_rgba[3] = 0.01
        self.cmc2_bottom_rgba[3] = 0.99

        self.cmc3_bottom_rgb = (191, 64, 255)  # low # H: 180 S: 75 V: 50 - pronounced difference
        self.cmc3_mid_rgb = (0, 0, 0)  # H: 180 S: 0 V: 50 - pronounced difference
        self.cmc3_top_rgb = (64, 191, 255)  # H: 180 S: 75 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.99]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.99]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.99]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')
        self.cmc3_top_rgba[3] = 0.99
        self.cmc3_mid_rgba[3] = 0.01
        self.cmc3_bottom_rgba[3] = 0.99

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [0.5, 0.5, 0.5, 1.0]
        self.cmc4_bottom_rgb = [0.0, 0.0, 0.0, 1.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        self.cmc4_top_rgba = [0.0, 0.0, 0.0, 1.0]
        self.cmc4_mid_rgba = [0.99, 0.99, 0.99, 1.0]
        self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')
        self.cmc4_top_rgba[3] = 0.99
        self.cmc4_mid_rgba[3] = 0.01
        self.cmc4_bottom_rgba[3] = 0.99

    def to_dict(self):
        blue_quadrant_palette_blackbase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return blue_quadrant_palette_blackbase


class GreenQuadrant_BlackBase_GreyGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #

        self.cmc1_bottom_rgb = (204, 255, 238)  # low # H: 160 S: 20 V: 100 - pronounced difference
        self.cmc1_mid_rgb = (0, 0, 0)  # H: 120 S: 0 V: 0 - pronounced difference
        self.cmc1_top_rgb = (238, 255, 204)  # high # H: 80 S: 20 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')
        self.cmc1_top_rgba[3] = 0.99
        self.cmc1_mid_rgba[3] = 0.01
        self.cmc1_bottom_rgba[3] = 0.99

        self.cmc2_bottom_rgb = (128, 255, 212)  # low # H: 160 S: 50 V: 100 - pronounced difference
        self.cmc2_mid_rgb = (0, 0, 0)  # H: 120 S: 0 V: 0 - pronounced difference
        self.cmc2_top_rgb = (213, 255, 128)  # H: 80 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.99]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.99]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.99]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')
        self.cmc2_top_rgba[3] = 0.99
        self.cmc2_mid_rgba[3] = 0.01
        self.cmc2_bottom_rgba[3] = 0.99

        self.cmc3_bottom_rgb = (51, 255, 187)  # low # H: 160 S: 80 V: 50 - pronounced difference
        self.cmc3_mid_rgb = (0, 0, 0)  # H: 120 S: 0 V: 0 - pronounced difference
        self.cmc3_top_rgb = (187, 255, 51)  # H: 80 S: 80 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.99]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.99]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.99]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')
        self.cmc3_top_rgba[3] = 0.99
        self.cmc3_mid_rgba[3] = 0.01
        self.cmc3_bottom_rgba[3] = 0.99

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [0.5, 0.5, 0.5, 1.0]
        self.cmc4_bottom_rgb = [0.0, 0.0, 0.0, 1.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        self.cmc4_top_rgba = [0.0, 0.0, 0.0, 1.0]
        self.cmc4_mid_rgba = [0.99, 0.99, 0.99, 1.0]
        self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')
        self.cmc4_top_rgba[3] = 0.99
        self.cmc4_mid_rgba[3] = 0.01
        self.cmc4_bottom_rgba[3] = 0.99

    def to_dict(self):
        green_quadrant_palette_blackbase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return green_quadrant_palette_blackbase


class RedQuadrant_BlackBase_GreyGrading(MultivariateColourScale):

    def __init__(self):
        # ===================================================== #
        # ==== Trying colour composition and mixing =========== #
        # ===================================================== #

        self.cmc1_bottom_rgb = (255, 238, 204)  # low # H: 40 S: 20 V: 100 - pronounced difference
        self.cmc1_mid_rgb = (0, 0, 0)  # H: 0 S: 0 V: 0 - pronounced difference
        self.cmc1_top_rgb = (255, 204, 238)  # high # H: 320 S: 20 V: 100 - pronounced difference
        tstack1 = np.array([[self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack1 = np.array([[self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack1 = np.array([[self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose1 = np.vstack((tstack1, mstack1, bstack1))
        self.cm_compose_u = ListedColormap(self.cm_compose1, name='u_h')

        self.cmc1_top_rgba = [self.cmc1_top_rgb[0] / 255.0, self.cmc1_top_rgb[1] / 255.0, self.cmc1_top_rgb[2] / 255.0, 1.0]
        self.cmc1_mid_rgba = [self.cmc1_mid_rgb[0] / 255.0, self.cmc1_mid_rgb[1] / 255.0, self.cmc1_mid_rgb[2] / 255.0, 1.0]
        self.cmc1_bottom_rgba = [self.cmc1_bottom_rgb[0] / 255.0, self.cmc1_bottom_rgb[1] / 255.0, self.cmc1_bottom_rgb[2] / 255.0, 1.0]
        bstack1 = np.linspace(start=self.cmc1_bottom_rgba, stop=self.cmc1_mid_rgba, num=128)
        tstack1 = np.linspace(start=self.cmc1_mid_rgba, stop=self.cmc1_top_rgba, num=128)
        cm1_stack = np.vstack((bstack1, tstack1))
        self.cm1 = ListedColormap(cm1_stack, name='u')
        self.cmc1_top_rgba[3] = 0.99
        self.cmc1_mid_rgba[3] = 0.01
        self.cmc1_bottom_rgba[3] = 0.99

        self.cmc2_bottom_rgb = (255, 212, 128)  # low # H: 40 S: 50 V: 100 - pronounced difference
        self.cmc2_mid_rgb = (0, 0, 0)  # H: 360 S: 0 V: 0 - pronounced difference
        self.cmc2_top_rgb = (255, 128, 213)  # H: 320 S: 50 V: 100 - pronounced difference
        tstack2 = np.array([[self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack2 = np.array([[self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack2 = np.array([[self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose2 = np.vstack((tstack2, mstack2, bstack2))
        self.cm_compose_v = ListedColormap(self.cm_compose2, name='v_h')

        self.cmc2_top_rgba = [self.cmc2_top_rgb[0] / 255.0, self.cmc2_top_rgb[1] / 255.0, self.cmc2_top_rgb[2] / 255.0, 0.99]
        self.cmc2_mid_rgba = [self.cmc2_mid_rgb[0] / 255.0, self.cmc2_mid_rgb[1] / 255.0, self.cmc2_mid_rgb[2] / 255.0, 0.99]
        self.cmc2_bottom_rgba = [self.cmc2_bottom_rgb[0] / 255.0, self.cmc2_bottom_rgb[1] / 255.0, self.cmc2_bottom_rgb[2] / 255.0, 0.99]
        tstack2 = np.linspace(start=self.cmc2_mid_rgba, stop=self.cmc2_top_rgba, num=128)
        bstack2 = np.linspace(start=self.cmc2_bottom_rgba, stop=self.cmc2_mid_rgba, num=128)
        cm2_stack = np.vstack((bstack2, tstack2))
        self.cm2 = ListedColormap(cm2_stack, name='v')
        self.cmc2_top_rgba[3] = 0.99
        self.cmc2_mid_rgba[3] = 0.01
        self.cmc2_bottom_rgba[3] = 0.99

        self.cmc3_bottom_rgb = (255, 187, 51)  # low # H: 40 S: 80 V: 50 - pronounced difference
        self.cmc3_mid_rgb = (0, 0, 0)  # H: 360 S: 0 V: 0 - pronounced difference
        self.cmc3_top_rgb = (255, 51, 187)  # H: 320 S: 80 V: 100 - pronounced difference
        tstack3 = np.array([[self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        mstack3 = np.array([[self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 1.0], ] * 2, dtype=np.float32)
        bstack3 = np.array([[self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 1.0], ] * 127, dtype=np.float32)
        self.cm_compose3 = np.vstack((tstack3, mstack3, bstack3))
        self.cm_compose_w = ListedColormap(self.cm_compose3, name='w_h')

        self.cmc3_top_rgba = [self.cmc3_top_rgb[0] / 255.0, self.cmc3_top_rgb[1] / 255.0, self.cmc3_top_rgb[2] / 255.0, 0.99]
        self.cmc3_mid_rgba = [self.cmc3_mid_rgb[0] / 255.0, self.cmc3_mid_rgb[1] / 255.0, self.cmc3_mid_rgb[2] / 255.0, 0.99]
        self.cmc3_bottom_rgba = [self.cmc3_bottom_rgb[0] / 255.0, self.cmc3_bottom_rgb[1] / 255.0, self.cmc3_bottom_rgb[2] / 255.0, 0.99]
        tstack3 = np.linspace(start=self.cmc3_mid_rgba, stop=self.cmc3_top_rgba, num=128)
        bstack3 = np.linspace(start=self.cmc3_bottom_rgba, stop=self.cmc3_mid_rgba, num=128)
        cm3_stack = np.vstack((bstack3, tstack3))
        self.cm3 = ListedColormap(cm3_stack, name='w')
        self.cmc3_top_rgba[3] = 0.99
        self.cmc3_mid_rgba[3] = 0.01
        self.cmc3_bottom_rgba[3] = 0.99

        self.cmc4_top_rgb = [1.0, 1.0, 1.0, 1.0]  # high
        self.cmc4_mid_rgb = [0.5, 0.5, 0.5, 1.0]
        self.cmc4_bottom_rgb = [0.0, 0.0, 0.0, 1.0]  # low
        self.speed_compose = np.linspace(start=self.cmc4_bottom_rgb, stop=self.cmc4_top_rgb, num=256)
        self.cm_compose_speed = ListedColormap(self.speed_compose, name='s_h')

        self.cmc4_top_rgba = [0.0, 0.0, 0.0, 1.0]
        self.cmc4_mid_rgba = [0.99, 0.99, 0.99, 1.0]
        self.cmc4_bottom_rgba = [1.0, 1.0, 1.0, 1.0]
        tstack3 = np.linspace(start=self.cmc4_mid_rgba, stop=self.cmc4_top_rgba, num=248)
        bstack3 = np.linspace(start=self.cmc4_bottom_rgba, stop=self.cmc4_mid_rgba, num=8)
        self.cm0 = ListedColormap(np.vstack((bstack3, tstack3)), name='s')
        self.cmc4_top_rgba[3] = 0.99
        self.cmc4_mid_rgba[3] = 0.01
        self.cmc4_bottom_rgba[3] = 0.99

    def to_dict(self):
        red_quadrant_palette_blackbase = {
            'U': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1},
            'V': {'top': self.cmc2_top_rgb, 'mid': self.cmc2_mid_rgb, 'bottom': self.cmc2_bottom_rgb, 'colour_scale': self.cm2},
            'W': {'top': self.cmc3_top_rgb, 'mid': self.cmc3_mid_rgb, 'bottom': self.cmc3_bottom_rgb, 'colour_scale': self.cm3},
            'VelMag': {'top': self.cmc4_top_rgb, 'mid': self.cmc4_mid_rgb, 'bottom': self.cmc4_bottom_rgb, 'colour_scale': self.cm0}
        }
        return red_quadrant_palette_blackbase



class BlackToRed_Monotone(MultivariateColourScale):

    def __init__(self):
        self.cmc1_top_rgb = [1., 0., 0., 1.]
        self.cmc1_mid_rgb = None
        self.cmc1_bottom_rgb = [0., 0., 0., 1.]
        self.cm1 = ListedColormap(np.linspace(start=self.cmc1_bottom_rgb, stop=self.cmc1_top_rgb, num=256, dtype=np.float32), name='B_Rm')

    def to_dict(self):
        black_to_red_palette_blackbase = {
            'Any': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1}
        }
        return black_to_red_palette_blackbase

class WhiteToBlue_AlphaMonotone(MultivariateColourScale):

    def __init__(self):
        super(WhiteToBlue_AlphaMonotone, self).__init__()
        self.cmc1_top_rgb = [0., 0., 1., 1.]
        self.cmc1_mid_rgb = None
        self.cmc1_bottom_rgb = [1., 1., 1., 0.]
        self.cm1 = ListedColormap(np.linspace(start=self.cmc1_bottom_rgb, stop=self.cmc1_top_rgb, num=256, dtype=np.float32), name='W_BAm')

    def to_dict(self):
        white_to_blue_palette_alpha = {
            'Any': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1}
        }
        return white_to_blue_palette_alpha

class WhiteToBlue_OpaqueMonotone(MultivariateColourScale):

    def __init__(self):
        super(WhiteToBlue_OpaqueMonotone, self).__init__()
        self.cmc1_top_rgb = [0., 0., 1., 1.]
        self.cmc1_mid_rgb = None
        self.cmc1_bottom_rgb = [1., 1., 1., 1.]
        self.cm1 = ListedColormap(np.linspace(start=self.cmc1_bottom_rgb, stop=self.cmc1_top_rgb, num=256, dtype=np.float32), name='W_BAm')

    def to_dict(self):
        white_to_blue_palette_opaque = {
            'Any': {'top': self.cmc1_top_rgb, 'mid': self.cmc1_mid_rgb, 'bottom': self.cmc1_bottom_rgb, 'colour_scale': self.cm1}
        }
        return white_to_blue_palette_opaque

bluequadrant_greybase_alphagrading = BlueQuadrant_GreyBase_AlphaGrading()
greenquadrant_greybase_alphagrading = GreenQuadrant_GreyBase_AlphaGrading()
redquadrant_greybase_alphagrading = RedQuadrant_GreyBase_AlphaGrading()
bluequadrant_blackbase_greygrading = BlueQuadrant_BlackBase_GreyGrading()
greenquadrant_blackbase_greygrading = GreenQuadrant_BlackBase_GreyGrading()
redquadrant_blackbase_greygrading = RedQuadrant_BlackBase_GreyGrading()
black_red_monotone = BlackToRed_Monotone()
white_blue_alpha_monotone = WhiteToBlue_AlphaMonotone()
white_blue_opaque_monotone = WhiteToBlue_OpaqueMonotone()

colour_scales = {
    'blue_quadrant_palette_greybase': bluequadrant_greybase_alphagrading.to_dict(),
    'green_quadrant_palette_greybase': greenquadrant_greybase_alphagrading.to_dict(),
    'red_quadrant_palette_greybase': redquadrant_greybase_alphagrading.to_dict(),
    'blue_quadrant_palette_blackbase': bluequadrant_blackbase_greygrading.to_dict(),
    'green_quadrant_palette_blackbase': greenquadrant_blackbase_greygrading.to_dict(),
    'red_quadrant_palette_blackbase': redquadrant_blackbase_greygrading.to_dict(),
    'black_to_red_palette_blackbase': black_red_monotone.to_dict(),
    'white_to_blue_pallete_alpha': white_blue_alpha_monotone.to_dict(),
    'white_to_blue_pallete_opaque': white_blue_opaque_monotone.to_dict()
}