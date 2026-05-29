# Copyright 2020-2024 AstroLab Software
# Author: Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import urllib.parse
import importlib
import pkgutil
import base64
import yaml
import gzip
import io

import numpy as np
import pandas as pd

import qrcode
import requests
from astropy.convolution import Box2DKernel, Gaussian2DKernel
from astropy.convolution import convolve as astropy_convolve
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import AsymmetricPercentileInterval, simple_norm
from astroquery.mpc import MPC
from fink_utils.xmatch.simbad import get_simbad_labels
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

# TODO: split these UI snippets into separate file?..
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import html

import fink_filters.ztf.livestream as ffz

# Access local or remove API endpoint


simbad_types = get_simbad_labels("old_and_new")
simbad_types = sorted(simbad_types, key=lambda s: s.lower())

class_colors = {
    "Early SN Ia candidate": "red",
    "SN candidate": "orange",
    "SLSN candidate": "grape",
    "Kilonova candidate": "dark",
    "Microlensing candidate": "lime",
    "Tracklet": "violet",
    "Solar System MPC": "yellow",
    "Solar System candidate": "indigo",
    "Ambiguous": "grape",
    "Unknown": "gray",
    "Simbad": "blue",
}


def isoify_time(t):
    try:
        tt = Time(t)
    except ValueError:
        ft = float(t)
        if ft // 2400000:
            tt = Time(ft, format="jd")
        else:
            tt = Time(ft, format="mjd")
    return tt.iso


def query_and_order_statistics(date="", columns="*", index_by="key:key", drop=True):
    """Query /statistics, and order the resulting dataframe

    Parameters
    ----------
    date: str, optional
        Date (default is '')
    columns: str
        Column names (default is '*')
    index_by: str, optional
        Column name on which to index on (default is key:key)
    drop: bool
        If True, drop original column used to index the dataframe.
        Default is False.

    Returns
    -------
    pdf: Pandas DataFrame
        DataFrame with statistics data, ordered from
        oldest (top) to most recent (bottom)
    """
    pdf = request_api(
        "/api/v1/statistics",
        json={
            "date": date,
            "columns": columns,
            "output-format": "json",
        },
    )

    pdf = pdf.sort_values(index_by)
    pdf = pdf.set_index(index_by, drop=drop)

    # Remove hbase specific fields
    if "key:time" in pdf.columns:
        pdf = pdf.drop(columns=["key:time"])

    return pdf


def markdownify_objectid(objectid):
    """ """
    objectid_markdown = f"[{objectid}](/{objectid})"
    return objectid_markdown


def readstamp(stamp: str, return_type="array", gzipped=True) -> np.array:
    """Read the stamp data inside an alert.

    Parameters
    ----------
    stamp: str
        String containing binary data for the stamp
    return_type: str
        Data block of HDU 0 (`array`) or original FITS uncompressed (`FITS`) as file-object.
        Default is `array`.

    Returns
    -------
    data: np.array
        2D array containing image data (`array`) or FITS file uncompressed as file-object (`FITS`)
    """

    def extract_stamp(fitsdata):
        with fits.open(fitsdata, ignore_missing_simple=True) as hdul:
            if return_type == "array":
                data = hdul[0].data
            elif return_type == "FITS":
                data = io.BytesIO()
                hdul.writeto(data)
                data.seek(0)
        return data

    if not isinstance(stamp, io.BytesIO):
        stamp = io.BytesIO(stamp)

    if gzipped:
        with gzip.open(stamp, "rb") as f:
            return extract_stamp(io.BytesIO(f.read()))
    else:
        return extract_stamp(stamp)


def convert_jd(jd, to="iso", format="jd"):
    """Convert Julian Date into ISO date (UTC)."""
    return Time(jd, format=format).to_value(to)


def convolve(image, smooth=3, kernel="gauss"):
    """Convolve 2D image. Hacked from aplpy"""
    if smooth is None and isinstance(kernel, str) and kernel in ["box", "gauss"]:
        return image

    if smooth is not None and not np.isscalar(smooth):
        raise ValueError(
            "smooth= should be an integer - for more complex "
            "kernels, pass an array containing the kernel "
            "to the kernel= option"
        )

    # The Astropy convolution doesn't treat +/-Inf values correctly yet, so we
    # convert to NaN here.
    image_fixed = np.array(image, dtype=float, copy=True)
    image_fixed[np.isinf(image)] = np.nan

    if isinstance(kernel, str):
        if kernel == "gauss":
            kernel = Gaussian2DKernel(smooth, x_size=smooth * 5, y_size=smooth * 5)
        elif kernel == "box":
            kernel = Box2DKernel(smooth, x_size=smooth * 5, y_size=smooth * 5)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    return astropy_convolve(image, kernel, boundary="extend")


def _data_stretch(
    image,
    vmin=None,
    vmax=None,
    pmin=0.25,
    pmax=99.75,
    stretch="linear",
    vmid: float = 10,
    exponent=2,
):
    """Hacked from aplpy"""
    if vmin is None or vmax is None:
        interval = AsymmetricPercentileInterval(pmin, pmax, n_samples=10000)
        try:
            vmin_auto, vmax_auto = interval.get_limits(image)
        except IndexError:  # no valid values
            vmin_auto = vmax_auto = 0

    if vmin is None:
        # log.info("vmin = %10.3e (auto)" % vmin_auto)
        vmin = vmin_auto
    else:
        pass
        # log.info("vmin = %10.3e" % vmin)

    if vmax is None:
        # log.info("vmax = %10.3e (auto)" % vmax_auto)
        vmax = vmax_auto
    else:
        pass
        # log.info("vmax = %10.3e" % vmax)

    if stretch == "arcsinh":
        stretch = "asinh"

    normalizer = simple_norm(
        image,
        stretch=stretch,
        power=exponent,
        asinh_a=vmid,
        min_cut=vmin,
        max_cut=vmax,
        clip=False,
    )

    data = normalizer(image, clip=True).filled(0)
    data = np.nan_to_num(data)
    # data = np.clip(data * 255., 0., 255.)

    return data  # .astype(np.uint8)


def extract_color(pdf: pd.DataFrame, tolerance: float = 0.3, colnames=None):
    """Extract g-r values for single object a pandas DataFrame

    Parameters
    ----------
    pdf: pandas DataFrame
        DataFrame containing alert parameters from an API call
    tolerance: float
        Maximal distance in days between data points to be associated
    colnames: list
        List of extra column names to keep in the output.

    Returns
    -------
    pdf_gr: pandas DataFrame
        DataFrame containing the time and magnitudes of matched points,
        along with g-r color and its error
    """
    if colnames is None:
        colnames = ["i:jd", "i:magpsf", "i:sigmapsf"]
    else:
        colnames = ["i:jd", "i:magpsf", "i:sigmapsf"] + colnames
        colnames = list(np.unique(colnames))

    pdf_g = pdf[pdf["i:fid"] == 1][colnames]
    pdf_r = pdf[pdf["i:fid"] == 2][colnames]

    # merge_asof expects sorted data
    pdf_g = pdf_g.sort_values("i:jd")
    pdf_r = pdf_r.sort_values("i:jd")

    # As merge_asof does not keep the second jd column - let's make it manually
    pdf_g["v:mjd"] = pdf_g["i:jd"] - 2400000.5
    pdf_r["v:mjd"] = pdf_r["i:jd"] - 2400000.5

    pdf_gr = pd.merge_asof(
        pdf_g,
        pdf_r,
        on="i:jd",
        suffixes=("_g", "_r"),
        direction="nearest",
        tolerance=0.3,
    )
    pdf_gr = pdf_gr[~pdf_gr.isna()["i:magpsf_r"]]  # Keep only matched rows

    pdf_gr["v:g-r"] = pdf_gr["i:magpsf_g"] - pdf_gr["i:magpsf_r"]
    pdf_gr["v:sigma_g-r"] = np.hypot(pdf_gr["i:sigmapsf_g"], pdf_gr["i:sigmapsf_r"])
    pdf_gr["v:delta_jd"] = pdf_gr["v:mjd_g"] - pdf_gr["v:mjd_r"]

    return pdf_gr


def query_mpc(number, kind="asteroid"):
    """Query MPC for information about object 'designation'.

    Parameters
    ----------
    designation: str
        A name for the object that the MPC will understand.
        This can be a number, proper name, or the packed designation.
    kind: str
        asteroid or comet

    Returns
    -------
    pd.Series
        Series containing orbit and select physical information.
    """
    try:
        mpc = MPC.query_object(target_type=kind, number=number)
        mpc = mpc[0]
    except IndexError:
        try:
            mpc = MPC.query_object(target_type=kind, designation=number)
            mpc = mpc[0]
        except IndexError:
            return pd.Series({})
    except RuntimeError:
        return pd.Series({})
    orbit = pd.Series(mpc)
    return orbit


def convert_mpc_type(index):
    dic = {
        0: "Unclassified (mostly Main Belters)",
        1: "Atiras",
        2: "Atens",
        3: "Apollos",
        4: "Amors",
        5: "Mars Crossers",
        6: "Hungarias",
        7: "Phocaeas",
        8: "Hildas",
        9: "Jupiter Trojans",
        10: "Distant Objects",
    }
    return dic[index]


def extract_parameter_value_from_url(param_dic, key, default):
    """ """
    if key in param_dic:
        val = param_dic[key]
    else:
        val = default
    return val


def is_float(s: str) -> bool:
    """Check if s can be transformed as a float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_bayestar_query_url(search: str):
    """Try to infer the query from an URL (GW search)

    Parameters
    ----------
    search: str
        String returned by `dcc.Location.search` property.
        Typically starts with ?

    Returns
    -------
    credible_level: float
        The credible level (0-1)
    event_name: str
        Event name (O3 or O4)
    """
    # remove trailing ?
    search = search[1:]

    # split parameters
    parameters = search.split("&")

    # Make a dictionary with the parameter keys and values
    param_dic = {s.split("=")[0]: s.split("=")[1] for s in parameters}

    credible_level = extract_parameter_value_from_url(param_dic, "credible_level", "")
    event_name = extract_parameter_value_from_url(param_dic, "event_name", "")
    if is_float(credible_level):
        credible_level = float(credible_level)

    return credible_level, event_name


def sine_fit(x, a, b):
    """Sinusoidal function a*sin( 2*(x-b) )

    Parameters
    ----------
    x: float
        in degrees
    a: float
        Amplitude
    b: float
        Phase offset

    """
    return a * np.sin(2 * np.radians(x - b))


def pil_to_b64(im, enc_format="png", **kwargs):
    """Converts a PIL Image into base64 string for HTML displaying

    Parameters
    ----------
    im: PIL Image object
        PIL Image object
    enc_format: str
        The image format for displaying.

    Returns
    -------
    base64 encoding
    """
    buff = io.BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded


def generate_qr(data):
    """Generate a QR code from the data

    To check the generated QR code, simply use:
    >>> img = generate_qr("https://fink-broker.org")
    >>> img.get_image().show()

    Parameters
    ----------
    data: str
        Typically an URL

    Returns
    -------
    PIL image
    """
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(data)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        # color_mask=RadialGradiantColorMask(edge_color=(245, 98, 46))
    )

    return img


def retrieve_oid_from_metaname(name):
    """Search for the corresponding ZTF objectId given a metaname"""
    r = request_api(
        "/api/v1/metadata",
        json={
            "internal_name_encoded": name,
        },
        method="GET",
        output="json",
    )

    if r != []:
        return r[0]["i:objectId"]
    return None


def get_first_finite_value(data, pos=0):
    """Returns first finite value from the array at or after given position.

    Parameters
    ----------
    data: np.array
        Data array
    pos: int
        Position in the array to start search

    Returns
    -------
    Value from the array, or np.nan if no finite values found
    """
    data = data[pos:]

    if len(data):
        data = data[np.isfinite(data)]

    if len(data):
        return data[0]
    else:
        return np.nan


def get_first_value(pdf, colname, default=None):
    """Get first value from given column of a DataFrame, or default value if not exists."""
    if colname in pdf.columns:
        return pdf.loc[0, colname]
    else:
        return default


def get_multi_labels(pdf, colname, default=None, to_avoid=None):
    """Get aggregation of unique labels from given column of a DataFrame, or default value if not exists.

    Parameters
    ----------
    pdf: pd.DataFrame
        Fink Pandas DataFrame
    colname: str
        Column name
    default: NoneType, optional
        Default value to assign if the
        column is not defined in the DataFrame schema
    to_avoid: NoneType or list, optional
        If provided, list of labels to avoid

    Returns
    -------
    out: str

    Examples
    --------
    >>> pdf = pd.DataFrame({"a": ["nan", "AM", "AM", "toto"]})
    >>> get_multi_labels(pdf, "a", to_avoid=["nan"])
    'AM/toto'
    """
    if colname in pdf.columns:
        if to_avoid is None:
            to_avoid = []

        if len(np.unique(pdf[colname])) == 1:
            return pdf.loc[0, colname]

        # Case for multilabels
        out = "/".join(
            [
                i
                for i in np.unique(pdf[colname].values)
                if not i.startswith("Fail") and i not in to_avoid
            ]
        )
        return out
    else:
        return default


def request_api(endpoint, json=None, output="pandas", method="POST", **kwargs):
    """Output is one of 'pandas' (default), 'raw' or 'json'"""
    args = extract_configuration("config.yml")
    APIURL = args["APIURL"]
    if method == "POST":
        r = requests.post(
            f"{APIURL}{endpoint}",
            json=json,
        )
    elif method == "GET":
        URL = f"{APIURL}{endpoint}"
        ARGS = ""
        if json is not None and isinstance(json, dict):
            URL += "?"
            for k, v in json.items():
                # encode reserved characters
                ARGS += "{}={}&".format(
                    urllib.parse.quote_plus(k), urllib.parse.quote_plus(v)
                )
        r = requests.get(URL + ARGS)

    if output == "json":
        if r.status_code != 200:
            return []
        return r.json()
    elif output == "raw":
        if r.status_code != 200:
            return io.BytesIO()
        return io.BytesIO(r.content)
    else:
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.read_json(io.BytesIO(r.content), **kwargs)


def loading(item):
    return html.Div(
        [
            item,
            dmc.LoadingOverlay(
                loaderProps={"variant": "dots", "color": "orange", "size": "xl"},
                overlayProps={"radius": "sm", "blur": 2},
                zIndex=100000,
            ),
        ]
    )


def help_popover(text, id, trigger=None, className=None):
    """Make clickable help icon with popover at the bottom right corner of current element"""
    if trigger is None:
        trigger = html.I(
            className="fa fa-question-circle fa-1x",
            id=id,
        )
        if className is None:
            className = "d-flex align-items-end justify-content-end"

    return html.Div(
        [
            trigger,
            dbc.Popover(
                dbc.PopoverBody(
                    text,
                    style={
                        "overflow-y": "auto",
                        "white-space": "pre-wrap",
                        "max-height": "80vh",
                    },
                ),
                target=id,
                trigger="legacy",
                placement="auto",
                style={"width": "80vw", "max-width": "800px"},
                className="shadow-lg",
            ),
        ],
        className=className,
    )


def template_button_for_external_conesearch(
    className="btn btn-default zoom btn-circle btn-lg btn-image",
    style=None,
    color="dark",
    outline=True,
    title="",
    target="_blank",
    href="",
):
    """Template button for external conesearch

    Parameters
    ----------
    className: str, optional
        Styling options. Default is `btn btn-default zoom btn-circle btn-lg btn-image`
    style: dict, optional
        Extra styling options. Default is {}
    color: str, optional
        Color of the button (default is `dark`)
    outline: bool, optional
    title: str, optional
        Title of the object. Default is ''
    target: str, optional
        Open in the same window or in a new tab (default).
    href: str, optional
        targeted URL
    """
    if style is None:
        style = {}

    button = dbc.Button(
        className=className,
        style=style,
        color=color,
        outline=outline,
        title=title,
        target=target,
        href=href,
    )

    return button


def create_button_for_external_conesearch(
    kind: str, ra0: float, dec0: float, radius=None, width=4
):
    """Create a button that triggers an external conesearch

    The button is wrapped within a dbc.Col object.

    Parameters
    ----------
    kind: str
        External resource name. Currently available:
        - asas-sn, snad, vsx, tns, simbad, datacentral, ned, sdss
    ra0: float
        RA for the conesearch center
    dec0: float
        DEC for the conesearch center
    radius: int or float, optional
        Radius for the conesearch. Each external resource has its
        own default value (default), as well as its own units.
    width: int, optional
        dbc.Col width parameter. Default is 4.
    """
    if kind == "asas-sn-variable":
        if radius is None:
            radius = 0.5
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={
                    "background-image": "url(/assets/buttons/assassin_logo.png)",
                    "background-color": "black",
                },
                title="ASAS-SN",
                href=f"https://asas-sn.osu.edu/variables?ra={ra0}&dec={dec0}&radius={radius}&vmag_min=&vmag_max=&amplitude_min=&amplitude_max=&period_min=&period_max=&lksl_min=&lksl_max=&class_prob_min=&class_prob_max=&parallax_over_err_min=&parallax_over_err_max=&name=&references[]=I&references[]=II&references[]=III&references[]=IV&references[]=V&references[]=VI&sort_by=raj2000&sort_order=asc&show_non_periodic=true&show_without_class=true&asassn_discov_only=false&",
            ),
            width=width,
        )
    elif kind == "asas-sn":
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={
                    "background-image": "url(/assets/buttons/assassin_logo.png)",
                    "background-color": "black",
                },
                title="ASAS-SN",
                href=f"https://asas-sn.osu.edu/?ra={ra0}&dec={dec0}",
            ),
            width=width,
        )
    elif kind == "snad":
        if radius is None:
            radius = 5
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={"background-image": "url(/assets/buttons/snad.svg)"},
                title="SNAD",
                href=f"https://ztf.snad.space/search/{ra0} {dec0}/{radius}",
            ),
            width=width,
        )
    elif kind == "vsx":
        if radius is None:
            radius = 0.1
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={"background-image": "url(/assets/buttons/vsx.png)"},
                title="AAVSO VSX",
                href=f"https://www.aavso.org/vsx/index.php?view=results.get&coords={ra0}+{dec0}&format=d&size={radius}",
            ),
            width=width,
        )
    elif kind == "tns":
        if radius is None:
            radius = 5
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={
                    "background-image": "url(/assets/buttons/tns_logo.png)",
                    "background-size": "auto 100%",
                    "background-position-x": "left",
                },
                title="TNS",
                href=f"https://www.wis-tns.org/search?ra={ra0}&decl={dec0}&radius={radius}&coords_unit=arcsec",
            ),
            width=width,
        )
    elif kind == "simbad":
        if radius is None:
            radius = 0.08
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={"background-image": "url(/assets/buttons/simbad.png)"},
                title="SIMBAD",
                href=f"http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra0}%20{dec0}&Radius={radius}",
            ),
            width=width,
        )
    elif kind == "datacentral":
        if radius is None:
            radius = 2.0
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={"background-image": "url(/assets/buttons/dclogo_small.png)"},
                title="DataCentral Data Aggregation Service",
                href=f"https://das.datacentral.org.au/open?RA={ra0}&DEC={dec0}&FOV={0.5}&ERR={radius}",
            ),
            width=width,
        )
    elif kind == "ned":
        if radius is None:
            radius = 1.0
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={
                    "background-image": "url(/assets/buttons/NEDVectorLogo_WebBanner_100pxTall_2NoStars.png)",
                    "background-color": "black",
                },
                title="NED",
                href=f"http://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0&ra={ra0}&dec={dec0}&radius={radius}&obj_sort=Distance+to+search+center&img_stamp=Yes",
            ),
            width=width,
        )
    elif kind == "sdss":
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={"background-image": "url(/assets/buttons/sdssIVlogo.png)"},
                title="SDSS",
                href=f"http://skyserver.sdss.org/dr13/en/tools/chart/navi.aspx?ra={ra0}&dec={dec0}",
            ),
            width=width,
        )
    elif kind == "casda":
        button = dbc.Col(
            template_button_for_external_conesearch(
                style={
                    "background-image": "url(/assets/buttons/csiro-logo.png)",
                    "background-color": "black",
                },
                title="CASDA",
                href=f"https://data.csiro.au/domain/casdaCutoutService/results?surveys=RACS-Low&surveys=RACS-Mid&surveys=RACS-High&size={radius}&ra={ra0}&dec={dec0}",
            ),
            width=width,
        )

    return button


def extract_configuration(filename):
    """Extract user defined configuration

    Parameters
    ----------
    filename: str
        Full path to the `config.yml` file.

    Returns
    -------
    out: dict
        Dictionary with user defined values.
    """
    config = yaml.load(open("config.yml"), yaml.Loader)
    if config["HOST"].endswith(".org"):
        config["SITEURL"] = "https://" + config["HOST"]
    else:
        config["SITEURL"] = "http://" + config["HOST"] + ":" + str(config["PORT"])
    return config


def apparent_flux_dr(mag_dc, err_dc, mjy=False):
    """ """
    if mjy:
        scale = 1000
    else:
        scale = 1

    flux = 3631 * 10 ** (-0.4 * mag_dc) * scale
    sigma_flux = flux * 0.4 * np.log(10) * err_dc

    return flux, sigma_flux


def select_struct(k, prefix=""):
    """Select struct field"""
    if k.startswith("mangrove"):
        return "{}.{}".format(*k.split("_", maxsplit=1))
    elif k.startswith("blazar"):
        return "{}.{}".format(*k.rsplit("_", maxsplit=1))
    elif k.startswith("cutout"):
        return "{}".format(k.split("_")[0])
    else:
        return "{}{}".format(prefix, k)


def format_field_for_data_transfer(
    datasource, with_predefined_options=True, cutouts_allowed=True
):
    """Get schema from API, and make it suitable for Data Transfer"""
    data = []
    if datasource != "ZTF":
        # high level
        packet = {
            "group": "Pre-defined schema",
            "items": [
                {"value": "Full packet", "label": "Full packet"},
            ],
        }
        data.append(packet)
    else:
        schema = request_api("/api/v1/schema", method="GET", output="json")

        if with_predefined_options:
            # high level
            packet = {
                "group": "Pre-defined schema",
                "items": [
                    {"value": "Full packet", "label": "Full packet"},
                    {
                        "value": "Medium packet",
                        "label": "Medium packet (all but cutouts)",
                    },
                    {
                        "value": "Light packet",
                        "label": "Light packet (all but cutouts and history)",
                    },
                ],
            }
            data.append(packet)

        # objectId
        objectid = {
            "group": "ZTF unique object identifier",
            "items": [{"value": "objectId", "label": "objectId"}],
        }
        data.append(objectid)

        # classification
        objectid = {
            "group": "Fink derived classification",
            "items": [{"value": "finkclass", "label": "finkclass"}],
        }
        data.append(objectid)

        # Fink added values
        labels = [
            "{}".format(select_struct(k))
            for k in schema["Fink science module outputs (d:)"].keys()
            if k not in ["tag"]
        ]
        fink = {
            "group": "Fink science module outputs",
            "items": [{"value": label, "label": label} for label in labels],
        }
        data.append(fink)

        # candidate
        labels = [
            "{}".format(select_struct(k, "candidate."))
            for k in schema["ZTF original fields (i:)"].keys()
            if k != "objectId"
        ]
        candidate = {
            "group": "ZTF original fields",
            "items": [{"value": label, "label": label} for label in labels],
        }
        data.append(candidate)

        if cutouts_allowed:
            # Cutouts
            labels = [
                "{}".format(select_struct(k))
                for k in schema["ZTF original cutouts (b:)"].keys()
            ]
            cutout = {
                "group": "ZTF original cutouts",
                "items": [{"value": label, "label": label} for label in labels],
            }
            data.append(cutout)

    return data


def create_datatransfer_livestream_table():
    """ """
    # Get list of filters
    modules = [
        "fink_filters.ztf.livestream.{}.filter".format(mod)
        for _, mod, _ in pkgutil.iter_modules(ffz.__path__)
        if mod.startswith("filter")
    ]

    # header
    rows = []
    for module in modules:
        name = module.split(".")[-2]
        doc = importlib.import_module(module).__doc__
        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(name),
                    dmc.TableTd(doc),
                ]
            )
        )

    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh("Name", w="35%"),
                dmc.TableTh("Description", w="65%"),
            ]
        )
    )
    body = dmc.TableTbody(rows)
    caption = dmc.TableCaption("Filters description")

    table_candidate = dmc.TableScrollContainer(
        dmc.Table(
            [head, body, caption],
            horizontalSpacing="xl",
            highlightOnHover=True,
        ),
        maxHeight=300,
        minWidth=0,
        type="scrollarea",
    )
    return table_candidate


def create_datatransfer_schema_table(cutouts_allowed=True):
    """ """
    schema = request_api("/api/v1/schema", method="GET", output="json")

    def format_type(t):
        if isinstance(t, list):
            return t[-1]
        else:
            return t

    rows = []
    rows.append(
        dmc.TableTr(
            [
                dmc.TableTd("objectId"),
                dmc.TableTd("ZTF"),
                dmc.TableTd("string"),
                dmc.TableTd("Unique identifier for an object"),
            ]
        )
    )
    rows.append(
        dmc.TableTr(
            [
                dmc.TableTd("finkclass"),
                dmc.TableTd("Fink"),
                dmc.TableTd("string"),
                dmc.TableTd("Fink derived classification"),
            ]
        )
    )

    provenances = ["Fink science module outputs (d:)", "ZTF original fields (i:)"]
    prefixes = ["", "candidate."]

    if cutouts_allowed:
        provenances.append("ZTF original cutouts (b:)")
        prefixes.append("")
    for prov, prefix in zip(provenances, prefixes):
        # Table candidates
        labels = [
            select_struct(k, prefix)
            for k in schema[prov].keys()
            if k not in ["objectId", "tag"]
        ]
        types = [
            format_type(v["type"])
            for k, v in schema[prov].items()
            if k not in ["objectId", "tag"]
        ]
        docs = [
            v["doc"] for k, v in schema[prov].items() if k not in ["objectId", "tag"]
        ]

        [
            rows.append(
                dmc.TableTr(
                    [
                        dmc.TableTd(label),
                        dmc.TableTd(prov.split(" ")[0]),
                        dmc.TableTd(type_),
                        dmc.TableTd(doc),
                    ]
                )
            )
            for label, type_, doc in zip(labels, types, docs)
        ]

    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh("Name", w="25%"),
                dmc.TableTh("From", w="15%"),
                dmc.TableTh("Type", w="15%"),
                dmc.TableTh("Documentation"),
            ]
        )
    )
    body = dmc.TableTbody(rows)
    caption = dmc.TableCaption("Alert schema")

    table_candidate = dmc.TableScrollContainer(
        dmc.Table(
            [head, body, caption],
            horizontalSpacing="xl",
            highlightOnHover=True,
        ),
        maxHeight=300,
        minWidth=0,
        type="scrollarea",
    )
    return dmc.Stack(
        [
            dmc.Text(
                "Full packet will give you all ZTF + Fink content. Light packet will give you all ZTF but cutouts, and all Fink."
            ),
            table_candidate,
        ]
    )
