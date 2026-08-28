# Copyright 2020-2022 AstroLab Software
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
import io
import textwrap

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from dash import Input, Output, State, clientside_callback, dcc, html
from dash_iconify import DashIconify
from fink_utils.photometry.utils import is_source_behind

from app import app
from apps.plotting import all_radio_options
from apps.utils import (
    class_colors,
    create_button_for_external_conesearch,
    get_first_value,
    get_multi_labels,
    help_popover,
    loading,
    request_api,
    simbad_types,
)

from apps.utils import extract_configuration

args = extract_configuration("config.yml")
APIURL = args["APIURL"]

lc_help = r"""
##### Difference magnitude

Circles (&#9679;) with error bars show valid alerts that pass the Fink quality cuts.
In addition, the _Difference magnitude_ view shows:
- upper triangles with errors (&#9650;), representing alert measurements that do not satisfy Fink quality cuts, but are nevetheless contained in the history of valid alerts and used by classifiers.
- lower triangles (&#9661;), representing 5-sigma magnitude limit in difference image based on PSF-fit photometry contained in the history of valid alerts.

If the `Color` switch is turned on, the view also shows the panel with `g - r` color, estimated by combining nearby (closer than 0.3 days) measurements in two filters.

##### DC magnitude
DC magnitude is computed by combining the nearest reference image catalog magnitude (`magnr`),
differential magnitude (`magpsf`), and `isdiffpos` (positive or negative difference image detection) as follows:
$$
m_{DC} = -2.5\log_{10}(10^{-0.4m_{magnr}} + \texttt{sign} 10^{-0.4m_{magpsf}})
$$

where `sign` = 1 if `isdiffpos` = 't' or `sign` = -1 if `isdiffpos` = 'f'.
Before using the nearest reference image source magnitude (`magnr`), you will need
to ensure the source is close enough to be considered an association
(e.g., `distnr` $\leq$ 1.5 arcsec). It is also advised you check the other associated metrics
(`chinr` and/or `sharpnr`) to ensure it is a point source. ZTF recommends
0.5 $\leq$ `chinr` $\leq$ 1.5 and/or -0.5 $\leq$ `sharpnr` $\leq$ 0.5.

The view also shows, with dashed horizontal lines, the levels corresponding to the magnitudes of the nearest reference image catalog entry (`magnr`) used in computing DC magnitudes.

This view may be augmented with the photometric points from [ZTF Data Releases](https://www.ztf.caltech.edu/ztf-public-releases.html) by clicking `Get DR photometry` button. The points will be shown with semi-transparent dots (&#8226;).

##### Difference flux
Difference flux (in Jansky) is constructed from difference magnitude by using the following:
$$
f = 3631 \times \texttt{sign} 10^{-0.4m_{magpsf}}
$$
where `sign` = 1 if `isdiffpos` = 't' or `sign` = -1 if `isdiffpos` = 'f'.

This view also shows the photometry from ZTF Data Releases (see above), which is converted to fluxes using the same formula. Then, the "baseline" flux, which is computed from the nearest reference image catalog magnitude (`magnr`), is subtracted from it, so that the value represent the flux variation w.r.t. the template image, i.e. the difference flux.

Note that we display the flux in milli-Jansky.

##### DC flux
DC flux (in Jansky) is constructed from DC magnitude by using the following:
$$
f_{DC} = 3631 \times 10^{-0.4m_{DC}}
$$

This view also shows the fluxes from ZTF Data Releases, without any baseline correction.

Note that we display the flux in milli-Jansky.
"""


def card_lightcurve_summary():
    """Add a card containing the lightcurve

    Returns
    -------
    card: dbc.Card
        Card with the lightcurve drawn inside
    """
    card = dmc.Paper(
        [
            loading(
                dcc.Graph(
                    id="lightcurve_cutouts",
                    style={
                        "width": "100%",
                        "height": "30pc",
                    },
                    config={"displayModeBar": False},
                    className="mb-2",
                ),
            ),
            dmc.Stack(
                [
                    dmc.Group(
                        [
                            dmc.RadioGroup(
                                id="switch-mag-flux",
                                children=dmc.Group(
                                    [
                                        dmc.Radio(k, value=k, color="orange")
                                        for k in all_radio_options.keys()
                                    ]
                                ),
                                value="Difference magnitude",
                                size="sm",
                            ),
                        ],
                        justify="center",
                        align="center",
                    ),
                    dmc.Group(
                        [
                            dmc.Switch(
                                "Color",
                                id="lightcurve_show_color",
                                color="gray",
                                radius="xl",
                                size="sm",
                                persistence=True,
                            ),
                            dmc.Button(
                                "Get DR photometry",
                                id={
                                    "type": "lightcurve_request_release",
                                    "name": "main",
                                },
                                variant="outline",
                                color="gray",
                                radius="xl",
                                size="xs",
                            ),
                            help_popover(
                                dcc.Markdown(
                                    lc_help,
                                    mathjax=True,
                                ),
                                "help_lc",
                                trigger=dmc.ActionIcon(
                                    DashIconify(icon="mdi:help"),
                                    id="help_lc",
                                    color="gray",
                                    variant="outline",
                                    radius="xl",
                                    size="md",
                                ),
                            ),
                        ],
                        justify="center",
                        align="center",
                    ),
                ]
            ),
        ],
    )
    return card  # dmc.Paper([comp1, comp2, comp3]) #card


def card_explanation_xmatch():
    """Explain how xmatch works"""
    msg = """
    The Fink Xmatch service allows you to cross-match your catalog data with
    all Fink alert data processed so far (more than 60 million alerts, from ZTF). Just drag and drop
    a csv file containing at least position columns named `RA` and `Dec`, and a
    column containing ids named `ID` (could be string, integer, ... anything to identify your objects). Required column names are case insensitive. The catalog can also contained
    other columns that will be displayed.

    The xmatch service will perform a conesearch around the positions with a fix radius of 1.5 arcseconds.
    The initializer for RA/Dec is very flexible and supports inputs provided in a number of convenient formats.
    The following ways of declaring positions are all equivalent:

    * 271.3914265, 45.2545134
    * 271d23m29.1354s, 45d15m16.2482s
    * 18h05m33.9424s, +45d15m16.2482s
    * 18 05 33.9424, +45 15 16.2482
    * 18:05:33.9424, 45:15:16.2482

    The final table will contain the original columns of your catalog for all rows matching a Fink object, with two new columns:

    * `objectId`: clickable ZTF objectId.
    * `classification`: the class of the last alert received for this object, inferred by Fink.

    This service is still experimental, and your feedback is welcome. Note that the system will limit to the first 1000 rows of your file (or 5MB max) for the moment.
    Contact us by opening an [issue](https://github.com/astrolabsoftware/fink-science-portal/issues) if you need other file formats or encounter problems.
    """
    card = dbc.Card(
        dbc.CardBody(
            dcc.Markdown(msg),
        ),
    )
    return card


def create_external_conesearches(ra0, dec0):
    """Create two rows of buttons to trigger external conesearch

    Parameters
    ----------
    ra0: float
        RA for the conesearch center
    dec0: float
        DEC for the conesearch center
    """
    width = 4
    buttons = [
        dbc.Row(
            [
                create_button_for_external_conesearch(
                    kind="tns", ra0=ra0, dec0=dec0, radius=5, width=width
                ),
                create_button_for_external_conesearch(
                    kind="simbad", ra0=ra0, dec0=dec0, radius=0.08, width=width
                ),
                create_button_for_external_conesearch(
                    kind="snad", ra0=ra0, dec0=dec0, radius=5, width=width
                ),
            ],
            justify="around",
        ),
        dbc.Row(
            [
                create_button_for_external_conesearch(
                    kind="datacentral", ra0=ra0, dec0=dec0, radius=2.0, width=width
                ),
                create_button_for_external_conesearch(
                    kind="ned", ra0=ra0, dec0=dec0, radius=1.0, width=width
                ),
                create_button_for_external_conesearch(
                    kind="sdss", ra0=ra0, dec0=dec0, width=width
                ),
            ],
            justify="around",
        ),
        dbc.Row(
            [
                create_button_for_external_conesearch(
                    kind="asas-sn", ra0=ra0, dec0=dec0, radius=0.5, width=width
                ),
                create_button_for_external_conesearch(
                    kind="vsx", ra0=ra0, dec0=dec0, radius=0.1, width=width
                ),
                create_button_for_external_conesearch(
                    kind="casda", ra0=ra0, dec0=dec0, radius=5, width=width
                ),
            ],
            justify="around",
        ),
    ]
    return buttons


def create_external_links_brokers(objectId):
    """ """
    buttons = dbc.Row(
        [
            dbc.Col(
                dbc.Button(
                    className="btn btn-default btn-circle btn-lg zoom btn-image",
                    style={"background-image": "url(/assets/buttons/logo_alerce.png)"},
                    color="dark",
                    outline=True,
                    id="alerce",
                    title="ALeRCE",
                    target="_blank",
                    href=f"https://alerce.online/object/{objectId}",
                ),
            ),
            dbc.Col(
                dbc.Button(
                    className="btn btn-default btn-circle btn-lg zoom btn-image",
                    style={"background-image": "url(/assets/buttons/logo_antares.png)"},
                    color="dark",
                    outline=True,
                    id="antares",
                    title="ANTARES",
                    target="_blank",
                    href=f"https://antares.noirlab.edu/loci?query=%7B%22currentPage%22%3A1,%22filters%22%3A%5B%7B%22type%22%3A%22query_string%22,%22field%22%3A%7B%22query%22%3A%22%2a{objectId}%2a%22,%22fields%22%3A%5B%22properties.ztf_object_id%22,%22locus_id%22%5D%7D,%22value%22%3Anull,%22text%22%3A%22ID%20Lookup%3A%20ZTF21abfmbix%22%7D%5D,%22sortBy%22%3A%22properties.newest_alert_observation_time%22,%22sortDesc%22%3Atrue,%22perPage%22%3A25%7D",
                ),
            ),
            dbc.Col(
                dbc.Button(
                    className="btn btn-default btn-circle btn-lg zoom btn-image",
                    style={"background-image": "url(/assets/buttons/logo_lasair.png)"},
                    color="dark",
                    outline=True,
                    id="lasair",
                    title="Lasair",
                    target="_blank",
                    href=f"https://lasair-ztf.lsst.ac.uk/objects/{objectId}",
                ),
            ),
        ],
        justify="around",
    )
    return buttons


def card_neighbourhood(pdf):
    distnr = get_first_value(pdf, "i:distnr")
    ssnamenr = get_first_value(pdf, "i:ssnamenr")
    distpsnr1 = get_first_value(pdf, "i:distpsnr1")
    neargaia = get_first_value(pdf, "i:neargaia")
    constellation = get_first_value(pdf, "v:constellation")
    gaianame = get_multi_labels(pdf, "d:DR3Name", to_avoid=["nan"])
    cdsxmatch = get_multi_labels(pdf, "d:cdsxmatch", to_avoid=["nan"])
    vsx = get_multi_labels(pdf, "d:vsx", to_avoid=["nan"])
    gcvs = get_multi_labels(pdf, "d:gcvs", to_avoid=["nan"])

    # Sanitize empty values
    if ssnamenr == "null":
        ssnamenr = "N/A"

    if not vsx or vsx == "nan":
        vsx = "Unknown"

    card = dmc.Paper(
        [
            dcc.Markdown(
                f"""
                Constellation: `{constellation}`
                Class (SIMBAD): `{cdsxmatch}`
                Class (VSX): `{vsx}`
                Class (GCVS): `{gcvs}`
                Name (MPC): `{ssnamenr}`
                Name (Gaia): `{gaianame}`
                Distance (Gaia): `{float(neargaia):.2f}` arcsec
                Distance (PS1): `{float(distpsnr1):.2f}` arcsec
                Distance (ZTF): `{float(distnr):.2f}` arcsec
                """,
                className="markdown markdown-pre ps-2 pe-2",
            ),
        ],
        radius="sm",
        p="xs",
        shadow="sm",
        withBorder=True,
        style={"width": "100%"},
    )

    return card


def make_modal_stamps(pdf):
    return [
        dbc.Modal(
            [
                dbc.ModalHeader(
                    [
                        dmc.ActionIcon(
                            DashIconify(icon="tabler:chevron-left"),
                            id="stamps_prev",
                            # title="Next alert",
                            n_clicks=0,
                            variant="default",
                            size=36,
                            color="gray",
                            className="me-1",
                        ),
                        dmc.Select(
                            label="",
                            placeholder="Select a date",
                            searchable=True,
                            nothingFoundMessage="No options found",
                            id="date_modal_select",
                            value=pdf["v:lastdate"].to_numpy()[0],
                            data=[
                                {"value": i, "label": i}
                                for i in pdf["v:lastdate"].to_numpy()
                            ],
                            style={"z-index": 10000000},
                        ),
                        dmc.ActionIcon(
                            DashIconify(icon="tabler:chevron-right"),
                            id="stamps_next",
                            # title="Previous alert",
                            n_clicks=0,
                            variant="default",
                            size=36,
                            color="gray",
                            className="ms-1",
                        ),
                    ],
                    close_button=True,
                    className="p-2 pe-4",
                ),
                loading(
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                id="stamps_modal_content",
                                justify="around",
                                className="g-0 mx-auto",
                            ),
                        ],
                    )
                ),
            ],
            id="stamps_modal",
            scrollable=True,
            centered=True,
            size="xl",
            # style={'max-width': '800px'}
        ),
        dmc.Center(
            dmc.ActionIcon(
                DashIconify(icon="tabler:arrows-maximize"),
                id="maximise_stamps",
                n_clicks=0,
                variant="default",
                radius=30,
                size=36,
                color="gray",
            ),
        ),
    ]


# Toggle stamps modal
clientside_callback(
    """
    function toggle_stamps_modal(n_clicks, is_open) {
        return !is_open;
    }
    """,
    Output("stamps_modal", "is_open"),
    Input("maximise_stamps", "n_clicks"),
    State("stamps_modal", "is_open"),
    prevent_initial_call=True,
)

# Prev/Next for stamps modal
clientside_callback(
    """
    function stamps_prev_next(n_clicks_prev, n_clicks_next, clickData, value, data) {
        let id = data.findIndex((x) => x.value === value);
        let step = 1;

        const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);

        if (triggered == 'lightcurve_cutouts.clickData')
            return clickData.points[0].x;

        if (triggered == 'stamps_prev.n_clicks')
            step = -1;

        id += step;
        if (step > 0 && id >= data.length)
            id = 0;
        if (step < 0 && id < 0)
            id = data.length - 1;

        return data[id].value;
    }
    """,
    Output("date_modal_select", "value"),
    [
        Input("stamps_prev", "n_clicks"),
        Input("stamps_next", "n_clicks"),
        Input("lightcurve_cutouts", "clickData"),
    ],
    State("date_modal_select", "value"),
    State("date_modal_select", "data"),
    prevent_initial_call=True,
)


def card_id(pdf):
    """Add a card containing basic alert data"""
    objectid = pdf["i:objectId"].to_numpy()[0]
    ra0 = pdf["i:ra"].to_numpy()[0]
    dec0 = pdf["i:dec"].to_numpy()[0]

    python_download = f"""import requests
import pandas as pd
import io

# get data for {objectid}
r = requests.post(
    '{APIURL}/api/v1/objects',
    json={{
        'objectId': '{objectid}',
        'output-format': 'json'
    }}
)

# Format output in a DataFrame
pdf = pd.read_json(io.BytesIO(r.content))"""

    curl_download = f"""
curl -H "Content-Type: application/json" -X POST \\
    -d '{{"objectId":"{objectid}", "output-format":"csv"}}' \\
    {APIURL}/api/v1/objects \\
    -o {objectid}.csv
    """

    download_tab = dmc.Tabs(
        [
            dmc.TabsList(
                [
                    dmc.TabsTab("Python", value="Python"),
                    dmc.TabsTab("Curl", value="Curl"),
                ],
            ),
            dmc.TabsPanel(
                dmc.CodeHighlight(code=python_download, language="python"),
                value="Python",
            ),
            dmc.TabsPanel(
                children=dmc.CodeHighlight(code=curl_download, language="bash"),
                value="Curl",
            ),
        ],
        color="red",
        value="Python",
    )

    card = dmc.Accordion(
        multiple=True,
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Alert cutouts",
                        icon=[
                            DashIconify(
                                icon="tabler:flare",
                                color=dmc.DEFAULT_THEME["colors"]["dark"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        [
                            loading(
                                dmc.Paper(
                                    [
                                        dbc.Row(
                                            dmc.Skeleton(
                                                style={
                                                    "width": "100%",
                                                    "aspect-ratio": "3/1",
                                                }
                                            ),
                                            id="stamps",
                                            justify="around",
                                            className="g-0",
                                        ),
                                    ],
                                    radius="sm",
                                    shadow="sm",
                                    withBorder=True,
                                    style={"padding": "5px"},
                                ),
                            ),
                            dmc.Space(h=4),
                            *make_modal_stamps(pdf),
                        ],
                    ),
                ],
                value="stamps",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Alert content",
                        icon=[
                            DashIconify(
                                icon="tabler:file-description",
                                color=dmc.DEFAULT_THEME["colors"]["blue"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        html.Div([], id="alert_table"),
                    ),
                ],
                value="last_alert",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Coordinates",
                        icon=[
                            DashIconify(
                                icon="tabler:target",
                                color=dmc.DEFAULT_THEME["colors"]["orange"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        [
                            html.Div(id="coordinates"),
                            dmc.Center(
                                dmc.RadioGroup(
                                    id="coordinates_chips",
                                    value="EQU",
                                    size="sm",
                                    children=dmc.Group(
                                        [
                                            dmc.Radio(k, value=k, color="orange")
                                            for k in ["EQU", "GAL"]
                                        ]
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
                value="coordinates",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Download data",
                        icon=[
                            DashIconify(
                                icon="tabler:database-export",
                                color=dmc.DEFAULT_THEME["colors"]["red"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        html.Div(
                            [
                                dmc.Group(
                                    [
                                        dmc.Button(
                                            "JSON",
                                            id="download_json",
                                            variant="outline",
                                            color="indigo",
                                            size="compact-sm",
                                            leftSection=DashIconify(
                                                icon="mdi:code-json"
                                            ),
                                        ),
                                        dmc.Button(
                                            "CSV",
                                            id="download_csv",
                                            variant="outline",
                                            color="indigo",
                                            size="compact-sm",
                                            leftSection=DashIconify(
                                                icon="mdi:file-csv-outline"
                                            ),
                                        ),
                                        dmc.Button(
                                            "VOTable",
                                            id="download_votable",
                                            variant="outline",
                                            color="indigo",
                                            size="compact-sm",
                                            leftSection=DashIconify(icon="mdi:xml"),
                                        ),
                                        help_popover(
                                            [
                                                dcc.Markdown(
                                                    "You may also download the data programmatically."
                                                ),
                                                download_tab,
                                                dcc.Markdown(
                                                    f"See {APIURL} for more options"
                                                ),
                                            ],
                                            "help_download",
                                            trigger=dmc.ActionIcon(
                                                DashIconify(icon="mdi:help"),
                                                id="help_download",
                                                variant="outline",
                                                color="indigo",
                                            ),
                                        ),
                                        html.Div(
                                            objectid,
                                            id="download_objectid",
                                            className="d-none",
                                        ),
                                        html.Div(
                                            APIURL,
                                            id="download_apiurl",
                                            className="d-none",
                                        ),
                                    ],
                                    align="center",
                                    justify="center",
                                    gap="xs",
                                ),
                            ],
                        ),
                    ),
                ],
                value="api",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Neighbourhood",
                        icon=[
                            DashIconify(
                                icon="tabler:external-link",
                                color="#15284F",
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        dmc.Stack(
                            [
                                card_neighbourhood(pdf),
                                *create_external_conesearches(ra0, dec0),
                            ],
                            align="center",
                        ),
                    ),
                ],
                value="external",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Other brokers",
                        icon=[
                            DashIconify(
                                icon="tabler:atom-2",
                                color=dmc.DEFAULT_THEME["colors"]["green"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        dmc.Stack(
                            [
                                create_external_links_brokers(objectid),
                            ],
                            align="center",
                        ),
                    ),
                ],
                value="external_brokers",
            ),
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Share",
                        icon=[
                            DashIconify(
                                icon="tabler:share",
                                color=dmc.DEFAULT_THEME["colors"]["gray"][6],
                                width=20,
                            ),
                        ],
                    ),
                    dmc.AccordionPanel(
                        [
                            dmc.Center(
                                html.Div(id="qrcode"),
                                style={"width": "100%", "height": "200"},
                            ),
                        ],
                    ),
                ],
                value="qr",
            ),
        ],
        value=["stamps"],
        styles={"content": {"padding": "5px"}},
    )

    return card


# Downloads handling. Requires CORS to be enabled on the server.
# TODO: We are mostly using it like this until GET requests properly initiate
# downloads instead of just opening the file (so, Content-Disposition etc)
download_js = """
function(n_clicks, name, apiurl){
    if(n_clicks > 0){
        fetch(apiurl + '/api/v1/objects', {
            method: 'POST',
            body: JSON.stringify({
                 'objectId': name,
                 'withupperlim': true,
                 'output-format': '$FORMAT'
            }),
            headers: {
                'Content-type': 'application/json'
            }
        }).then(function(response) {
            return response.blob();
        }).then(function(data) {
            window.saveAs(data, name + '.$EXTENSION');
        }).catch(error => console.error('Error:', error));
    };
    return true;
}
"""
app.clientside_callback(
    download_js.replace("$FORMAT", "json").replace("$EXTENSION", "json"),
    Output("download_json", "n_clicks"),
    [
        Input("download_json", "n_clicks"),
        Input("download_objectid", "children"),
        Input("download_apiurl", "children"),
    ],
)
app.clientside_callback(
    download_js.replace("$FORMAT", "csv").replace("$EXTENSION", "csv"),
    Output("download_csv", "n_clicks"),
    [
        Input("download_csv", "n_clicks"),
        Input("download_objectid", "children"),
        Input("download_apiurl", "children"),
    ],
)
app.clientside_callback(
    download_js.replace("$FORMAT", "votable").replace("$EXTENSION", "vot"),
    Output("download_votable", "n_clicks"),
    [
        Input("download_votable", "n_clicks"),
        Input("download_objectid", "children"),
        Input("download_apiurl", "children"),
    ],
)


def make_badge(text="", color=None, outline=None, tooltip=None, **kwargs):
    style = kwargs.pop("style", {})
    if outline is not None:
        style["border-color"] = outline

    badge = dmc.Badge(
        text,
        color=color,
        variant=kwargs.pop("variant", "dot"),
        style=style,
        **kwargs,
    )

    if tooltip is not None:
        badge = dmc.Tooltip(
            badge,
            label=tooltip,
            color=outline if outline is not None else color,
            className="d-inline",
            multiline=True,
        )

    return badge


def generate_tns_badge(oid, get_icon=False):
    """Generate TNS badge

    Parameters
    ----------
    oid: str
        ZTF object ID

    Returns
    -------
    badge: dmc.Badge or None
    """
    r = request_api(
        "/api/v1/resolver",
        json={
            "resolver": "tns",
            "name": oid,
            "reverse": True,
        },
        output="json",
    )

    icon = None

    if r != []:
        entries = [i["d:fullname"] for i in r]
        if len(entries) > 1:
            types = [i.get("type", "") for i in r]
            # Check if object is classified
            try:
                index = [~i.startswith("AT") for i in types].index(True)
            except ValueError:
                # no classification in list -- take the last one (most recent)
                index = -1
        else:
            index = 0

        payload = r[index]

        if payload["d:type"] != "nan":
            tns_type = payload["d:type"].strip()
            msg = "TNS: {} ({})".format(payload["d:fullname"], tns_type)

            if tns_type.startswith("SN Ia"):
                icon = "/assets/types/SNIa.png"
            elif tns_type.startswith("SN"):
                icon = "/assets/types/SNCC.png"
            elif tns_type.startswith("SLSN"):
                icon = "/assets/types/SLSN.png"
            elif tns_type.startswith("TDE"):
                icon = "/assets/types/TDE.png"

            elif tns_type.startswith("AGN"): # ???
                icon = "/assets/types/AGN.png"
            elif tns_type.startswith("QSO"):
                icon = "/assets/types/AGN.png"

            elif tns_type.startswith("Kilonova"):
                icon = "/assets/types/DDSN.png"

        else:
            msg = "TNS: {}".format(payload["d:fullname"])
        badge = make_badge(
            msg,
            color="red",
            tooltip="Transient Name Server classification",
        )
    else:
        badge = None

    if get_icon:
        return badge, icon

    return badge


def generate_generic_badges(row, variant="dot", get_icon=False):
    """Operates on first row of a DataFrame, or directly on Series from pdf.iterrow()"""
    icon = None

    if isinstance(row, pd.DataFrame):
        # for VSX, aggregate values
        vsx_label = get_multi_labels(row, "d:vsx", default="Unknown", to_avoid=["nan"])
        if vsx_label != row.loc[0].get("d:vsx"):
            row["d:vsx"] = vsx_label

        # Get first row from DataFrame
        row = row.loc[0]

    badges = []

    # SSO
    ssnamenr = row.get("i:ssnamenr")
    if ssnamenr and ssnamenr != "null":
        badges.append(
            make_badge(
                f"SSO: {ssnamenr}",
                variant=variant,
                color="yellow",
                tooltip="Nearest Solar System object",
            ),
        )
        icon = "/assets/types/Asteroid.png"

    tracklet = row.get("d:tracklet")
    if tracklet and tracklet != "null":
        badges.append(
            make_badge(
                f"{tracklet}",
                variant=variant,
                color="violet",
                tooltip="Fink detected tracklet",
            ),
        )

    gcvs = row.get("d:gcvs")
    if gcvs and gcvs != "Unknown":
        badges.append(
            make_badge(
                f"GCVS: {gcvs}",
                variant=variant,
                color=class_colors["Simbad"],
                tooltip="General Catalogue of Variable Stars classification",
            ),
        )

    vsx = row.get("d:vsx")
    if (
        vsx
        and vsx != "Unknown"
        and vsx != "nan"
        and vsx == vsx
        and (isinstance(vsx, str) and not vsx.startswith("Fail"))
    ):
        badges.append(
            make_badge(
                f"VSX: {vsx}",
                variant=variant,
                color=class_colors["Simbad"],
                tooltip="AAVSO VSX classification",
            ),
        )

    # Nearby objects
    distnr = row.get("i:distnr")
    if distnr:
        is_source = is_source_behind(distnr)
        badges.append(
            make_badge(
                f'ZTF: {distnr:.1f}"',
                variant=variant,
                color="cyan",
                outline="red" if is_source else None,
                tooltip="""There is a source behind in ZTF reference image.
                You might want to check the DC magnitude plot, and get DR photometry to see its long-term behaviour
                """
                if is_source
                else "Distance to closest object in ZTF reference image",
            ),
        )

    distpsnr = row.get("i:distpsnr1")
    if distpsnr:
        badges.append(
            make_badge(
                f'PS1: {distpsnr:.1f}"',
                variant=variant,
                color="teal",
                tooltip="Distance to closest object in Pan-STARRS DR1 catalogue",
            ),
        )

    distgaia = row.get("i:neargaia")
    if distgaia:
        badges.append(
            make_badge(
                f'Gaia: {distgaia:.1f}"',
                variant=variant,
                color="teal",
                tooltip="Distance to closest object in Gaia DR3 catalogue",
            ),
        )

    if get_icon:
        return badges, icon

    return badges


def generate_metadata_name(oid):
    """Generate name from metadata

    Parameters
    ----------
    oid: str
        ZTF object ID

    Returns
    -------
    name: str
    """
    r = request_api(
        "/api/v1/metadata",
        json={
            "objectId": oid,
        },
        method="GET",
        output="json",
    )

    if r != []:
        name = r[0]["d:internal_name"]
    else:
        name = None

    return name


@app.callback(
    Output("card_id_left", "children"),
    [
        Input("object-data", "data"),
        Input("object-uppervalid", "data"),
        Input("object-upper", "data"),
    ],
    prevent_initial_call=True,
)
def card_id1(object_data, object_uppervalid, object_upper):
    """Add a card containing basic alert data"""
    pdf = pd.read_json(io.StringIO(object_data))

    objectid = pdf["i:objectId"].to_numpy()[0]
    date_end = pdf["v:lastdate"].to_numpy()[0]
    discovery_date = pdf["v:lastdate"].to_numpy()[-1]
    jds = pdf["i:jd"].to_numpy()
    ndet = len(pdf)

    pdf_upper_valid = pd.read_json(io.StringIO(object_uppervalid))
    if not pdf_upper_valid.empty:
        mask = pdf_upper_valid["i:jd"].apply(lambda x: x not in jds)
        nupper_valid = len(pdf_upper_valid[mask])
    else:
        nupper_valid = 0

    pdf_upper = pd.read_json(io.StringIO(object_upper))
    if not pdf_upper.empty:
        nupper = len(pdf_upper)
    else:
        nupper = 0

    icon = None
    badges = []
    used = set()
    # We need to iterate latest-to-oldest, so no unique(), alas
    for c in pdf["v:classification"]:
        if c in simbad_types:
            color = class_colors["Simbad"]

            if icon is None: # Use latest only
                # Fallback to default icon on unknown/unsupported latest class
                icon = "/assets/Fink_SecondaryLogo_WEB.png"

                if c.startswith("EB*") or c.startswith("Candidate_EB*"):
                    icon = "/assets/types/EclBin.png"

                for _ in ["AGN", "QSO", "QVV", "LINER", "BLLac", "Blazar", "Seyfert"]:
                    if c.startswith(_):
                        icon = "/assets/types/AGN.png"

                for _ in [
                        "YSO", "Candidate_YSO",
                        "pMS", "Candidate_pMS",
                        "TTau", "Candidate_TTau",
                        "Orion_V", "FUOr", "HH",
                        "Eruptive", "Outflow", "outflow"
                ]:
                    if c.startswith(_):
                        icon = "/assets/types/YSO.png"

        elif c in class_colors.keys():
            color = class_colors[c]
        else:
            # Sometimes SIMBAD mess up names :-)
            color = class_colors["Simbad"]

        if c not in used:
            used.add(c)

            badges.append(
                make_badge(
                    c,
                    color=color,
                    tooltip="Fink classification",
                ),
            )

    tns_badge, tns_icon = generate_tns_badge(get_first_value(pdf, "i:objectId"), get_icon=True)
    if tns_badge is not None:
        badges.append(tns_badge)
    if tns_icon is not None:
        # TNS classification takes precedence over Simbad
        icon = tns_icon

    generic_badges, generic_icon = generate_generic_badges(pdf, variant="dot", get_icon=True)
    badges += generic_badges
    if icon is None:
        icon = generic_icon

    meta_name = generate_metadata_name(get_first_value(pdf, "i:objectId"))
    if meta_name is not None:
        extra_div = dbc.Row(
            [
                dbc.Col(
                    dmc.Title(meta_name, order=4, style={"color": "#15284F"}), width=10
                ),
            ],
            justify="start",
            align="center",
        )
    else:
        extra_div = html.Div()

    coords = SkyCoord(
        get_first_value(pdf, "i:ra"), get_first_value(pdf, "i:dec"), unit="deg"
    )

    if icon is not None:
        c1 = dmc.Avatar(
            src=icon, size="xl",
            #radius="lg",
            variant="filled",
            # styles={"image": {"padding": "5px"}}
        )
    else:
        c1 = dmc.Avatar(src="/assets/Fink_SecondaryLogo_WEB.png", size="xl")

    c2 = dmc.Title(
        objectid, order=1, style={"color": "#15284F", "wordWrap": "break-word"}
    )
    card = dmc.Paper(
        [
            dmc.Grid(
                [dmc.GridCol(c1, span="content"), dmc.GridCol(c2, span="auto")],
                gutter="xs",
                align="center",
            ),
            extra_div,
            html.Div(badges),
            dcc.Markdown(
                """
                Discovery date: `{}`
                Last detection: `{}`
                Duration: `{:.2f}` / `{:.2f}` days
                Detections: `{}` good, `{}` bad, `{}` upper
                RA/Dec: `{} {}`
                """.format(
                    discovery_date[:19],
                    date_end[:19],
                    jds[0] - jds[-1],
                    get_first_value(pdf, "i:jdendhist")
                    - get_first_value(pdf, "i:jdstarthist"),
                    ndet,
                    nupper_valid,
                    nupper,
                    coords.ra.to_string(pad=True, unit="hour", precision=2, sep=" "),
                    coords.dec.to_string(
                        pad=True, unit="deg", alwayssign=True, precision=1, sep=" "
                    ),
                ),
                className="markdown markdown-pre ps-2 pe-2 mt-2",
            ),
        ],
        radius="xl",
        p="md",
        shadow="xl",
        withBorder=True,
    )
    return card


def card_search_result(row, i):
    """Display single item for search results"""
    badges = []

    name = row["i:objectId"]
    if name[0] == "[":  # Markdownified
        name = row["i:objectId"].split("[")[1].split("]")[0]

    # Handle different variants for key names from different API entry points
    classification = None
    for key in ["v:classification", "d:classification"]:
        if key in row:
            # Classification
            classification = row.get(key)
            if classification in simbad_types:
                color = class_colors["Simbad"]
            elif classification in class_colors.keys():
                color = class_colors[classification]
            else:
                # Sometimes SIMBAD mess up names :-)
                color = class_colors["Simbad"]

            badges.append(
                make_badge(
                    classification,
                    variant="outline",
                    color=color,
                    tooltip="Fink classification",
                ),
            )

    cdsxmatch = row.get("d:cdsxmatch")
    if cdsxmatch and cdsxmatch != "Unknown" and cdsxmatch != classification:
        badges.append(
            make_badge(
                f"SIMBAD: {cdsxmatch}",
                variant="outline",
                color=class_colors["Simbad"],
                tooltip="SIMBAD classification",
            ),
        )

    badges += generate_generic_badges(row, variant="outline")

    if "i:ndethist" in row:
        ndethist = row.get("i:ndethist")
    elif "d:nalerthist" in row:
        ndethist = row.get("d:nalerthist")
    else:
        ndethist = "?"

    jdend = row.get("i:jdendhist", row.get("i:jd"))
    jdstart = row.get("i:jdstarthist")
    lastdate = row.get("i:lastdate", Time(jdend, format="jd").iso)

    coords = SkyCoord(row["i:ra"], row["i:dec"], unit="deg")

    text = """
    `{}` detection(s) in `{:.1f}` days
    First: `{}`
    Last: `{}`
    Equ: `{} {}`
    Gal: `{}`
    """.format(
        ndethist,
        jdend - jdstart,
        Time(jdstart, format="jd").iso[:19],
        lastdate[:19],
        coords.ra.to_string(pad=True, unit="hour", precision=2, sep=" "),
        coords.dec.to_string(
            pad=True, unit="deg", alwayssign=True, precision=1, sep=" "
        ),
        coords.galactic.to_string(style="decimal"),
    )

    text = textwrap.dedent(text)
    if "i:rb" in row:
        text += "RealBogus: `{:.2f}`\n".format(row["i:rb"])
    if "d:anomaly_score" in row:
        text += "Anomaly score: `{:.2f}`\n".format(row["d:anomaly_score"])

    if "v:separation_degree" in row:
        corner_str = "{:.1f}''".format(row["v:separation_degree"] * 3600)
    else:
        corner_str = f"#{i!s}"

    item = dbc.Card(
        [
            # dbc.CardHeader(
            dbc.CardBody(
                [
                    html.A(
                        dmc.Group(
                            [
                                dmc.Text(
                                    f"{name}", style={"fontWeight": 700, "fontSize": 26}
                                ),
                                dmc.Space(w="sm"),
                                *badges,
                            ],
                            gap=3,
                        ),
                        href=f"/{name}",
                        target="_blank",
                        className="text-decoration-none",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dmc.Skeleton(
                                    style={
                                        "width": "12pc",
                                        "height": "12pc",
                                    },
                                ),
                                id={
                                    "type": "search_results_cutouts",
                                    "objectId": name,
                                    "index": i,
                                },
                                width="auto",
                            ),
                            dbc.Col(
                                dcc.Markdown(
                                    text,
                                    style={"white-space": "pre-wrap"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dmc.Skeleton(
                                    style={
                                        "width": "100%",
                                        "height": "15pc",
                                    },
                                ),
                                id={
                                    "type": "search_results_lightcurve",
                                    "objectId": name,
                                    "index": i,
                                },
                                xs=12,
                                md=True,
                            ),
                        ],
                        justify="start",
                        className="g-2",
                    ),
                    # Upper right corner badge
                    dbc.Badge(
                        corner_str,
                        color="light",
                        pill=True,
                        text_color="dark",
                        className="position-absolute top-0 start-100 translate-middle border",
                    ),
                ],
            ),
        ],
        color="white",
        className="mb-2 shadow border-1",
    )

    return item
