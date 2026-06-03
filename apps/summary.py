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
import io
import requests
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from datetime import datetime
from astropy.coordinates import EarthLocation

import visdcc
from dash import Input, Output, State, dcc, html, no_update, ALL
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from app import app
from apps.cards import card_id, card_lightcurve_summary
from apps.plotting import (
    draw_sso_astrometry,
    draw_sso_lightcurve,
    draw_tracklet_lightcurve,
    draw_tracklet_radec,
)
from apps.sso.cards import card_sso_left
from apps.supernovae.cards import card_sn_scores
from apps.utils import (
    generate_qr,
    loading,
    pil_to_b64,
    request_api,
    retrieve_oid_from_metaname,
)
from apps.varstars.cards import card_explanation_variable
from apps.blazars.cards import card_explanation_blazar
from apps.observability.cards import card_explanation_observability
from apps.observability.utils import additional_observatories

dcc.Location(id="url", refresh=False)


def tab1_content(pdf):
    """Summary tab"""
    tab1_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            style={
                                "width": "100%",
                                "height": "4pc",
                            },
                            config={"displayModeBar": False},
                            id="classbar",
                        ),
                        width=12,
                    ),
                ],
                justify="around",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        children=[card_lightcurve_summary()],
                        md=8,
                    ),
                    dbc.Col(
                        children=[card_id(pdf)],
                        md=4,
                    ),
                ],
                className="g-1",
            ),
        ]
    )

    return tab1_content_


def tab2_content():
    """Supernova tab"""
    tab2_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(card_sn_scores(), md=8),
                    dbc.Col(id="card_sn_properties", md=4),
                ],
                className="g-1",
            ),
        ]
    )
    return [tab2_content_]


def tab3_content():
    """Variable stars tab"""
    nterms_base = dmc.Container(
        [
            dbc.Label("Number of base terms"),
            dbc.Input(
                placeholder="1",
                value=1,
                type="number",
                id="nterms_base",
                debounce=True,
                min=0,
                max=4,
            ),
            dbc.Label("Number of band terms"),
            dbc.Input(
                placeholder="1",
                value=1,
                type="number",
                id="nterms_band",
                debounce=True,
                min=0,
                max=4,
            ),
            dbc.Label("Set manually the period (days)"),
            dbc.Input(
                placeholder="Optional",
                value=None,
                type="number",
                id="manual_period",
                debounce=True,
            ),
            dbc.Label("Range of periods (days)"),
            dbc.InputGroup(
                [
                    dbc.Input(
                        value=0.1,
                        type="number",
                        id="period_min",
                        debounce=True,
                    ),
                    dbc.InputGroupText(" < P < "),
                    dbc.Input(
                        value=1.2,
                        type="number",
                        id="period_max",
                        debounce=True,
                    ),
                ],
            ),
        ],
        className="mb-3",  # , style={'width': '100%', 'display': 'inline-block'}
    )

    submit_varstar_button = dmc.Button(
        "Fit data",
        id="submit_variable",
        color="dark",
        variant="outline",
        fullWidth=True,
        radius="xl",
    )

    card2 = dmc.Paper(
        [
            nterms_base,
        ],
        radius="sm",
        p="xs",
        shadow="sm",
        withBorder=True,
    )

    tab3_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(
                        loading(
                            dmc.Paper(
                                [
                                    html.Div(id="variable_plot"),
                                    card_explanation_variable(),
                                ],
                            ),
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(id="card_variable_button"),
                            html.Br(),
                            card2,
                            html.Br(),
                            submit_varstar_button,
                        ],
                        md=4,
                    ),
                ],
                className="g-1",
            ),
        ]
    )
    return [tab3_content_]


@app.callback(
    Output("tab_sso", "children"),
    [
        Input("object-sso", "data"),
    ],
    prevent_initial_call=True,
)
def tab5_content(object_soo):
    """SSO tab"""
    pdf = pd.read_json(io.StringIO(object_soo))
    if pdf.empty:
        ssnamenr = "null"
        sso_name = "null"
    else:
        ssnamenr = pdf["i:ssnamenr"].to_numpy()[0]
        sso_name = pdf["sso_name"].to_numpy()[0]

    msg = """
    Alert data from ZTF (filled circle) in g (blue) and r (orange) filters, with ephemerides provided by the
    [Miriade ephemeride service](https://ssp.imcce.fr/webservices/miriade/api/ephemcc/) (shaded circle).
    """
    tab1 = dbc.Row(
        [
            dbc.Col(
                [
                    dmc.Space(h=10),
                    draw_sso_lightcurve(pdf),
                    html.Br(),
                    dmc.Accordion(
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        "Information",
                                        icon=[
                                            DashIconify(
                                                icon="tabler:help-hexagon",
                                                color="#3C8DFF",
                                                width=20,
                                            ),
                                        ],
                                    ),
                                    dmc.AccordionPanel(dcc.Markdown(msg)),
                                ],
                                value="info",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    tab2 = dbc.Row(
        [
            dbc.Col(
                [
                    dmc.Space(h=10),
                    draw_sso_astrometry(pdf),
                    dmc.Accordion(
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        "How the residuals are computed?",
                                        icon=[
                                            DashIconify(
                                                icon="tabler:help-hexagon",
                                                color="#3C8DFF",
                                                width=20,
                                            ),
                                        ],
                                    ),
                                    dmc.AccordionPanel(
                                        dcc.Markdown(
                                            "The residuals are the difference between the alert positions and the positions returned by the [Miriade ephemeride service](https://ssp.imcce.fr/webservices/miriade/api/ephemcc/)."
                                        )
                                    ),
                                ],
                                value="residuals",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    msg_phase = r"""
    We propose different phase curve modeling using the traditional `HG`, `HG12` and
    `HG1G2` models. By default, the data is modeled after the three-parameter $H$,
    $G_1$, $G_2$ magnitude phase function for asteroids from
    [Muinonen et al. 2010](https://doi.org/10.1016/j.icarus.2010.04.003).
    We use the implementation in [sbpy](https://sbpy.readthedocs.io/en/latest/sbpy/photometry.html#disk-integrated-phase-function-models) to fit the data.
    In addition, two recent models are available: `SHG1G2` and `sfHG1G2` described below. The title displays the value for the reduced $\chi^2$ of the fit.
    Hit buttons to see the fitted values!

    **SHG1G2**: The model `SHG1G2` - spinned `HG1G2` - adds spin parameters on top of the `HG1G2` model ([Carry et al 2024](https://doi.org/10.1051/0004-6361/202449789)).
    Note that $H$, $G_1$, and $G_2$ are fitted per band, but the spin parameters
    (R, $\alpha_0$, $\beta_0$) are fitted on all bands simultaneously.

    **sfHG1G2**: The model `sfHG1G2` - simultaneous fit `HG1G2` - is a single fit across all
    apparitions/oppositions using the `HG1G2` model, where the fitted $H$ values can vary for each opposition,
    but $G_1$ and $G_2$ remain the same for all ([Colazo et al 2025](https://arxiv.org/abs/2503.05412)).
    This means we fit $N+2$ parameters, where $N$ is the number of oppositions. Here we display the
    mean value for $H$ per band.
    """

    tab3 = dbc.Row(
        [
            dbc.Col(
                [
                    dmc.Space(h=10),
                    html.Div(id="sso_phasecurve"),
                    html.Br(),
                    dmc.Stack(
                        [
                            dmc.RadioGroup(
                                children=dmc.Group(
                                    [
                                        dmc.Radio(k, value=k, color="orange")
                                        for k in [
                                            "SHG1G2",
                                            "sfHG1G2",
                                            "HG1G2",
                                            "HG12",
                                            "HG",
                                        ]
                                    ]
                                ),
                                id="switch-phase-curve-func",
                                value="HG1G2",
                                size="sm",
                            ),
                        ],
                        align="center",
                        justify="center",
                    ),
                    dmc.Space(h=10),
                    dmc.Accordion(
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        "How the phase curve is modeled?",
                                        icon=[
                                            DashIconify(
                                                icon="tabler:help-hexagon",
                                                color="#3C8DFF",
                                                width=20,
                                            ),
                                        ],
                                    ),
                                    dmc.AccordionPanel(
                                        dcc.Markdown(msg_phase, mathjax=True)
                                    ),
                                ],
                                value="phase_curve",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    if ssnamenr != "null":
        left_side = dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab("Lightcurve", value="Lightcurve"),
                        dmc.TabsTab("Astrometry", value="Astrometry"),
                        dmc.TabsTab("Phase curve", value="Phase curve"),
                    ],
                ),
                dmc.TabsPanel(tab1, value="Lightcurve"),
                dmc.TabsPanel(tab2, value="Astrometry"),
                dmc.TabsPanel(tab3, value="Phase curve"),
            ],
            value="Lightcurve",
        )
    else:
        msg = """
        Object not referenced in the Minor Planet Center
        """
        left_side = [
            html.Br(),
            dmc.Alert(children="", title=msg, radius="md", color="red"),
        ]

    tab5_content_ = dbc.Row(
        [
            dmc.Space(h=10),
            dbc.Col(
                left_side,
                md=8,
            ),
            dbc.Col(
                [
                    card_sso_left(ssnamenr, sso_name),
                ],
                md=4,
            ),
        ],
        className="g-1",
    )
    return tab5_content_


@app.callback(
    Output("tab_tracklet", "children"),
    [
        Input("object-tracklet", "data"),
    ],
    prevent_initial_call=True,
)
def tab6_content(object_tracklet):
    """Tracklet tab"""
    pdf = pd.read_json(io.StringIO(object_tracklet))
    tab6_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            draw_tracklet_lightcurve(pdf),
                            html.Br(),
                            draw_tracklet_radec(pdf),
                        ],
                    ),
                ],
            ),
        ]
    )
    return tab6_content_


def tab_observability(pdf):
    """Displays the observation plot (altitude and airmass) of the source after selecting an observatory and a date.

    Also displays the observation plot of the Moon as well as its illumination, and the various definition of night. Bottom axis shows UTC time and top axis shows Local time.
    """
    observatories = np.sort(
        np.concatenate(
            (
                np.unique(EarthLocation.get_site_names()),
                list(additional_observatories.keys()),
            )
        )
    )
    nterms_base = dmc.Container(
        [
            dmc.Divider(variant="solid", label="Follow-up"),
            dmc.Select(
                label="Select your Observatory",
                placeholder="Select an observatory from the list",
                id="observatory",
                data=[{"value": obs, "label": obs} for obs in observatories],
                value="Palomar",
                searchable=True,
                clearable=True,
            ),
            dmc.DateInput(
                id="dateobs",
                label="Pick a date for the follow-up",
                value=datetime.now().date(),
            ),
            dmc.Space(h=10),
            dmc.Switch(
                id="moon_elevation", size="sm", radius="xl", label="Show moon elevation"
            ),
            dmc.Space(h=5),
            dmc.Switch(
                id="moon_phase", size="sm", radius="xl", label="Show moon phase"
            ),
            dmc.Space(h=5),
            dmc.Switch(
                id="moon_illumination",
                size="sm",
                radius="xl",
                label="Show moon illumination",
            ),
        ],
        className="mb-3",  # , style={'width': '100%', 'display': 'inline-block'}
    )

    card2 = dmc.Paper(
        [
            nterms_base,
        ],
        radius="sm",
        p="xs",
        shadow="sm",
        withBorder=True,
    )

    nterms_base_custom_observatory = dmc.Container(
        [
            dmc.Divider(variant="solid", label="Coordinates"),
            dmc.TextInput(
                id="longitude",
                label="Longitude",
                placeholder="in decimal degrees",
                size="sm",
                radius="sm",
                required=False,
                persistence=True,
                persistence_type="session",
            ),
            dmc.TextInput(
                id="latitude",
                label="Latitude",
                placeholder="in decimal degrees",
                size="sm",
                radius="sm",
                required=False,
                persistence=True,
                persistence_type="session",
            ),
            dmc.Space(h=5),
            html.Button(
                "Clear",
                id="clear_button",
                style={
                    "border": "none",
                    "border-radius": "10px",
                    "font-size": "15px",
                    "float": "right",
                },
            ),
        ],
        className="mb-3",  # , style={'width': '100%', 'display': 'inline-block'}
    )

    card_custom_observatory = dmc.Paper(
        [
            nterms_base_custom_observatory,
        ],
        radius="sm",
        p="xs",
        shadow="sm",
        withBorder=True,
    )

    card3 = dmc.Accordion(
        disableChevronRotation=True,
        multiple=True,
        children=[
            dmc.AccordionItem(
                [
                    dmc.AccordionControl(
                        "Custom Observatory",
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
                            card_custom_observatory,
                        ),
                    ),
                ],
                value="external",
            ),
        ],
        styles={"content": {"padding": "5px"}},
    )

    submit_button = dmc.Button(
        "Update plot",
        id="submit_observability",
        color="dark",
        variant="outline",
        fullWidth=True,
        radius="xl",
    )

    tab_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(
                        loading(
                            dmc.Paper(
                                [
                                    # dmc.Center(html.Img(id="observability_plot")),
                                    dmc.Space(h=10),
                                    dmc.Center(dcc.Markdown(id="observability_title")),
                                    html.Div(id="observability_plot"),
                                    dmc.Center(dcc.Markdown(id="moon_data")),
                                    dmc.Space(h=20),
                                    card_explanation_observability(),
                                ],
                            ),
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Br(),
                            card2,
                            html.Br(),
                            dmc.Divider(variant="solid"),
                            card3,
                            html.Br(),
                            submit_button,
                        ],
                        md=4,
                    ),
                ],
                className="g-1",
            ),
        ]
    )
    return tab_content_


def tab7_content():
    """Blazar tab"""
    nterms_base = dmc.Container(
        [
            dmc.Divider(variant="solid", label="Extreme states threshold"),
            dmc.Text("Select your quantile", size="sm"),
            dmc.Slider(
                id="quantile_blazar",
                marks=[
                    {"value": i, "label": "{}%".format(i)} for i in range(0, 51, 10)
                ],
                value=10,
                max=50,
                min=0,
            ),
        ],
        className="mb-3",  # , style={'width': '100%', 'display': 'inline-block'}
    )

    submit_blazar_button = dmc.Button(
        "Update plot",
        id="submit_blazar",
        color="dark",
        variant="outline",
        fullWidth=True,
        radius="xl",
    )

    card2 = dmc.Paper(
        [
            nterms_base,
        ],
        radius="sm",
        p="xs",
        shadow="sm",
        withBorder=True,
    )

    tab_content_ = html.Div(
        [
            dmc.Space(h=10),
            dbc.Row(
                [
                    dbc.Col(
                        loading(
                            dmc.Paper(
                                [
                                    html.Div(id="blazar_plot"),
                                    dmc.Space(h=20),
                                    dmc.Center(
                                        dmc.Button(
                                            "Get DR photometry",
                                            id={
                                                "type": "lightcurve_request_release",
                                                "name": "blazar",
                                            },
                                            variant="outline",
                                            color="gray",
                                            radius="xl",
                                            size="xs",
                                        )
                                    ),
                                    card_explanation_blazar(),
                                ],
                            ),
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(id="card_blazar_button"),
                            html.Br(),
                            card2,
                            html.Br(),
                            submit_blazar_button,
                        ],
                        md=4,
                    ),
                ],
                className="g-1",
            ),
        ]
    )
    return [tab_content_]


def tabs(pdf):
    tabs_list = [
        dmc.TabsTab("Summary", value="Summary"),
    ]
    tabs_panels = [
        dmc.TabsPanel(children=[tab1_content(pdf)], value="Summary"),
    ]

    if not is_tracklet(pdf):
        tabs_list.append(dmc.TabsTab("Observability", value="Observability"))
        tabs_panels.append(
            dmc.TabsPanel(children=[tab_observability(pdf)], value="Observability")
        )

    if len(pdf.index) > 1:
        tabs_list.append(dmc.TabsTab("Supernovae", value="Supernovae"))
        tabs_panels.append(dmc.TabsPanel(tab2_content(), value="Supernovae"))

        tabs_list.append(dmc.TabsTab("Variable stars", value="Variable stars"))
        tabs_panels.append(dmc.TabsPanel(tab3_content(), value="Variable stars"))

    if is_sso(pdf):
        tabs_list.append(dmc.TabsTab("Solar System", value="Solar System"))
        tabs_panels.append(
            dmc.TabsPanel(children=[], id="tab_sso", value="Solar System")
        )

    if is_tracklet(pdf):
        tabs_list.append(dmc.TabsTab("Tracklets", value="Tracklets"))

        tabs_panels.append(
            dmc.TabsPanel(children=[], id="tab_tracklet", value="Tracklets")
        )

    if is_blazar(pdf):
        tabs_list.append(dmc.TabsTab("Blazars", value="Blazars"))
        tabs_panels.append(
            dmc.TabsPanel(tab7_content(), id="tab_blazar", value="Blazars")
        )

    # Default view
    if is_sso(pdf):
        default = "Solar System"
    elif is_tracklet(pdf):
        default = "Tracklets"
    else:
        default = "Summary"

    tabs_ = dmc.Tabs(
        [
            dmc.TabsList(
                tabs_list,
                justify="flex-end",
            ),
            *tabs_panels,
        ],
        value=default,
        id="summary_tabs",
    )

    return tabs_


def is_sso(pdfs):
    """Auxiliary function to check whether the object is a SSO"""
    payload = pdfs["i:ssnamenr"].to_numpy()[0]
    if str(payload) == "null" or str(payload) == "None":
        return False

    if np.all([i == payload for i in pdfs["i:ssnamenr"].to_numpy()]):
        return True

    return False


def is_tracklet(pdfs):
    """Auxiliary function to check whether the object is a tracklet"""
    payload = pdfs["d:tracklet"].to_numpy()[0]

    if str(payload).startswith("TRCK"):
        return True

    return False


def is_blazar(pdfs):
    """Auxiliary function to check whether the object is a blazar

    Notes
    -----
    Works for old schema (_m0), and low/high states with new schema
    """
    # Handle schema migration, and low/high states
    cols = [
        "d:blazar_stats_m0",
        "d:blazar_stats_instantness_low",
        "d:blazar_stats_instantness_high",
    ]
    for col in cols:
        payload = pdfs[col].to_numpy()
        if np.any(payload != -1):
            return True
    return False


@app.callback(
    [
        Output("object-data", "data"),
        Output("object-upper", "data"),
        Output("object-uppervalid", "data"),
        Output("object-sso", "data"),
        Output("object-tracklet", "data"),
    ],
    [
        Input("url", "pathname"),
    ],
)
def store_query(name):
    """Cache query results (data and upper limits) for easy re-use

    https://dash.plotly.com/sharing-data-between-callbacks
    """
    if not name[1:].startswith("ZTF"):
        # check this is not a name generated by a user
        oid = retrieve_oid_from_metaname(name[1:])
        if oid is None:
            raise PreventUpdate
    else:
        oid = name[1:]

    pdf = request_api(
        "/api/v1/objects",
        json={
            "objectId": oid,
            "withupperlim": True,
            "withcutouts": False,
        },
        dtype={"i:ssnamenr": str},  # Force reading this field as string
    )

    pdf["i:ssnamenr"] = pdf["i:ssnamenr"].replace(
        "None", "null"
    )  # For backwards compatibility

    pdfs = pdf[pdf["d:tag"] == "valid"]
    pdfsU = pdf[pdf["d:tag"] == "upperlim"]
    pdfsUV = pdf[pdf["d:tag"] == "badquality"]

    payload = pdfs["i:ssnamenr"].to_numpy()[0]
    is_sso = np.all([i == payload for i in pdfs["i:ssnamenr"].to_numpy()])
    if str(payload) != "null" and is_sso:
        pdfsso = request_api(
            "/api/v1/sso",
            json={"n_or_d": payload, "withEphem": True, "withResiduals": False},
        )

    else:
        pdfsso = pd.DataFrame()

    payload = pdfs["d:tracklet"].to_numpy()[0]

    if str(payload).startswith("TRCK"):
        pdftracklet = request_api(
            "/api/v1/tracklet",
            json={
                "id": payload,
            },
        )

    else:
        pdftracklet = pd.DataFrame()

    return (
        pdfs.to_json(),
        pdfsU.to_json(),
        pdfsUV.to_json(),
        pdfsso.to_json(),
        pdftracklet.to_json(),
    )


@app.callback(
    [
        Output("object-release", "data"),
        Output({"type": "lightcurve_request_release", "name": ALL}, "children"),
        Output("switch-mag-flux", "value"),
    ],
    [
        Input({"type": "lightcurve_request_release", "name": ALL}, "n_clicks"),
    ],
    State("object-data", "data"),
    prevent_initial_call=True,
    background=True,
    running=[
        (
            Output({"type": "lightcurve_request_release", "name": ALL}, "disabled"),
            True,
            True,
        ),
        (
            Output({"type": "lightcurve_request_release", "name": ALL}, "loading"),
            True,
            False,
        ),
    ],
)
def store_release_photometry(n_clicks, object_data):
    if (not np.any(n_clicks)) or not object_data:
        raise PreventUpdate

    pdf = pd.read_json(io.StringIO(object_data))

    mean_ra = np.mean(pdf["i:ra"])
    mean_dec = np.mean(pdf["i:dec"])
    sr_arcsec = 2.0

    r = requests.get(
        "https://db.ztf.snad.space/api/v3/data/latest/circle/full/json",
        params={"ra": mean_ra, "dec": mean_dec, "radius_arcsec": sr_arcsec},
    )

    if r.status_code != 200:
        return no_update, "No DR photometry (error {})".format(r.status_code), no_update

    lc = []
    for v in r.json().values():
        lc1 = pd.DataFrame(v["lc"])
        lc1["filtercode"] = v["meta"]["filter"]
        lc.append(lc1)

    if len(lc):
        pdf_release = pd.concat(lc, ignore_index=True)
        return (
            pdf_release.to_json(),
            [f"DR photometry: {len(pdf_release.index)} points"] * len(n_clicks),
            "DC magnitude",
        )

    return no_update, ["No DR photometry"] * len(n_clicks), no_update


@app.callback(
    Output("qrcode", "children"),
    [
        Input("url", "pathname"),
    ],
)
def make_qrcode(path):
    qrdata = f"https://ztf.fink-portal.org/{path[1:]}"
    qrimg = generate_qr(qrdata)

    return html.Img(src="data:image/png;base64, " + pil_to_b64(qrimg))


def layout(name):
    # even if there is one object ID, this returns  several alerts
    pdf = request_api(
        "/api/v1/objects",
        json={
            "objectId": name[1:],
        },
    )

    if pdf.empty:
        inner = html.Div(
            children=dmc.Container(
                dmc.Center(
                    style={"height": "100%", "width": "100%"},
                    children=[
                        dmc.Alert(
                            title=f"{name[1:]} not found",
                            children="Either the object name does not exist, or it has not yet been injected in our database (nightly data appears at the end of the night).",
                            color="gray",
                            radius="md",
                        ),
                    ],
                ),
                fluid=True,
                className="home",
            )
        )
        layout_ = dmc.MantineProvider(
            [inner],
        )
        return layout_
    else:
        col1 = dmc.GridCol(
            dmc.Skeleton(style={"width": "100%", "height": "15pc"}),
            id="card_id_left",
            className="p-1",
            span=12,
            # lg=12,
            # md=6,
            # sm=12,
        )
        col2 = dmc.GridCol(
            html.Div(
                [
                    visdcc.Run_js(id="aladin-lite-runner"),
                    dmc.Skeleton(
                        style={
                            "width": "100%",
                            "height": "100%",
                        },
                    ),
                    html.Div(
                        id="aladin-lite-div", style={"width": "100%", "height": "27pc"}
                    ),
                ],
                className="p-1",
            ),
            # lg=12,
            # md=6,
            # sm=12,
            span=12,
        )
        struct_left = dmc.Grid([col1, col2], gutter=0, className="g-0")
        struct = dmc.Grid(
            [
                dmc.GridCol(
                    struct_left, span={"base": 12, "md": 3, "lg": 3}, className="p-1"
                ),
                dmc.GridCol(
                    [
                        dmc.Space(h=10),
                        tabs(pdf),
                    ],
                    span={"base": 12, "md": 9, "lg": 9},
                    className="p-1",
                ),
                dcc.Store(id="object-data"),
                dcc.Store(id="object-upper"),
                dcc.Store(id="object-uppervalid"),
                dcc.Store(id="object-sso"),
                dcc.Store(id="object-tracklet"),
                dcc.Store(id="object-release"),
            ],
            gutter="xl",
        )
        # I do not know why I have to pad here...
        return dmc.MantineProvider(
            dmc.Container(struct, fluid="xxl", style={"padding-top": "20px"})
        )
