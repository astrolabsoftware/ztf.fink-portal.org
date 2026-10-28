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
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from astropy.time import Time
from dash import Input, Output, callback_context, dcc, html
from fink_utils.xmatch.simbad import get_simbad_labels

from app import app
from apps.utils import class_colors, get_first_finite_value, help_popover


def card_sn_scores() -> html.Div:
    """Card containing the SN ML score evolution

    Returns
    -------
    card: html.Div
        Div with the scores drawn inside
    """
    graph_lc = dcc.Graph(
        id="lightcurve_scores",
        style={
            "width": "100%",
            "height": "20pc",
        },
        config={"displayModeBar": False},
    )
    graph_scores = dcc.Graph(
        id="scores",
        style={
            "width": "100%",
            "height": "20pc",
        },
        config={"displayModeBar": False},
    )
    graph_t2 = html.Div(
        id="t2",
        style={"height": "25pc", "width": "100%"},
    )
    graph_color = dcc.Graph(
        id="colors",
        style={
            "width": "100%",
            "height": "20pc",
        },
        config={"displayModeBar": False},
    )
    graph_color_rate = dcc.Graph(
        id="colors_rate",
        style={
            "width": "100%",
            "height": "20pc",
        },
        config={"displayModeBar": False},
    )
    color_explanation = dcc.Markdown(
        """
        - `g-r`: Color computed from nearest `g` and `r` measurements closer than 0.3 days to each other
        """,
    )
    color_rate_explanation = dcc.Markdown(
        """
        - `rate g-r`: color increase rate per day.
        - `rate g`: magnitude increase rate per day for the `g` band.
        - `rate r`: magnitude increase rate per day for the `r` band.
        """,
    )
    card = html.Div(
        [
            dmc.Paper(graph_lc),
            dmc.Space(h=10),
            dmc.Paper(
                [
                    dmc.Tabs(
                        [
                            dmc.TabsList(
                                [
                                    dmc.TabsTab("SN scores", value="snt0"),
                                    dmc.TabsTab("T2 scores", value="snt0a"),
                                    dmc.TabsTab("Color evolution", value="snt1"),
                                    dmc.TabsTab("Color and mag rate", value="snt2"),
                                ],
                            ),
                            dmc.TabsPanel(
                                graph_scores,
                                value="snt0",
                            ),
                            dmc.TabsPanel(
                                graph_t2,
                                value="snt0a",
                            ),
                            dmc.TabsPanel(
                                [
                                    graph_color,
                                    html.Br(),
                                    color_explanation,
                                ],
                                value="snt1",
                            ),
                            dmc.TabsPanel(
                                [
                                    graph_color_rate,
                                    html.Br(),
                                    color_rate_explanation,
                                ],
                                value="snt2",
                            ),
                        ],
                        value="snt0",
                    ),
                ],
            ),
        ],
    )
    return card


@app.callback(
    Output("card_sn_properties", "children"),
    [
        Input("lightcurve_scores", "clickData"),
        Input("scores", "clickData"),
        Input("colors", "clickData"),
        Input("colors_rate", "clickData"),
        Input("object-data", "data"),
    ],
    prevent_initial_call=True,
)
def card_sn_properties(clickData1, clickData2, clickData3, clickData4, object_data):
    """Add an element containing SN alert data (right side of the page)

    The element is updated when the the user click on the point in the lightcurve
    """
    msg = dcc.Markdown(
        """
        Fink's machine learning classification scores for Supernovae are derived from:
        - [SuperNNova](https://github.com/supernnova/SuperNNova) ([Möller & de Boissière 2019](https://academic.oup.com/mnras/article-abstract/491/3/4277/5651173)) to classify SNe at all light-curve epochs (`SN Ia score` & `SNe score`)
        - Random Forest ([Leoni et al. 2021](https://arxiv.org/abs/2111.11438)) and ([Ishida et al. 2019b](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483....2I/abstract)) to classify early (pre-max) SN candidates (`Early SN Ia score`)
        - Transformers for general multi-variate time-series data, based on [Allam Jr. et al. 2022](https://arxiv.org/abs/2105.06178)

        Note that we then combine these scores, with other criteria,
        to give a final classification to the alert. An `SN candidate` requires that:
        - the alert passes the Fink quality cuts
        - the alert has no known transient association (from catalogues)
        - the alert has at least one of a SuperNNova model trained to identify SNe Ia or SNe (`SN Ia score` or `SNe score`) with a probability higher than 50% of this alert being a SN.

        In addition, the alert is considered as `Early SN Ia candidate` if it also satisfies:
        - the alert is relatively new (number of previous detections < 20)
        - the alert has the Random Forest model trained to select early supernovae Ia (`Early SN Ia score`) with a probability higher than 50% of this alert being a SN Ia.
        """,
    )

    pdf = pd.read_json(io.StringIO(object_data))
    pdf = pdf.sort_values("i:jd", ascending=False)

    # Which graph was clicked, if any?
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "lightcurve_scores":
        clickData = clickData1
    elif triggered_id == "scores":
        clickData = clickData2
    elif triggered_id == "colors":
        clickData = clickData3
    elif triggered_id == "colors_rate":
        clickData = clickData4
    else:
        clickData = None

    if clickData is not None:
        time0 = clickData["points"][0]["x"]
        # Round to avoid numerical precision issues
        jds = pdf["i:jd"].apply(lambda x: np.round(x, 3)).to_numpy()
        jd0 = np.round(Time(time0, format="iso").jd, 3)
        position = np.where(jds == jd0)[0][0]
    else:
        position = 0

    date0 = pdf["v:lastdate"].to_numpy()[position]
    snn_snia_vs_nonia = pdf["d:snn_snia_vs_nonia"].to_numpy()[position]
    snn_sn_vs_all = pdf["d:snn_sn_vs_all"].to_numpy()[position]
    rf_snia_vs_nonia = pdf["d:rf_snia_vs_nonia"].to_numpy()[position]
    classtar = pdf["i:classtar"].to_numpy()[position]
    drb = pdf["i:drb"].to_numpy()[position]

    g_minus_r = get_first_finite_value(pdf["v:g-r"].to_numpy(), position)
    sigma_g_minus_r = get_first_finite_value(pdf["v:sigma(g-r)"].to_numpy(), position)

    rate_g_minus_r = get_first_finite_value(pdf["v:rate(g-r)"].to_numpy(), position)
    sigma_rate_g_minus_r = get_first_finite_value(
        pdf["v:sigma(rate(g-r))"].to_numpy(), position
    )

    rate_g = get_first_finite_value(
        pdf["v:rate"].to_numpy()[position:][pdf["i:fid"][position:] == 1]
    )
    sigma_rate_g = get_first_finite_value(
        pdf["v:sigma(rate)"].to_numpy()[position:][pdf["i:fid"][position:] == 1]
    )

    rate_r = get_first_finite_value(
        pdf["v:rate"].to_numpy()[position:][pdf["i:fid"][position:] == 2]
    )
    sigma_rate_r = get_first_finite_value(
        pdf["v:sigma(rate)"].to_numpy()[position:][pdf["i:fid"][position:] == 2]
    )

    classification = pdf["v:classification"].to_numpy()[position]

    simbad_types = [] #get_simbad_labels("old_and_new")
    simbad_types = sorted(simbad_types, key=lambda s: s.lower())

    if classification in simbad_types:
        color = class_colors["Simbad"]
    elif classification in class_colors.keys():
        color = class_colors[classification]
    else:
        # Sometimes SIMBAD mess up names :-)
        color = class_colors["Simbad"]

    badge = dmc.Badge(
        classification,
        color=color,
        variant="dot",
    )

    card = html.Div(
        [
            dcc.Markdown(
                """
                Click on a point in the lightcurve to update parameters below.
                """,
                className="m-2",
            ),
            # html.Br(),
            dmc.Paper(
                [
                    badge,
                    dcc.Markdown(
                        rf"""
                        Date: `{date0}`

                        ###### SuperNNova classifiers
                        SN Ia score: `{float(snn_snia_vs_nonia):.2f}`
                        SNe score: `{float(snn_sn_vs_all):.2f}`

                        ###### Early SN Ia classifier
                        RF score: `{float(rf_snia_vs_nonia):.2f}`

                        ###### Variability (diff. magnitude)
                        g-r (last): `{g_minus_r:.2f}` ± `{sigma_g_minus_r:.2f}` mag
                        Rate g-r (last): `{rate_g_minus_r:.2f}` ± `{sigma_rate_g_minus_r:.2f}` mag/day
                        Rate g (last): `{rate_g:.2f}` ± `{sigma_rate_g:.2f}` mag/day
                        Rate r (last): `{rate_r:.2f}` ± `{sigma_rate_r:.2f}` mag/day

                        ###### Extra properties
                        Classtar: `{float(classtar):.2f}`
                        DL Real bogus: `{float(drb):.2f}`
                        """,
                        className="markdown markdown-pre ps-2 pe-2",
                    ),
                    help_popover(msg, "help_sn"),
                ],
                radius="sm",
                p="xs",
                shadow="sm",
                withBorder=True,
            ),
        ],
    )
    return card
