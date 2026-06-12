# Copyright 2020-2025 AstroLab Software
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

import dash
from dash import (
    html,
    dcc,
    Input,
    Output,
    State,
    dash_table,
    no_update,
    clientside_callback,
    ALL,
    MATCH,
)
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
# import visdcc

import dash_mantine_components as dmc
from dash_iconify import DashIconify

from dash_autocomplete_input import AutocompleteInput

from app import server
from app import app

from apps import summary, about, statistics, query_cluster, gw, xmatch

from apps.utils import markdownify_objectid, class_colors, simbad_types
from apps.utils import isoify_time
from apps.utils import convert_jd
from apps.utils import retrieve_oid_from_metaname
from apps.utils import help_popover
from apps.utils import request_api
from apps.utils import extract_configuration
from apps.plotting import draw_cutouts_quickview, draw_lightcurve_preview
from apps.cards import card_search_result
from apps.parse import parse_query

import pandas as pd
import numpy as np

import urllib

config_args = extract_configuration("config.yml")

tns_types = pd.read_csv("assets/tns_types.csv", header=None)[0].to_numpy()
tns_types = sorted(tns_types, key=lambda s: s.lower())

fink_classes = [
    "Anomaly",
    "Unknown",
    # Fink derived classes
    "Early SN Ia candidate",
    "SN candidate",
    "SLSN candidate",
    "Kilonova candidate",
    "Microlensing candidate",
    "Solar System MPC",
    "Solar System candidate",
    "Tracklet",
    "Ambiguous",
    "(CTA) Blazar low state",
    "(CTA) Blazar high state",
    # TNS classified data
    *["(TNS) " + t for t in tns_types],
    # Simbad crossmatch
    *["(SIMBAD) " + t for t in simbad_types],
]

message_help = """
You may search for different kinds of data depending on what you enter. Below you will find the description of syntax rules and some examples.

The search is defined by the set of search terms (object names or coordinates) and options (e.g. search radius, number of returned entries, etc). The latters may be entered either manually or by using the `Quick fields` above the search bar. Supported formats for the options are both `name=value` and `name:value`, where `name` is case-insensitive, and `value` may use quotes to represent multi-word sentences. For some of the options, interactive drop-down menu will be shown with possible values.

The search bar has a button (on the left) to show you the list of your latest search queries so that you may re-use them, and a switch (on the right, right above the search field) to choose the format of results - either card-based (default, shows the previews of object latest cutout and light curve), or tabular (allows to see all the fields of objects).

##### Search for specific ZTF objects

To search for specific object, just use its name, or just a part of it. In the latter case, all objects with the names starting with this pattern will be returned.

Supported name patterns are as follows:
- `ZTFyyccccccc` for ZTF objects, i.e. ZTF followed with 2 digits for the year, and 7 characters after that
- `TRCK_YYYYMMDD_HHMMSS_NN` for tracklets detected at specific moment of time

Examples:
- `ZTF21abfmbix` - search for exact ZTF object
- `ZTF21abfmb` - search for partially matched ZTF object name
- `TRCK_20231213_133612_00` - search for all objects associated with specific tracklet
- `TRCK_20231213` - search for all tracklet events from the night of Dec 13, 2023

##### Search around known astronomical objects

You can run a conesearch around a known astronomical name. Examples:
- Extended objects: M31
- Catalog names: TXS 0506+056
- TNS names: AT 2019qiz, SN 2024aaj

By default, the conesearch radius is 10 arcseconds. You can change the radius by specifying `r=<number>` after the name, e.g. `Crab Nebula r=10m` (see the section Cone Search below).

##### Cone search

If you specify the position and search radius (using `r` option), all objects inside the given cone will be returned. Position may be specified by either coordinates, in either decimal or sexagesimal form, as exact ZTF object name, or as an object name resolvable through TNS or Simbad.

The coordinates may be specified as:
- Pair of degrees
- `HH MM SS.S [+-]?DD MM SS.S`
- `HH:MM:SS.S [+-]?DD:MM:SS.S`
- `HHhMMhSS.Ss [+-]?DDhMMhSS.Ss`
- optionally, you may use one more number as a radius, in either arcseconds, minutes (suffixed with `m`) or degrees (`d`). If specified, you do not need to provide the corresponding keyword (`r`) separately

If the radius is not specified, but the coordinates or resolvable object names are given, the default search radius is 10 arcseconds.

You may also restrict the alert variation time by specifying `after` and `before` keywords. They may be given as UTC timestamps in either ISO string format, as Julian date, or MJD. Alternatively, you may use `window` keyword to define the duration of time window in days.

Examples:
- `246.0422 25.669 30` - search within 30 arcseconds around `RA=246.0422 deg` `Dec=25.669 deg`
- `246.0422 25.669 30 after="2023-03-29 13:36:52" window=10` - the same but also within 10 days since specified time moment
- `16 24 10.12 +25 40 09.3` - search within 10 arcseconds around `RA=10:22:31` `Dec=+40:50:55.5`
- `Vega r=10m` - search within 600 arcseconds (10 arcminutes) from Vega
- `ZTF21abfmbix r=20` - search within 20 arcseconds around the position of ZTF21abfmbix
- `AT2021co` or `AT 2021co` - search within 10 arcseconds around the position of AT2021co

##### Solar System objects

To search for all ZTF objects associated with specific SSO, you may either directly specify `sso` keyword, which should be equal to contents of `i:ssnamenr` field of ZTF packets, or just enter the number or name of the SSO object that the system will try to resolve.

So you may e.g. search for:
- Asteroids by proper name
  - `Vesta`
- Asteroids by number
  - Asteroids (Main Belt): `8467`, `1922`, `33803`
  - Asteroids (Hungarians): `18582`, `77799`
  - Asteroids (Jupiter Trojans): `4501`, `1583`
  - Asteroids (Mars Crossers): `302530`
- Asteroids by designation
  - `2010JO69`, `2017AD19`, `2012XK111`
- Comets by number
  - `10P`, `249P`, `124P`
- Comets by designation
  - `C/2020V2`, `C/2020R2`

##### Class-based search

To see the list of latest objects of specific class (as listed in `v:classification` alert field), just specify the `class` keyword. By default it will return 100 latest ones, but you may also directly specify `last` keywords to alter it.

You may also specify the time interval to refine the search, using the self-explanatory keywords `before` and `after`. The limits may be specified with either time string, JD or MJD values. You may either set both limiting values, or just one of them. The results will be sorted in descending order by time, and limited to specified number of entries.

Finally, you can specify a trend to your search, such as rising or fading. Use the keyword `trend` to see the list of available trends. This is an experimental feature that is expected to evolve.

Examples:
- `class=Unknown` - return 100 latest objects with class `Unknown`
- `last=10 class="Early SN Ia candidate"` - return 10 latest arly SN Ia candidates
- `class="Early SN Ia candidate" before="2023-12-01" after="2023-11-07 04:00:00"` - objects of the same class between 4am on Nov 15, 2023 and Dec 1, 2023
- `class="Early SN Ia candidate" before="2023-12-01" after="2023-11-07 04:00:00" trend=rising` - objects of the same class between 4am on Nov 15, 2023 and Dec 1, 2023, that were rising (becoming brighter).
- `class="(CTA) Blazar low state" trend=low_state after=2025-02-01 before=2025-02-13` - Blazars selected by CTA which were in a low state between the 1st February and 13th February 2025.

"""

msg_info = """
The `Card view` shows the cutout from science image, some basic alert info, and its light curve. Its header also displays the badges for alert classification and the distances from several reference catalogs, as listed in the alert.

By default, the `Table view` shows the following fields:

- i:objectId: Unique identifier for this object
- i:ra: Right Ascension of candidate; J2000 (deg)
- i:dec: Declination of candidate; J2000 (deg)
- v:lastdate: last date the object has been seen by Fink
- v:classification: Classification inferred by Fink (Supernova candidate, Microlensing candidate, Solar System, SIMBAD class, ...)
- i:ndethist: Number of spatially coincident detections falling within 1.5 arcsec going back to the beginning of the survey; only detections that fell on the same field and readout-channel ID where the input candidate was observed are counted. All raw detections down to a photometric S/N of ~ 3 are included.
- v:lapse: number of days between the first and last spatially coincident detections.

You can also add more columns using the dropdown button above the result table. Full documentation of all available fields can be found at {}/api/v1/schema.

The button `Sky Map` will open a popup with embedded Aladin sky map showing the positions of the search results on the sky.
""".format(config_args["APIURL"])

# Smart search field
quick_fields = [
    ["class", "Alert class\nSelect one of Fink supported classes from the menu"],
    [
        "trend",
        "Lightcurve trends: rising, fading.\nExperimental feature, only available in combination with class search.\nFor the class (CTA) `Blazar low state`, the tags low_state and new_low_State are also available.",
    ],
    ["last", "Number of latest alerts to show. Must be used with the `class` keyword."],
    [
        "radius",
        "Radius for cone search\nMay be used as either `r` or `radius`\nIn arcseconds by default, use `r=1m` or `r=2d` for arcminutes or degrees, correspondingly",
    ],
    ["after", "Lower timit on alert time\nISO time, MJD or JD"],
    ["before", "Upper timit on alert time\nISO time, MJD or JD"],
    ["window", "Time window length\nDays"],
]

fink_search_bar = [
    html.Div(
        [
            html.Span("Quick fields:", className="text-secondary"),
        ]
        + [
            html.Span(
                [
                    html.A(
                        __[0],
                        title=__[1],
                        id={
                            "type": "search_bar_quick_field",
                            "index": _,
                            "text": __[0],
                        },
                        n_clicks=0,
                        className="ms-2 link text-decoration-none",
                    ),
                    " ",
                ]
            )
            for _, __ in enumerate(quick_fields)
        ]
        + [
            html.Span(
                dmc.Switch(
                    radius="xl",
                    size="sm",
                    offLabel=DashIconify(icon="radix-icons:id-card", width=15),
                    onLabel=DashIconify(icon="radix-icons:table", width=15),
                    color="orange",
                    checked=False,
                    persistence=True,
                    id="results_table_switch",
                ),
                className="float-end",
                title="Show results as cards or table",
            ),
        ],
        className="ps-4 pe-4 mb-0 mt-1",
    ),
] + [
    html.Div(
        # className='p-0 m-0 border shadow-sm rounded-3',
        className="pt-0 pb-0 ps-1 pe-1 m-0 rcorners2 shadow",
        id="search_bar",
        # className='rcorners2',
        children=[
            dbc.InputGroup(
                [
                    # History
                    dmc.Menu(
                        [
                            dmc.MenuTarget(
                                dmc.ActionIcon(
                                    DashIconify(icon="bi:clock-history"),
                                    color="gray",
                                    variant="transparent",
                                    radius="xl",
                                    size="lg",
                                    # title="Search history",
                                )
                            ),
                            dmc.MenuDropdown(
                                [
                                    dmc.MenuLabel("Search history is empty"),
                                ],
                                className="shadow rounded",
                                id="search_history_menu",
                            ),
                        ],
                        zIndex=1000000,
                    ),
                    # Main input
                    AutocompleteInput(
                        id="search_bar_input",
                        placeholder="Search, and you will find",
                        component="input",
                        trigger=[
                            "class:",
                            "class=",
                            "last:",
                            "last=",
                            "radius:",
                            "radius=",
                            "r:",
                            "r=",
                            "trend=",
                            "trend:",
                        ],
                        options={
                            "class:": fink_classes,
                            "class=": fink_classes,
                            "last:": ["1", "10", "100", "1000"],
                            "last=": ["1", "10", "100", "1000"],
                            "radius:": ["10", "60", "10m", "30m"],
                            "radius=": ["10", "60", "10m", "30m"],
                            "r:": ["10", "60", "10m", "30m"],
                            "r=": ["10", "60", "10m", "30m"],
                            "trend=": [
                                "rising",
                                "fading",
                                "low_state",
                                "new_low_state",
                            ],
                        },
                        maxOptions=0,
                        className="inputbar form-control border-0",
                        quoteWhitespaces=True,
                        autoFocus=True,
                        ignoreCase=True,
                        triggerInsideWord=False,
                        matchAny=True,
                    ),
                    # Clear
                    dmc.ActionIcon(
                        DashIconify(icon="mdi:clear-bold"),
                        n_clicks=0,
                        id="search_bar_clear",
                        color="gray",
                        variant="subtle",
                        radius="xl",
                        size="lg",
                        # title="Clear the input",
                    ),
                    # Submit
                    dbc.Spinner(
                        dmc.ActionIcon(
                            DashIconify(icon="tabler:search", width=20),
                            n_clicks=0,
                            id="search_bar_submit",
                            color="gray",
                            variant="transparent",
                            radius="xl",
                            size="lg",
                            loaderProps={"variant": "dots", "color": "orange"},
                            # title="Search",
                        ),
                        size="sm",
                        color="warning",
                    ),
                    # Help popup
                    help_popover(
                        [dcc.Markdown(message_help)],
                        "help_search",
                        trigger=dmc.ActionIcon(
                            DashIconify(icon="mdi:help"),
                            id="help_search",
                            color="gray",
                            variant="transparent",
                            radius="xl",
                            size="lg",
                            # className="d-none d-sm-flex"
                            # title="Show some help",
                        ),
                    ),
                ]
            ),
            # Search suggestions
            dbc.Collapse(
                dbc.ListGroup(
                    id="search_bar_suggestions",
                ),
                id="search_bar_suggestions_collapser",
                is_open=False,
            ),
            # Debounce timer
            dcc.Interval(
                id="search_bar_timer", interval=2000, max_intervals=1, disabled=True
            ),
            dcc.Store(
                id="search_history_store",
                storage_type="local",
            ),
        ],
    )
]

# Time-based debounce from https://joetatusko.com/2023/07/11/time-based-debouncing-with-plotly-dash/
clientside_callback(
    """
    function start_suggestion_debounce_timer(value, n_submit, n_clicks, n_intervals) {
        const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);
        if (triggered == 'search_bar_input.n_submit' || triggered == 'search_bar_submit.n_clicks')
            return [dash_clientside.no_update, true];

        if (n_intervals > 0)
            return [0, false];
        else
            return [dash_clientside.no_update, false];
    }
    """,
    [Output("search_bar_timer", "n_intervals"), Output("search_bar_timer", "disabled")],
    Input("search_bar_input", "value"),
    Input("search_bar_input", "n_submit"),
    Input("search_bar_submit", "n_clicks"),
    State("search_bar_timer", "n_intervals"),
    prevent_initial_call=True,
)


# Search history
@app.callback(
    Output("search_history_menu", "children"),
    Input("search_history_store", "timestamp"),
    Input("search_history_store", "data"),
)
def update_search_history_menu(timestamp, history):
    if history:
        return [
            dmc.MenuLabel("Search history"),
        ] + [
            dmc.MenuItem(
                item,
                id={
                    "type": "search_bar_completion",
                    "index": 1000 + i,
                    "text": str(item),
                },
            )
            for i, item in enumerate(history[::-1])
        ]
    else:
        return no_update


# Update suggestions on (debounced) input
@app.callback(
    Output("search_bar_suggestions", "children"),
    Output("search_bar_submit", "children"),
    Output("search_bar_suggestions_collapser", "is_open"),
    Input("search_bar_timer", "n_intervals"),
    Input("search_bar_input", "n_submit"),
    Input("search_bar_submit", "n_clicks"),
    # Next input uses dynamically created source, so has to be pattern-matching
    Input({"type": "search_bar_suggestion", "value": ALL}, "n_clicks"),
    State("search_bar_input", "value"),
    prevent_initial_call=True,
)
def update_suggestions(n_intervals, n_submit, n_clicks, s_n_clicks, value):
    # Clear the suggestions on submit by various means
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id in [
        "search_bar_input",
        "search_bar_submit",
        '{"type":"search_bar_suggestion","value":0}',
    ]:
        return no_update, no_update, False

    # Did debounce trigger fire?..
    if n_intervals != 1:
        return no_update, no_update, no_update

    if not value.strip():
        return None, no_update, False

    query = parse_query(value, timeout=5)
    suggestions = []

    params = query["params"]

    if not query["action"]:
        return None, no_update, False

    if query["action"] == "unknown":
        content = [html.Div(html.Em("Query not recognized", className="m-0"))]
    else:
        content = []

        if query["completions"]:
            completions = []

            for i, item in enumerate(query["completions"]):
                if isinstance(item, list) or isinstance(item, tuple):
                    # We expect it to be (name, ext)
                    name = item[0]
                    ext = item[1]
                else:
                    name = item
                    ext = item

                completions.append(
                    html.A(
                        ext,
                        id={"type": "search_bar_completion", "index": i, "text": name},
                        title=name,
                        n_clicks=0,
                        className="ms-2 link text-decoration-none",
                    )
                )

            suggestions.append(
                dbc.ListGroupItem(
                    html.Div(
                        [
                            html.Span("Did you mean:", className="text-secondary"),
                        ]
                        + completions
                    ),
                    className="border-bottom p-1 mt-1 small",
                )
            )

        content += [
            dmc.Group(
                [
                    html.Strong(query["object"]) if query["object"] else None,
                    dmc.Badge(query["type"], variant="outline", color="blue")
                    if query["type"]
                    else None,
                    dmc.Badge(query["action"], variant="outline", color="red"),
                ],
                wrap="wrap",
                align="left",
            ),
            html.P(query["hint"], className="m-0"),
        ]

    if len(params):
        content += [
            html.Small(" ".join(["{}={}".format(_, params[_]) for _ in params]))
        ]

    suggestion = dbc.ListGroupItem(
        content,
        action=True,
        n_clicks=0,
        # We make it pattern-matching so that it is possible to catch it in global callbacks
        id={"type": "search_bar_suggestion", "value": 0},
        className="border-0",
    )

    suggestions.append(suggestion)

    return suggestions, no_update, True


# Completion clicked
clientside_callback(
    """
    function on_completion(n_clicks) {
        const ctx = dash_clientside.callback_context;
        let triggered_id = ctx.triggered[0].prop_id;

        if (!ctx.triggered[0].value)
            return dash_clientside.no_update;

        if (triggered_id.search('.n_clicks') > 0) {
            triggered_id = JSON.parse(triggered_id.substr(0, triggered_id.indexOf('.n_clicks')));
            return triggered_id.text + ' ';
        }
        return dash_clientside.no_update;
    }
    """,
    Output("search_bar_input", "value", allow_duplicate=True),
    Input({"type": "search_bar_completion", "index": ALL, "text": ALL}, "n_clicks"),
    prevent_initial_call=True,
)

# Quick field clicked
clientside_callback(
    """
    function on_quickfield(n_clicks, value) {
        const ctx = dash_clientside.callback_context;
        let triggered_id = ctx.triggered[0].prop_id;

        if (!ctx.triggered[0].value)
            return dash_clientside.no_update;

        if (triggered_id.search('.n_clicks') > 0) {
            triggered_id = JSON.parse(triggered_id.substr(0, triggered_id.indexOf('.n_clicks')));
            if (value)
                return value + ' ' + triggered_id.text + '=';
            else
                return triggered_id.text + '=';
        }
        return dash_clientside.no_update;
    }
    """,
    Output("search_bar_input", "value", allow_duplicate=True),
    Input({"type": "search_bar_quick_field", "index": ALL, "text": ALL}, "n_clicks"),
    State("search_bar_input", "value"),
    prevent_initial_call=True,
)

# Clear inpit field
clientside_callback(
    """
    function on_clear(n_clicks) {
        return '';
    }
    """,
    Output("search_bar_input", "value", allow_duplicate=True),
    Input("search_bar_clear", "n_clicks"),
    prevent_initial_call=True,
)

# Disable clear button for empty input field
# clientside_callback(
#    """
#    function on_input(value) {
#        if (value)
#            return false;
#        else
#            return true;
#    }
#    """,
#    Output("search_bar_clear", "disabled"),
#    Input("search_bar_input", "value"),
# )


def display_table_results(table, query):
    """Display explorer results in the form of a table with a dropdown menu on top to insert more data columns.

    The dropdown menu options are taken from the client schema (ZTF & Fink). It also
    contains other derived fields from the portal (fink_additional_fields).

    Parameters
    ----------
    table: dash_table.DataTable
        Dash DataTable containing the results. Can be empty.
    query: dictionary
        Dictionary containing the parsed parameters from the search url.

    Returns
    -------
    out: list of objects
        The list of objects contain:
          1. A dropdown menu to add new columns in the table
          2. Table of results
        The dropdown is shown only if the table is non-empty.
        The fields available in the dropdown are restricted
        if the search is a cone search. See
        https://github.com/astrolabsoftware/fink-broker/blob/f45549110e9f13a1bdec44690b91c7125656dca2/fink_broker/ztf/hbase_utils.py#L302
    """
    conesearch_colnames = {
        "d:cdsxmatch",
        "d:classification",
        "d:nalerthist",
        "i:candid",
        "i:dec",
        "i:distnr",
        "i:drb",
        "i:fid",
        "i:jd",
        "i:jdstarthist",
        "i:magpsf",
        "i:objectId",
        "i:ra",
        "i:sigmapsf",
    }
    data = request_api("/api/v1/schema", method="GET", output="json")

    fink_fields = ["d:" + i for i in data["Fink science module outputs (d:)"].keys()]
    ztf_fields = ["i:" + i for i in data["ZTF original fields (i:)"].keys()]
    fink_additional_fields = [
        "v:constellation",
        "v:g-r",
        "v:rate(g-r)",
        "v:classification",
        "v:lastdate",
        "v:firstdate",
        "v:lapse",
    ]
    if query["action"] == "conesearch":
        fink_fields = list(set(fink_fields) & conesearch_colnames)
        ztf_fields = list(set(ztf_fields) & conesearch_colnames)
        fink_additional_fields = ["v:separation_degree", "v:lapse"]

    dropdown = dcc.Dropdown(
        id="field-dropdown2",
        options=[
            {"label": "Fink science module outputs", "disabled": True, "value": "None"},
            *[{"label": field, "value": field} for field in fink_fields],
            {"label": "Fink additional values", "disabled": True, "value": "None"},
            *[{"label": field, "value": field} for field in fink_additional_fields],
            {
                "label": "Original ZTF fields (subset)",
                "disabled": True,
                "value": "None",
            },
            *[{"label": field, "value": field} for field in ztf_fields],
        ],
        searchable=True,
        clearable=True,
        placeholder="Add more fields to the table",
    )

    switch = dmc.Switch(
        size="xs",
        radius="xl",
        label="Unique objects",
        color="orange",
        checked=False,
        id="alert-object-switch",
    )
    switch_description = "Toggle the switch to list each object only once. Only the latest alert will be displayed."

    switch_sso = dmc.Switch(
        size="xs",
        radius="xl",
        label="Unique SSO",
        color="orange",
        checked=False,
        id="alert-sso-switch",
    )
    switch_sso_description = "Toggle the switch to list each Solar System Object only once. Only the latest alert will be displayed."

    switch_tracklet = dmc.Switch(
        size="xs",
        radius="xl",
        label="Unique tracklets",
        color="orange",
        checked=False,
        id="alert-tracklet-switch",
    )
    switch_tracklet_description = "Toggle the switch to list each Tracklet only once (fast moving objects). Only the latest alert will be displayed."

    results = [
        dbc.Row(
            [
                dbc.Col(dropdown, lg=5, md=6),
                dbc.Col(
                    dmc.Tooltip(
                        children=switch,
                        w=220,
                        multiline=True,
                        withArrow=True,
                        transitionProps={"transition": "fade", "duration": 200},
                        label=switch_description,
                    ),
                    md="auto",
                ),
                dbc.Col(
                    dmc.Tooltip(
                        children=switch_sso,
                        w=220,
                        multiline=True,
                        withArrow=True,
                        transitionProps={"transition": "fade", "duration": 200},
                        label=switch_sso_description,
                    ),
                    md="auto",
                ),
                dbc.Col(
                    dmc.Tooltip(
                        children=switch_tracklet,
                        w=220,
                        multiline=True,
                        withArrow=True,
                        transitionProps={"transition": "fade", "duration": 200},
                        label=switch_tracklet_description,
                    ),
                    md="auto",
                ),
            ],
            align="center",
            justify="start",
            className="mb-2",
        ),
        table,
    ]

    return [
        html.Div(
            results,
            className="results-inner bg-opaque-100 rounded mb-4 p-2 border shadow",
            style={"overflow": "visible"},
        )
    ]


@app.callback(
    Output("aladin-lite-div-skymap", "run"),
    [
        Input("result_table", "data"),
        Input("result_table", "columns"),
        Input("modal_skymap", "is_open"),
    ],
)
def display_skymap(data, columns, is_open):
    """Display explorer result on a sky map (Aladin lite). Limited to 1000 sources total.

    TODO: image is not displayed correctly the first time

    the default parameters are:
        * PanSTARRS colors
        * FoV = 360 deg
        * Fink alerts overlayed

    Callbacks
    ----------
    Input: takes the validation flag (0: no results, 1: results) and table data
    Output: Display a sky image around the alert position from aladin.
    """
    if not is_open:
        return no_update

    if len(data) > 0:
        if len(data) > 1000:
            # Silently limit the size of list we display
            data = data[:1000]

        pdf = pd.DataFrame(data)

        # Coordinate of the first alert
        ra0 = pdf["i:ra"].to_numpy()[0]
        dec0 = pdf["i:dec"].to_numpy()[0]

        # Javascript. Note the use {{}} for dictionary
        # Force redraw of the Aladin lite window
        img = """var container = document.getElementById('aladin-lite-div-skymap');var txt = ''; container.innerHTML = txt;"""

        # Aladin lite
        img += """
        var a = A.aladin('#aladin-lite-div-skymap',
        {{
            target: '{} {}',
            survey: 'https://alasky.cds.unistra.fr/Pan-STARRS/DR1/color-z-zg-g/',
            showReticle: true,
            allowFullZoomout: true,
            showContextMenu: true,
            showCooGridControl: true,
            fov: 360
            }}
        );
        """.format(ra0, dec0)

        ras = pdf["i:ra"].to_numpy()
        decs = pdf["i:dec"].to_numpy()
        filts = pdf["i:fid"].to_numpy()
        filts_dic = {1: "g", 2: "r"}

        if "v:lastdate" not in pdf.columns:
            # conesearch does not expose v:lastdate
            pdf["v:lastdate"] = convert_jd(pdf["i:jd"])
        times = pdf["v:lastdate"].to_numpy()
        link = '<a target="_blank" href="{}/{}">{}</a>'
        titles = [
            link.format(
                config_args["SITEURL"],
                i.split("]")[0].split("[")[1],
                i.split("]")[0].split("[")[1],
            )
            for i in pdf["i:objectId"].to_numpy()
        ]
        mags = pdf["i:magpsf"].to_numpy()

        if "v:classification" not in pdf.columns:
            if "d:classification" in pdf.columns:
                pdf["v:classification"] = pdf["d:classification"]
            else:
                pdf["v:classification"] = "Unknown"
        classes = pdf["v:classification"].to_numpy()
        n_alert_per_class = (
            pdf.groupby("v:classification").count().to_dict()["i:objectId"]
        )
        cats = []
        for ra, dec, fid, time_, title, mag, class_ in zip(
            ras, decs, filts, times, titles, mags, classes
        ):
            if class_ in simbad_types:
                cat = "cat_{}".format(simbad_types.index(class_))
                color = class_colors["Simbad"]
            elif class_ in class_colors.keys():
                cat = "cat_{}".format(class_.replace(" ", "_"))
                color = class_colors[class_]
            else:
                # Sometimes SIMBAD mess up names :-)
                cat = "cat_{}".format(class_)
                color = class_colors["Simbad"]

            if cat not in cats:
                img += """var {} = A.catalog({{name: '{}', sourceSize: 15, shape: 'circle', color: '{}', onClick: 'showPopup', limit: 1000}});""".format(
                    cat, class_ + " ({})".format(n_alert_per_class[class_]), color
                )
                cats.append(cat)
            img += """{}.addSources([A.source({}, {}, {{objectId: '{}', mag: {:.2f}, filter: '{}', time: '{}', Classification: '{}'}})]);""".format(
                cat, ra, dec, title, mag, filts_dic[fid], time_, class_
            )

        for cat in sorted(cats):
            img += """a.addCatalog({});""".format(cat)

        # img cannot be executed directly because of formatting
        # We split line-by-line and remove comments
        img_to_show = [i for i in img.split("\n") if "// " not in i]

        return " ".join(img_to_show)
    else:
        return ""


def modal_skymap():
    import visdcc

    button = dmc.Button(
        "Sky Map",
        id="open_modal_skymap",
        n_clicks=0,
        leftSection=DashIconify(icon="bi:stars"),
        color="gray",
        fullWidth=True,
        variant="default",
        radius="xl",
    )

    modal = html.Div(
        [
            button,
            dbc.Modal(
                [
                    # loading(
                    dbc.ModalBody(
                        html.Div(
                            [
                                visdcc.Run_js(
                                    id="aladin-lite-div-skymap",
                                    style={"border": "0"},
                                ),
                            ],
                            style={
                                "width": "100%",
                                "height": "100%",
                            },
                        ),
                        className="p-1",
                        style={"height": "30pc"},
                    ),
                    # ),
                    dbc.ModalFooter(
                        dmc.Button(
                            "Close",
                            id="close_modal_skymap",
                            className="ml-auto",
                            color="gray",
                            # fullWidth=True,
                            variant="default",
                            radius="xl",
                        ),
                    ),
                ],
                id="modal_skymap",
                is_open=False,
                size="lg",
                # fullscreen="lg-down",
            ),
        ]
    )

    return modal


clientside_callback(
    """
    function toggle_modal_skymap(n1, n2, is_open) {
        if (n1 || n2)
            return ~is_open;
        else
            return is_open;
    }
    """,
    Output("modal_skymap", "is_open"),
    [Input("open_modal_skymap", "n_clicks"), Input("close_modal_skymap", "n_clicks")],
    [State("modal_skymap", "is_open")],
    prevent_initial_call=True,
)


def populate_result_table(data, columns):
    """Define options of the results table, and add data and columns"""
    page_size = 100
    markdown_options = {"link_target": "_blank"}

    table = dash_table.DataTable(
        data=data,
        columns=columns,
        id="result_table",
        page_size=page_size,
        # page_action='none',
        style_as_list_view=True,
        sort_action="native",
        filter_action="native",
        markdown_options=markdown_options,
        # fixed_columns={'headers': True, 'data': 1},
        style_data={
            "backgroundColor": "rgb(248, 248, 248, 1.0)",
        },
        style_table={"maxWidth": "100%", "overflowX": "scroll"},
        style_cell={
            "padding": "5px",
            "textAlign": "right",
            "overflow": "hidden",
            "font-family": "sans-serif",
            "fontSize": 14,
        },
        style_data_conditional=[
            {
                "if": {"column_id": "i:objectId"},
                "backgroundColor": "rgb(240, 240, 240, 1.0)",
            }
        ],
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        # Align the text in Markdown cells
        css=[dict(selector="p", rule="margin: 0; text-align: left")],
    )
    return table


@app.callback(
    [
        Output("result_table", "data"),
        Output("result_table", "columns"),
    ],
    [
        Input("field-dropdown2", "value"),
        Input("alert-object-switch", "checked"),
        Input("alert-sso-switch", "checked"),
        Input("alert-tracklet-switch", "checked"),
    ],
    [
        State("result_table", "data"),
        State("result_table", "columns"),
    ],
)
def update_table(field_dropdown, groupby1, groupby2, groupby3, data, columns):
    """Update table by adding new columns (no server call)"""
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    # Adding new columns (no client call)
    if "field-dropdown2" in changed_id:
        if field_dropdown is None or len(columns) == 0:
            raise PreventUpdate

        incolumns = any(c.get("id") == field_dropdown for c in columns)

        if incolumns is True:
            raise PreventUpdate

        columns.append(
            {
                "name": field_dropdown,
                "id": field_dropdown,
                "type": "numeric",
                "format": dash_table.Format.Format(precision=8),
                "presentation": "markdown"
                if field_dropdown == "i:objectId"
                else "input",
                # 'hideable': True,
            }
        )

        return data, columns
    elif groupby1 is True:
        pdf = pd.DataFrame.from_dict(data)
        pdf = pdf.drop_duplicates(subset="i:objectId", keep="first")
        data = pdf.to_dict("records")
        return data, columns
    elif groupby2 is True:
        pdf = pd.DataFrame.from_dict(data)
        if not np.all(pdf["i:ssnamenr"] == "null"):
            mask = ~pdf.duplicated(subset="i:ssnamenr") | (pdf["i:ssnamenr"] == "null")
            pdf = pdf[mask]
            data = pdf.to_dict("records")
        return data, columns
    elif groupby3 is True:
        pdf = pd.DataFrame.from_dict(data)
        if not np.all(pdf["d:tracklet"] == ""):
            mask = ~pdf.duplicated(subset="d:tracklet") | (pdf["d:tracklet"] == "")
            pdf = pdf[mask]
            data = pdf.to_dict("records")
        return data, columns
    else:
        raise PreventUpdate


# Prepare and display the results
@app.callback(
    [
        Output("results", "children"),
        Output("logo", "is_open"),
        Output("search_bar_submit", "children", allow_duplicate=True),
        Output("search_history_store", "data"),
    ],
    [
        Input("search_bar_input", "n_submit"),
        Input("search_bar_submit", "n_clicks"),
        # Next input uses dynamically created source, so has to be pattern-matching
        Input({"type": "search_bar_suggestion", "value": ALL}, "n_clicks"),
        Input("url", "search"),
    ],
    State("search_bar_input", "value"),
    State("search_history_store", "data"),
    State("results_table_switch", "checked"),
    # prevent_initial_call=True
    prevent_initial_call="initial_duplicate",
)
def results(n_submit, n_clicks, s_n_clicks, searchurl, value, history, show_table):
    """Parse the search string and query the database"""
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not triggered_id:
        # FIXME: ???
        triggered_id = "url"

    # Safeguards against triggering on initial mount of components
    if (
        (triggered_id == "search_bar_input" and not n_submit)
        or (triggered_id == "search_bar_submit" and not n_clicks)
        or (
            triggered_id == '{"type":"search_bar_suggestion","value":0}'
            and (not s_n_clicks or not s_n_clicks[0])
        )
        or (triggered_id == "url" and not searchurl)
    ):
        raise PreventUpdate

    if not value and not searchurl:
        # TODO: show back the logo?..
        return None, no_update, no_update, no_update

    colnames_to_display = {
        "i:objectId": "objectId",
        "i:ra": "RA (deg)",
        "i:dec": "Dec (deg)",
        "v:lastdate": "Last alert",
        "v:classification": "Classification",
        "i:ndethist": "Number of measurements",
        "v:lapse": "Time variation (day)",
    }

    if searchurl and triggered_id == "url":
        # Parse GET parameters from url
        params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(searchurl).query))
        # Construct the query from them
        query = {}
        for _ in ["action", "partial", "object"]:
            if _ in params:
                query[_] = params.pop(_)
        query["params"] = params
    else:
        value = value.strip()
        query = parse_query(value)

    if not query or not query["action"]:
        return None, no_update, no_update, no_update

    if query["action"] != "class" and "trend" in query["params"]:
        msg = "trend is experimental and can only be used with class search. Add the keyword `class=` to your search."
        return (
            dbc.Alert(msg, color="warning", className="shadow-sm"),
            no_update,
            no_update,
            history,
        )

    if "last" in query["params"] and query["action"] == "unknown":
        msg = "last must be used with class search. Add the keyword `class=` to your search."
        return (
            dbc.Alert(msg, color="warning", className="shadow-sm"),
            no_update,
            no_update,
            history,
        )

    if query["action"] == "unknown":
        return (
            dbc.Alert(
                "Query not recognized: {}".format(value),
                color="danger",
                className="shadow-sm",
            ),
            no_update,
            no_update,
            no_update,
        )

    elif query["action"] == "objectid":
        # Search objects by objectId
        msg = "ObjectId search with {} name {}".format(
            "partial" if query.get("partial") else "exact", query["object"]
        )
        pdf = request_api(
            "/api/v1/objects",
            json={
                "objectId": query["object"],
            },
        )

    elif query["action"] == "sso":
        # Solar System Objects
        msg = "Solar System object search with ssnamenr {}".format(
            query["params"]["sso"]
        )
        pdf = request_api("/api/v1/sso", json={"n_or_d": query["params"]["sso"]})

    elif query["action"] == "tracklet":
        # Tracklet by (partial) name
        msg = "Tracklet search with {} name {}".format(
            "partial" if query.get("partial") else "exact", query["object"]
        )
        payload = {"id": query["object"]}

        pdf = request_api("/api/v1/tracklet", json=payload)

    elif query["action"] == "conesearch":
        # Conesearch
        ra = float(query["params"].get("ra"))
        dec = float(query["params"].get("dec"))
        # Default is 10 arcsec, max is 5 degrees
        sr = min(float(query["params"].get("r", 10)), 18000)

        msg = (
            "Cone search with center at {:.4f} {:.3f} and radius {:.1f} arcsec".format(
                ra, dec, sr
            )
        )

        payload = {
            "ra": ra,
            "dec": dec,
            "radius": sr,
        }

        if "after" in query["params"]:
            startdate = isoify_time(query["params"]["after"])

            msg += " after {}".format(startdate)

            payload["startdate"] = startdate

        if "before" in query["params"]:
            stopdate = isoify_time(query["params"]["before"])

            msg += " before {}".format(stopdate)

            payload["stopdate"] = stopdate

        elif "window" in query["params"]:
            window = query["params"]["window"]

            msg += " window {} days".format(window)

            payload["window"] = window

        pdf = request_api("/api/v1/conesearch", json=payload)

        colnames_to_display = {
            "i:objectId": "objectId",
            "v:separation_degree": "Separation (degree)",
            "d:classification": "Classification",
            "d:nalerthist": "Number of measurements",
            "v:lapse": "Time variation (day)",
        }

    elif query["action"] == "class":
        # Class-based search
        alert_class = query["params"].get("class")

        n_last = int(query["params"].get("last", 100))

        msg = "Last {} objects with class '{}'".format(n_last, alert_class)

        payload = {"class": alert_class, "n": n_last}

        if "after" in query["params"]:
            startdate = isoify_time(query["params"]["after"])

            msg += " after {}".format(startdate)

            payload["startdate"] = startdate

        if "before" in query["params"]:
            stopdate = isoify_time(query["params"]["before"])

            msg += " before {}".format(stopdate)

            payload["stopdate"] = stopdate

        if "trend" in query["params"]:
            msg += " and {} trend".format(query["params"]["trend"])
            payload["trend"] = query["params"]["trend"]

        pdf = request_api("/api/v1/latests", json=payload)

    elif query["action"] == "anomaly":
        # Anomaly search
        n_last = int(query["params"].get("last", 100))

        msg = "Last {} anomalies".format(n_last)

        payload = {"n": n_last}

        if "after" in query["params"]:
            startdate = isoify_time(query["params"]["after"])

            msg += " after {}".format(startdate)

            payload["start_date"] = startdate

        if "before" in query["params"]:
            stopdate = isoify_time(query["params"]["before"])

            msg += " before {}".format(stopdate)

            payload["stop_date"] = stopdate

        pdf = request_api("/api/v1/anomaly", json=payload)

    else:
        return (
            dbc.Alert(
                "Unhandled query: {}".format(query),
                color="danger",
                className="shadow-sm",
            ),
            no_update,
            no_update,
            no_update,
        )

    # Add to history
    if not history:
        history = []
    while value in history:
        history.remove(value)  # Remove duplicates
    history.append(value)
    history = history[-10:]  # Limit it to 10 latest entries

    msg = "{} - {} found".format(
        msg,
        "nothing"
        if pdf.empty
        else ("1 alert" if len(pdf.index) == 1 else str(len(pdf.index)) + " alerts"),
    )

    if pdf.empty:
        # text, header = text_noresults(query, query_type, dropdown_option, searchurl)
        return (
            dbc.Alert(msg, color="warning", className="shadow-sm"),
            no_update,
            no_update,
            history,
        )
    else:
        # Make clickable objectId
        pdf["i:objectId"] = pdf["i:objectId"].apply(markdownify_objectid)

        # Sort the results
        if query["action"] == "conesearch":
            pdf["v:lapse"] = pdf["i:jd"] - pdf["i:jdstarthist"]
            data = pdf.sort_values("v:separation_degree", ascending=True)
        else:
            data = pdf.sort_values("i:jd", ascending=False)

        if show_table:
            data = data.to_dict("records")

            columns = [
                {
                    "id": c,
                    "name": colnames_to_display[c],
                    "type": "numeric",
                    "format": dash_table.Format.Format(precision=8),
                    # 'hideable': True,
                    "presentation": "markdown" if c == "i:objectId" else "input",
                }
                for c in colnames_to_display.keys()
            ]

            table = populate_result_table(data, columns)
            results_ = display_table_results(table, query)
        else:
            results_ = display_cards_results(pdf)

        results = [
            # Common header for the results
            dbc.Row(
                [
                    dbc.Col(msg, md="auto"),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(modal_skymap(), xs="auto"),
                                dbc.Col(
                                    help_popover(
                                        dcc.Markdown(msg_info),
                                        id="help_msg_info",
                                        trigger=dmc.ActionIcon(
                                            DashIconify(icon="mdi:help"),
                                            id="help_msg_info",
                                            color="gray",
                                            variant="default",
                                            radius="xl",
                                            size="lg",
                                        ),
                                    ),
                                    xs="auto",
                                ),
                            ],
                            justify="end",
                        ),
                        md="auto",
                    ),
                ],
                align="end",
                justify="between",
                className="m-2",
            ),
        ] + results_

        return results, False, no_update, history


def display_cards_results(pdf, page_size=10):
    results_ = [
        # Data storage
        dcc.Store(
            id="results_store",
            storage_type="memory",
            data=pdf.to_json(),
        ),
        dcc.Store(
            id="results_page_size_store",
            storage_type="memory",
            data=str(page_size),
        ),
        # For Aladin
        dcc.Store(
            id="result_table",
            storage_type="memory",
            data=pdf.to_dict("records"),
        ),
        # Actual display of results
        html.Div(id="results_paginated"),
    ]

    npages = int(np.ceil(len(pdf.index) / page_size))
    results_ += [
        dmc.Space(h=10),
        dmc.Group(
            dmc.Pagination(
                id="results_pagination",
                value=1,
                total=npages,
                siblings=1,
                withControls=True,
                withEdges=True,
            ),
            align="center",
            justify="center",
            className="d-none" if npages == 1 else "",
        ),
        dmc.Space(h=20),
    ]

    return results_


@app.callback(
    Output("results_paginated", "children"),
    Input("results_pagination", "value"),
    State("results_store", "data"),
    State("results_page_size_store", "data"),
)
def on_paginate(page, data, page_size):
    pdf = pd.read_json(io.StringIO(data))
    page_size = int(page_size)

    if not page:
        page = 1

    results = []

    # Slice to selected page
    pdf_ = pdf.iloc[(page - 1) * page_size : min(page * page_size, len(pdf.index))]

    for i, row in pdf_.iterrows():
        results.append(card_search_result(row, i))

    return results


# Scroll to top on paginate
clientside_callback(
    """
    function scroll_top(value) {
        document.querySelector('#search_bar').scrollIntoView({behavior: "smooth"})
        return dash_clientside.no_update;
    }
    """,
    Output("results_pagination", "page"),  # Fake output!!!
    Input("results_pagination", "page"),
    prevent_initial_call=True,
)


@app.callback(
    Output(
        {"type": "search_results_lightcurve", "objectId": MATCH, "index": MATCH},
        "children",
    ),
    Input(
        {"type": "search_results_lightcurve", "objectId": MATCH, "index": MATCH}, "id"
    ),
)
def on_load_lightcurve(lc_id):
    if lc_id:
        # print(lc_id['objectId'])
        fig = draw_lightcurve_preview(lc_id["objectId"])
        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": False},
            style={"width": "100%", "height": "15pc"},
            responsive=True,
        )

    return no_update


@app.callback(
    Output(
        {"type": "search_results_cutouts", "objectId": MATCH, "index": MATCH},
        "children",
    ),
    Input({"type": "search_results_cutouts", "objectId": MATCH, "index": MATCH}, "id"),
)
def on_load_cutouts(lc_id):
    if lc_id:
        return html.Div(
            draw_cutouts_quickview(lc_id["objectId"]),
            style={"width": "12pc", "height": "12pc"},
        )
        # figs = draw_cutouts_quickview(lc_id['objectId'], ['science', 'template', 'difference'])
        # return dbc.Row(
        #     [
        #         dbc.Col(_, xs=4, className='p-0') for _ in figs
        #     ],
        #     justify='center',
        #     className='p-0',
        # )

    return no_update


clientside_callback(
    """
    function drawer_switch(n_clicks, pathname) {
        const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);

        /* Change the page title based on its path */
        if (triggered == 'url.pathname') {
            let title = 'Fink Science Portal';

            if (pathname.startsWith('/ZTF'))
                title = pathname.substring(1, 13) + ' : ' + title;
            else if (pathname.startsWith('/gw'))
                title = 'Gravitational Waves : ' + title;
            else if (pathname.startsWith('/download'))
                title = 'Data Transfer : ' + title;
            else if (pathname.startsWith('/stats'))
                title = 'Statistics : ' + title;
            else if (pathname.startsWith('/api'))
                title = 'API : ' + title;

            document.title = title;
        }

        if (triggered == 'drawer-button.n_clicks')
            return true;
        else
            return false;
    }
    """,
    Output("drawer", "opened"),
    Input("drawer-button", "n_clicks"),
    Input("url", "pathname"),
    prevent_initial_call=True,
)

navbar = dmc.AppShellHeader(
    id="navbar",
    zIndex=1000,
    p=0,
    m=0,
    className="shadow-sm",
    children=[
        dmc.Space(h=10),
        dmc.Container(
            fluid=True,
            children=dmc.Group(
                # align="stretch",
                justify="space-between",
                children=[
                    # Right menu
                    dmc.Group(
                        align="start",
                        justify="flex-start",
                        children=[
                            # Burger
                            dmc.ActionIcon(
                                DashIconify(icon="dashicons:menu", width=30),
                                id="drawer-button",
                                n_clicks=0,
                                variant="transparent",
                                style={"color": "gray"},
                            ),
                            dmc.Anchor(
                                dmc.Group(
                                    [
                                        dmc.ThemeIcon(
                                            DashIconify(
                                                icon="ion:search-outline",
                                                width=22,
                                            ),
                                            radius=30,
                                            size=32,
                                            variant="outline",
                                            color="gray",
                                        ),
                                        dmc.Text(
                                            "Search",
                                            visibleFrom="md",
                                            style={"color": "gray"},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                href="/",
                                variant="text",
                                style={
                                    "textTransform": "capitalize",
                                    "textDecoration": "none",
                                },
                            ),
                            dmc.Anchor(
                                dmc.Group(
                                    [
                                        dmc.ThemeIcon(
                                            DashIconify(
                                                icon="ion:cloud-download-outline",
                                                width=22,
                                            ),
                                            radius=30,
                                            size=32,
                                            variant="outline",
                                            color="gray",
                                        ),
                                        dmc.Text(
                                            "Data Transfer",
                                            visibleFrom="md",
                                            style={"color": "gray"},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                href="/download",
                                variant="text",
                                style={
                                    "textTransform": "capitalize",
                                    "textDecoration": "none",
                                },
                            ),
                            dmc.Anchor(
                                dmc.Group(
                                    [
                                        dmc.ThemeIcon(
                                            DashIconify(
                                                icon="material-symbols:join-right",
                                                width=22,
                                            ),
                                            radius=30,
                                            size=32,
                                            variant="outline",
                                            color="gray",
                                        ),
                                        dmc.Text(
                                            "Xmatch",
                                            visibleFrom="md",
                                            style={"color": "gray"},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                href="/xmatch",
                                variant="text",
                                style={
                                    "textTransform": "capitalize",
                                    "textDecoration": "none",
                                },
                            ),
                            dmc.Anchor(
                                dmc.Group(
                                    [
                                        dmc.ThemeIcon(
                                            DashIconify(
                                                icon="ion:infinite-outline",
                                                width=22,
                                            ),
                                            radius=30,
                                            size=32,
                                            variant="outline",
                                            color="gray",
                                        ),
                                        dmc.Text(
                                            "Gravitational Waves",
                                            visibleFrom="md",
                                            style={"color": "gray"},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                href="/gw",
                                variant="text",
                                style={
                                    "textTransform": "capitalize",
                                    "textDecoration": "none",
                                },
                            ),
                        ],
                    ),
                    # Left menu
                    dmc.Group(
                        align="end",
                        justify="flex-end",
                        children=[
                            dmc.Anchor(
                                dmc.Group(
                                    [
                                        dmc.ThemeIcon(
                                            DashIconify(
                                                icon="ion:stats-chart-outline",
                                                width=22,
                                            ),
                                            radius=30,
                                            size=32,
                                            variant="outline",
                                            color="gray",
                                        ),
                                        dmc.Text(
                                            "Statistics",
                                            visibleFrom="md",
                                            style={"color": "gray"},
                                        ),
                                    ],
                                    gap="xs",
                                ),
                                href="/stats",
                                variant="text",
                                style={
                                    "textTransform": "capitalize",
                                    "textDecoration": "none",
                                },
                            ),
                        ],
                    ),
                    # Sidebar
                    dmc.Drawer(
                        children=[
                            dmc.Divider(
                                labelPosition="left",
                                label=[
                                    DashIconify(
                                        icon="tabler:search",
                                        width=15,
                                        style={"marginRight": 10},
                                    ),
                                    "Explore",
                                ],
                                style={"marginTop": 20, "marginBottom": 20},
                            ),
                            dmc.Stack(
                                [
                                    dmc.Anchor(
                                        "Search",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="/",
                                        size="sm",
                                    ),
                                    dmc.Anchor(
                                        "Data Transfer",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="/download",
                                        size="sm",
                                    ),
                                    dmc.Anchor(
                                        "Gravitational Waves",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="/gw",
                                        size="sm",
                                    ),
                                    dmc.Anchor(
                                        "Statistics",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="/stats",
                                        size="sm",
                                    ),
                                ],
                                align="left",
                                gap="sm",
                                style={"paddingLeft": 30, "paddingRight": 20},
                            ),
                            dmc.Divider(
                                labelPosition="left",
                                label=[
                                    DashIconify(
                                        icon="carbon:api",
                                        width=15,
                                        style={"marginRight": 10},
                                    ),
                                    "Learn",
                                ],
                                style={"marginTop": 20, "marginBottom": 20},
                            ),
                            dmc.Stack(
                                [
                                    dmc.Anchor(
                                        "{ API }",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="https://doc.ztf.fink-broker.org/en/latest/services/search/getting_started/#quick-start-api",
                                        size="sm",
                                    ),
                                    dmc.Anchor(
                                        "Tutorials",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="https://github.com/astrolabsoftware/fink-tutorials",
                                        size="sm",
                                    ),
                                ],
                                align="left",
                                gap="sm",
                                style={"paddingLeft": 30, "paddingRight": 20},
                            ),
                            dmc.Divider(
                                labelPosition="left",
                                label=[
                                    DashIconify(
                                        icon="tabler:external-link",
                                        width=15,
                                        style={"marginRight": 10},
                                    ),
                                    "External links",
                                ],
                                style={"marginTop": 20, "marginBottom": 20},
                            ),
                            dmc.Stack(
                                [
                                    dmc.Anchor(
                                        "Fink broker",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="https://fink-broker.org",
                                        size="sm",
                                        # color="gray",
                                    ),
                                    dmc.Anchor(
                                        "Portal bug tracker",
                                        style={
                                            "textTransform": "capitalize",
                                            "textDecoration": "none",
                                            "color": "gray",
                                        },
                                        href="https://github.com/astrolabsoftware/fink-science-portal",
                                        size="sm",
                                    ),
                                ],
                                align="left",
                                gap="sm",
                                style={"paddingLeft": 30, "paddingRight": 20},
                            ),
                        ],
                        title="Fink Science Portal",
                        id="drawer",
                        padding="md",
                        zIndex=1e7,
                        transitionProps={"transition": "pop-top-left"},
                        style={"fontColor": "gray"},
                    ),
                    # dmc.ThemeSwitcher(
                    #     id="color-scheme-toggle",
                    #     style={"cursor": "pointer"},
                    # ),
                ],
            ),
        ),
    ],
)

# embedding the navigation bar
app.layout = dmc.MantineProvider(
    [
        dcc.Location(id="url", refresh=False),
        dmc.AppShell(
            children=[
                navbar,
                dmc.AppShellMain(
                    children=[],
                    id="page-content",
                    style={"padding-top": "55px"},  # header
                ),
            ],
            header={"height": 55},
        ),
    ],
    # forceColorScheme="dark"
)


@app.callback(
    [
        Output("page-content", "children"),
        Output("page-content", "className"),
    ],
    [
        Input("url", "pathname"),
        Input("url", "search"),
    ],
)
def display_page(pathname, searchurl):
    layout = dmc.MantineProvider(
        [
            dbc.Container(
                [
                    # Logo shown by default
                    dbc.Collapse(
                        # dmc.MediaQuery(
                        dbc.Row(
                            dbc.Col(
                                html.Img(
                                    src="/assets/Fink_PrimaryLogo_WEB.png",
                                    height="100%",
                                    width="40%",
                                    style={"min-width": "250px"},
                                )
                            ),
                            style={"textAlign": "center"},
                            className="mt-3",
                        ),
                        # query="(max-height: 400px) or (max-width: 300px)",
                        # styles={"display": "none"},
                        # ),
                        is_open=True,
                        id="logo",
                    ),
                    dbc.Row(
                        dbc.Col(
                            fink_search_bar,
                            lg={"size": 8, "offset": 2},
                            md={"size": 10, "offset": 1},
                        ),
                        className="mt-3 mb-3",
                    ),
                ],
                fluid="lg",
            ),
            dbc.Container(
                # Default content for results part - search history
                dbc.Row(
                    dbc.Col(
                        id="search_history",
                        # Size should match the one of fink_search_bar above
                        lg={"size": 8, "offset": 2},
                        md={"size": 10, "offset": 1},
                    ),
                    className="m-3",
                ),
                id="results",
                fluid="xxl",
            ),
        ],
        # forceColorScheme="dark"
    )
    if pathname == "/about":
        return about.layout, "home"
    elif pathname == "/stats":
        return statistics.layout(), "home_light"
    elif pathname == "/download":
        return query_cluster.layout(), "home_light"
    elif pathname == "/gw":
        return gw.layout(), "home_light"
    elif pathname == "/xmatch":
        return xmatch.layout(), "home_light"
    elif pathname.startswith("/ZTF"):
        return summary.layout(pathname), "home_light"
    else:
        if pathname[1:]:
            # check this is not a name generated by a user
            oid = retrieve_oid_from_metaname(pathname[1:])
            if oid is not None:
                return summary.layout("/" + oid), "home_light"
        return layout, "home"


server.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
server.config["JSON_SORT_KEYS"] = False

if __name__ == "__main__":
    app.run_server(config_args["HOST"], debug=True, port=config_args["PORT"])
