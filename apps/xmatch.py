# Copyright 2025 AstroLab Software
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
import dash_mantine_components as dmc

# import dash_core_components as dcc
import dash_bootstrap_components as dbc

import textwrap
import base64
import datetime
import io
import json
import os
import re

import numpy as np
import pandas as pd
import requests
import yaml
from dash import (
    Input,
    Output,
    State,
    html,
    dcc,
    ctx,
    callback,
    no_update,
    dash_table,
    clientside_callback,
)
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from astropy.io import votable
import astropy.units as u
from mocpy import MOC


from app import app
from apps.mining.utils import (
    submit_spark_job,
    upload_file_hdfs,
)
from apps.utils import extract_configuration
from apps.utils import format_field_for_data_transfer
from apps.utils import create_datatransfer_schema_table
from apps.plotting import COLORS_ZTF


args = extract_configuration("config.yml")
APIURL = args["APIURL"]

min_step = 0
max_step = 4

MAX_ROW = 100000


@app.callback(
    Output("radius-xmatch", "style"),
    Output("radius-column", "style"),
    Output("radius-xmatch", "value"),
    Output("radius-column", "value"),
    Input("radius-switch", "checked"),
    State("store-radius-number", "data"),
    State("store-radius-column", "data"),
)
def render_field(checked, number_store, select_store):
    num_val = number_store.get("value") if isinstance(number_store, dict) else None
    sel_val = select_store.get("value") if isinstance(select_store, dict) else None

    if not checked:
        return (
            {"display": "block"},
            {"display": "none"},
            num_val,
            sel_val,
        )
    else:
        return (
            {"display": "none"},
            {"display": "block"},
            num_val,
            sel_val,
        )


@app.callback(
    Output("store-radius-number", "data"),
    Output("store-radius-column", "data"),
    Input("radius-xmatch", "value"),
    Input("radius-column", "value"),
    State("store-radius-number", "data"),
    State("store-radius-column", "data"),
    prevent_initial_call=True,
)
def update_stores(number_val, select_val, num_store, sel_store):
    if num_store is None:
        num_store = {"value": None}
    if sel_store is None:
        sel_store = {"value": None}

    if number_val is not None:
        num_store["value"] = number_val
    if select_val is not None:
        sel_store["value"] = select_val

    return num_store, sel_store


def upload_catalog():
    """ """
    radius_number = dmc.NumberInput(
        placeholder="type value...",
        label="Crossmatch Radius (in arcsecond)",
        variant="default",
        hideControls=True,
        id="radius-xmatch",
        w=250,
        mb=10,
        disabled=True,
        style={"display": "none"},
    )
    radius_select = dmc.Select(
        label="Column for Radius (in arcsecond)",
        placeholder="Select one",
        id="radius-column",
        w=250,
        mb=10,
        disabled=True,
        style={"display": "none"},
    )
    radius_stack = dmc.Stack(
        [
            dcc.Store(id="store-radius-number", data={"value": None}),
            dcc.Store(id="store-radius-column", data={"value": None}),
            dmc.HoverCard(
                withArrow=True,
                position='top',
                width=250,
                shadow="md",
                children=[
                    dmc.HoverCardTarget(
                        dmc.Switch(
                            id="radius-switch",
                            onLabel=dmc.Text(
                                "Variable",
                                size="sm",
                                fw=400,
                            ),
                            offLabel=dmc.Text(
                                "Unique",
                                size="sm",
                                fw=400,
                            ),
                            size="lg",
                            mb=10,
                            checked=False,
                            withThumbIndicator=False,
                        )
                    ),
                    dmc.HoverCardDropdown(
                        dmc.Text(
                            """
Choose Unique if you want the same radius for all of the coordinates in your cone search.
Choose Variable if your dataset has a column for specific radiuses per sources.
                            """,
                            size="sm",
                            ta="center",
                        )
                    ),
                ],
            ),
            radius_number,
            radius_select,
        ],
        gap="xs",
        align="center",
    )

    ra = dmc.Select(
        label="Column for Right Ascension (J2000)",
        placeholder="Select one",
        id="ra-column",
        w=250,
        mb=10,
        disabled=True,
    )
    dec = dmc.Select(
        label="Column for Declination (J2000)",
        placeholder="Select one",
        id="dec-column",
        w=250,
        mb=10,
        disabled=True,
    )
    identifier = dmc.Select(
        label="Select column for the Identifier",
        placeholder="Select one",
        id="id-column",
        w=250,
        mb=10,
        disabled=True,
    )

    return html.Div(
        children=[
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    [
                        "Drag and Drop or ",
                        html.A("Select Files "),
                        "(csv, parquet, or votable)",
                    ]
                ),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
            ),
            html.Div(id="output-data-upload"),
            dmc.Space(h=10),
            dmc.Group(
                [ra, dec, identifier, radius_stack], justify="center", align="flex-end"
            ),
            dmc.Space(h=10),
            dmc.Center(modal_skymap()),
        ]
    )


def date_tab():
    options = html.Div(
        [
            dmc.YearPickerInput(
                type="range",
                id="date-range-picker-xmatch",
                label="Fink/ZTF alert date range",
                description="Pick up one or several years of Fink/ZTF data to crossmatch against",
                hideOutsideDates=True,
                numberOfColumns=2,
                dropdownType="modal",
                modalProps={"centered": True},
                minDate="2019-11-02",
                maxDate=(datetime.datetime.now().date() - datetime.timedelta(days=1)),
                allowSingleDateInRange=True,
                required=True,
                clearable=True,
            ),
        ]
    )
    tab = html.Div(
        [
            dmc.Space(h=50),
            options,
        ],
        id="date_tab-xmatch",
    )
    return tab


def filter_content_tab():
    options = html.Div(
        [
            dmc.MultiSelect(
                label="Alert fields",
                description="Select all Fink/ZTF fields you would like to be added.",
                placeholder="start typing...",
                id="field_select_xmatch",
                data=format_field_for_data_transfer(
                    "ZTF", with_predefined_options=False, cutouts_allowed=False
                ),
                searchable=True,
                clearable=True,
            ),
            dmc.Accordion(
                id="accordion-schema",
                children=[
                    dmc.AccordionItem(
                        [
                            dmc.AccordionControl(
                                "Alert schema",
                                icon=DashIconify(
                                    icon="tabler:help",
                                    color=dmc.DEFAULT_THEME["colors"]["blue"][6],
                                    width=20,
                                ),
                            ),
                            dmc.AccordionPanel(
                                create_datatransfer_schema_table(cutouts_allowed=False)
                            ),
                        ],
                        value="info",
                    ),
                ],
            ),
            dmc.Space(h=10),
        ],
    )
    tab = html.Div(
        [
            dmc.Space(h=50),
            dmc.Divider(variant="solid", label="Filter alert content"),
            options,
        ],
        id="filter_content_tab_xmatch",
    )
    return tab


@app.callback(
    Output("code_block_xmatch", "code"),
    Input("topic_name_xmatch", "children"),
    prevent_initial_call=True,
)
def update_code_block(topic_name):
    if topic_name is not None and topic_name != "":
        code_block = f"""
fink_datatransfer \\
    -topic {topic_name} \\
    -outdir {topic_name} \\
    --verbose \\
    -survey ztf
        """
        return code_block


@app.callback(
    Output("submit_xmatch", "disabled", allow_duplicate=True),
    Output("notification-container-xmatch", "children"),
    Output("batch_id_xmatch", "children"),
    Output("topic_name_xmatch", "children"),
    [
        Input("submit_xmatch", "n_clicks"),
        Input("radius-switch", "checked"),
    ],
    [
        State("object-catalog", "data"),
        State("upload-data", "filename"),
        State("ra-column", "value"),
        State("dec-column", "value"),
        State("store-radius-number", "data"),
        State("store-radius-column", "data"),
        State("id-column", "value"),
        State("date-range-picker-xmatch", "value"),
        State("field_select_xmatch", "value"),
    ],
    prevent_initial_call=True,
)
def submit_job(
    n_clicks,
    radius_switch,
    catalog,
    catalog_filename,
    ra,
    dec,
    radius,
    radius_col,
    identifier,
    date_range_picker,
    field_select,
):
    """Submit a job to the Apache Spark cluster via Livy"""
    if n_clicks:
        # define unique topic name
        d = datetime.datetime.utcnow()

        topic_name = f"fxmatch_ztf_{d.date().isoformat()}_{d.microsecond}"
        fn = "assets/spark_ztf_xmatch.py"
        basepath = "hdfs://vdmaster1:8020/user/julien.peloton/archive/science"

        filename = f"stream_{topic_name}.py"

        with open(fn) as f:
            data = f.read()
        code = textwrap.dedent(data)

        input_args = yaml.load(open("config_datatransfer.yml"), yaml.Loader)
        status_code, hdfs_log = upload_file_hdfs(
            code,
            input_args["WEBHDFS"],
            input_args["NAMENODE"],
            input_args["USER"],
            filename,
        )

        if status_code != 201:
            text = dmc.Stack(
                children=[
                    "Unable to upload {} on HDFS, with error: ".format(filename),
                    dmc.CodeHighlight(code=f"{hdfs_log}", language="html"),
                    "Contact an administrator at contact@fink-broker.org.",
                ]
            )
            alert = dmc.Alert(
                children=text, title=f"[Status code {status_code}]", color="red"
            )
            return True, alert, no_update, no_update

        # Send the data to HDFS as parquet file
        catalog_filename_parquet = os.path.splitext(catalog_filename)[0] + ".parquet"

        # Conversion in decimal degree as xmatch expects it
        pdf = pd.read_json(io.StringIO(catalog))
        pdf[ra], pdf[dec] = enforce_decimal(pdf, ra, dec)

        status_code, hdfs_log = upload_file_hdfs(
            pdf.to_parquet(),
            input_args["WEBHDFS"],
            input_args["NAMENODE"],
            input_args["USER"],
            catalog_filename_parquet,
        )

        if status_code != 201:
            text = dmc.Stack(
                children=[
                    "Unable to upload {} on HDFS, with error: ".format(
                        catalog_filename_parquet
                    ),
                    dmc.CodeHighlight(code=f"{hdfs_log}", language="html"),
                    "Contact an administrator at contact@fink-broker.org.",
                ]
            )
            alert = dmc.Alert(
                children=text, title=f"[Status code {status_code}]", color="red"
            )
            return True, alert, no_update, no_update

        # get the job args
        job_args = [
            f"-startDate={date_range_picker[0]}",
            f"-stopDate={date_range_picker[1]}",
            f"-basePath={basepath}",
            f"-topic_name={topic_name}",
            f"-ra_col={ra}",
            f"-dec_col={dec}",
            f"-radius_arcsec={radius_col if radius_switch else radius}",
            f"-id_col={identifier}",
            "-catalog_filename={}".format(catalog_filename_parquet),
            "-kafka_bootstrap_servers={}".format(input_args["KAFKA_BOOTSTRAP_SERVERS"]),
            "-kafka_sasl_username={}".format(input_args["KAFKA_SASL_USERNAME"]),
            "-kafka_sasl_password={}".format(input_args["KAFKA_SASL_PASSWORD"]),
            "-path_to_tns=/spark_mongo_tmp/julien.peloton/tns.parquet",
        ]
        if field_select is not None:
            [job_args.append(f"-ffield={elem}") for elem in field_select]

        # submit the job
        filepath = "hdfs://vdmaster1:8020/user/{}/{}".format(
            input_args["USER"], filename
        )
        batchid, status_code, spark_log = submit_spark_job(
            input_args["LIVYHOST"],
            filepath,
            input_args["SPARKCONF"],
            job_args,
        )

        if status_code != 201:
            text = dmc.Stack(
                children=[
                    "Unable to upload resources on HDFS, with error: ",
                    dmc.CodeHighlight(code=f"{spark_log}", language="html"),
                    "Contact an administrator at contact@fink-broker.org.",
                ]
            )
            alert = dmc.Alert(
                children=text,
                title=f"[Batch ID {batchid}][Status code {status_code}]",
                color="red",
            )
            return True, alert, no_update, no_update

        alert = dmc.Alert(
            children=f"Your topic name is: {topic_name}",
            title="Submitted successfully",
            color="green",
        )
        if n_clicks:
            return True, alert, batchid, topic_name
        else:
            return False, alert, batchid, topic_name
    else:
        return no_update, no_update, no_update, no_update


@app.callback(
    Output("batch_log_xmatch", "children"),
    [
        Input("batch_id_xmatch", "children"),
        Input("interval-component-xmatch", "n_intervals"),
    ],
)
def update_log(batchid, interval):
    """Update log from the Spark cluster"""
    if batchid != "":
        response = requests.get(f"http://vdmaster1:21111/batches/{batchid}/log")

        if "log" in response.json():
            bad_words = ["Error", "Traceback"]
            failure_log = [
                row
                for row in response.json()["log"]
                if np.any([i in row for i in bad_words])
            ]
            if len(failure_log) > 0:
                initial_traceback = failure_log[0]
                log = response.json()["log"]
                index = log.index(initial_traceback)
                failure_msg = [
                    f"Batch ID: {batchid}",
                    "Failed. Please, contact contact@fink-broker.org with your batch ID and the message below.",
                    "------------- Traceback -------------",
                    *log[index:],
                ]
                output = html.Div(
                    "\n".join(failure_msg), style={"whiteSpace": "pre-wrap"}
                )
                return output
            # catch and return tailored error msg if fail (with batchid and contact@fink-broker.org)
            livy_log = [row for row in response.json()["log"] if "-Livy-" in row]
            livy_log = [f"Batch ID: {batchid}", "Starting..."] + livy_log
            output = html.Div("\n".join(livy_log), style={"whiteSpace": "pre-wrap"})
        elif "msg" in response.json():
            output = html.Div(response.text)
        return output
    else:
        return no_update


instructions = """
#### 1. Review

You are about to submit a job on the Fink Apache Spark & Kafka clusters.
Review your parameters, before hitting submission!

#### 2. Register

To retrieve the data, you need to get an account on the Fink Kafka cluster. See [fink-client](https://github.com/astrolabsoftware/fink-client) and
the [documentation](https://fink-broker.readthedocs.io/en/latest/services/data_transfer) for more information.

#### 3. Retrieve

Once data has started to flow in the topic, you can easily download your alerts using the [fink-client](https://github.com/astrolabsoftware/fink-client).
Install the latest version and use e.g.
"""


def layout():
    active = 0
    helper = """
    The Fink xmatch service allows you to upload a catalog of sources and crossmatch again Fink-processed alert data at scale.
    We provide access to alert data from ZTF (over 200 million alerts as of 2025), and soon from the Rubin Observatory.

    Follow these steps: (1) upload your catalog (100,000 rows maximum), (2) choose one or several years of alert data from Fink, and (3) select only the relevant alert fields to be added.

    The accepted formats for catalog are: csv, parquet, and votable. Coordinates are expected to be J2000 and in decimal degrees or hourangle.
    You can easily visualise the overlap between your catalog and the ZTF footprint by using the button `Crossmatch Sky Map` below your table.
    For information, here are some expected performances for an input catalog of 75k rows (the size of the input catalog does not matter much):

    ---

    | Years spanned | Input number of alerts (in millions) | Execution time (in seconds) |
    |---------|-----------------------------------|-----------------------|
    | 2025 | 20 | 221 |
    | 24-25 | 58 | 342 |
    | 23-25| 93 | 638 |
    | 22-25| 128 | 953 |
    | 21-25| 168 | 1320 |
    | 20-25| 199 | 1840 |

    ---

    Once ready, submit your job on the Fink Apache Spark and Kafka clusters to retrieve your data wherever you like.
    To access the data, you need to create an account. See the [fink-client](https://github.com/astrolabsoftware/fink-client) and
    the [documentation](https://fink-broker.readthedocs.io/en/latest/services/xmatch) for more information. The data is available
    for download for 7 days.
    """

    layout = dmc.Container(
        size="90%",
        children=[
            dmc.Space(h=20),
            dmc.Grid(
                justify="center",
                gutter={"base": 5, "xs": "md", "md": "xl", "xl": 50},
                grow=True,
                children=[
                    dmc.GridCol(
                        children=[
                            dmc.Stack(
                                [
                                    dmc.Space(h=20),
                                    dmc.Center(
                                        dmc.Title(
                                            children="Fink Xmatch",
                                            style={"color": "#15284F"},
                                        ),
                                    ),
                                    dmc.Space(h=20),
                                    dmc.RingProgress(
                                        roundCaps=True,
                                        sections=[{"value": 0, "color": "grey"}],
                                        size=250,
                                        thickness=20,
                                        label="",
                                        id="gauge_entry_number",
                                    ),
                                    dmc.Accordion(
                                        variant="separated",
                                        radius="xl",
                                        children=[
                                            dmc.AccordionItem(
                                                [
                                                    dmc.AccordionControl(
                                                        "Help",
                                                        icon=DashIconify(
                                                            icon="material-symbols:info-outline",
                                                            color="black",
                                                            width=30,
                                                        ),
                                                    ),
                                                    dmc.AccordionPanel(
                                                        dcc.Markdown(
                                                            helper, link_target="_blank"
                                                        ),
                                                    ),
                                                ],
                                                value="description",
                                            ),
                                        ],
                                        value=None,
                                    ),
                                ],
                                align="center",
                            )
                        ],
                        span=3,
                    ),
                    dmc.GridCol(
                        children=[
                            dmc.Space(h=40),
                            # dmc.Space(h=10),
                            dmc.Stepper(
                                color="#15284F",
                                id="stepper-xmatch-usage",
                                active=active,
                                children=[
                                    dmc.StepperStep(
                                        label="Choose catalog",
                                        description="Upload your catalog of objects",
                                        children=upload_catalog(),
                                        id="stepper-catalog",
                                    ),
                                    dmc.StepperStep(
                                        label="Date Range",
                                        description="Choose your date range for Fink",
                                        children=date_tab(),
                                        id="stepper-date-xmatch",
                                    ),
                                    dmc.StepperStep(
                                        label="Choose content",
                                        description="Pick up only relevant fields",
                                        children=filter_content_tab(),
                                    ),
                                    dmc.StepperStep(
                                        label="Launch xmatch!",
                                        description="Get your data",
                                        children=dmc.Grid(
                                            justify="center",
                                            gutter={
                                                "base": 5,
                                                "xs": "md",
                                                "md": "xl",
                                                "xl": 50,
                                            },
                                            grow=True,
                                            children=[
                                                dmc.GridCol(
                                                    children=[
                                                        dmc.Stack(
                                                            children=[
                                                                dmc.Space(h=20),
                                                                dmc.Group(
                                                                    children=[
                                                                        dmc.Button(
                                                                            "Submit job",
                                                                            id="submit_xmatch",
                                                                            variant="outline",
                                                                            color=COLORS_ZTF[
                                                                                0
                                                                            ],
                                                                            leftSection=DashIconify(
                                                                                icon="fluent:database-plug-connected-20-filled"
                                                                            ),
                                                                        ),
                                                                        html.A(
                                                                            dmc.Button(
                                                                                "Clear and restart",
                                                                                id="refresh-xmatch",
                                                                                color="red",
                                                                            ),
                                                                            href="/xmatch",
                                                                        ),
                                                                    ]
                                                                ),
                                                                html.Div(
                                                                    id="notification-container-xmatch"
                                                                ),
                                                                dmc.Group(children=[]),
                                                                dcc.Interval(
                                                                    id="interval-component-xmatch",
                                                                    interval=1 * 3000,
                                                                    n_intervals=0,
                                                                ),
                                                                html.Div(
                                                                    id="batch_log_xmatch"
                                                                ),
                                                            ],
                                                            align="center",
                                                        )
                                                    ],
                                                    span=6,
                                                ),
                                                dmc.GridCol(
                                                    dmc.Stack(
                                                        children=[
                                                            dmc.Space(h=20),
                                                            dcc.Markdown(instructions),
                                                            dmc.CodeHighlight(
                                                                code="# Submit to see code",
                                                                id="code_block_xmatch",
                                                                language="bash",
                                                            ),
                                                        ]
                                                    ),
                                                    span=6,
                                                ),
                                            ],
                                        ),
                                    ),
                                ],
                            ),
                            dmc.Space(h=10),
                            dmc.Group(
                                justify="center",
                                mt="xl",
                                children=[
                                    dmc.Button(
                                        "Back",
                                        id="back-xmatch-usage",
                                        variant="default",
                                    ),
                                    dmc.Button("Next step", id="next-xmatch-usage"),
                                ],
                            ),
                            dcc.Store(id="object-catalog"),
                            html.Div(
                                "", id="batch_id_xmatch", style={"display": "none"}
                            ),
                            html.Div(
                                "", id="topic_name_xmatch", style={"display": "none"}
                            ),
                        ],
                        span=9,
                    ),
                ],
            ),
        ],
    )

    return layout


@app.callback(
    [
        Output("ra-column", "disabled"),
        Output("dec-column", "disabled"),
        Output("id-column", "disabled"),
        Output("ra-column", "data"),
        Output("dec-column", "data"),
        Output("id-column", "data"),
        Output("radius-column", "data"),
        Output("radius-xmatch", "disabled"),
        Output("radius-column", "disabled"),
    ],
    Input("object-catalog", "data"),
    prevent_initial_call=True,
)
def select_columns(catalog):
    """
    Populate select options and enable the column selectors
    + radius widgets when a catalog is provided.
    """
    if catalog is None or catalog == {}:
        raise PreventUpdate

    pdf = pd.read_json(io.StringIO(catalog))
    if pdf.empty:
        raise PreventUpdate

    cols = [{"value": c, "label": c} for c in pdf.columns]

    enabled = False
    return (
        enabled,
        enabled,
        enabled,
        cols,
        cols,
        cols,
        cols,
        enabled,
        enabled,
    )


def enforce_decimal(pdf, ra_label, dec_label):
    """Convert RA and Dec to decimal degree if need be

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame
    ra_label: str
        RA column name
    dec_label: str
        Dec column name

    Returns
    -------
    out: np.array, np.array
        RA, Dec in decimal degrees
    """
    ra = pdf[ra_label].to_numpy()
    dec = pdf[dec_label].to_numpy()

    # conversion if not degree
    if isinstance(ra[0], str) and not ra[0].isnumeric():
        out = []
        for ra_, dec_ in zip(ra, dec):
            string = "{} {}".format(ra_, dec_)
            m = re.search(
                r"^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)\s+([+-])?\s*(\d{1,3})\s+(\d{1,2})\s+(\d{1,2}\.?\d*)(\s+(\d+\.?\d*))?$",
                string,
            ) or re.search(
                r"^(\d{1,2})[:h](\d{1,2})[:m](\d{1,2}\.?\d*)[s]?\s+([+-])?\s*(\d{1,3})[d:](\d{1,2})[m:](\d{1,2}\.?\d*)[s]?(\s+(\d+\.?\d*))?$",
                string,
            )
            if m:
                ra_deg = (float(m[1]) + float(m[2]) / 60 + float(m[3]) / 3600) * 15
                dec_deg = float(m[5]) + float(m[6]) / 60 + float(m[7]) / 3600

                if m[4] == "-":
                    dec_deg *= -1

                out.append([ra_deg, dec_deg])
        if len(out) > 0:
            ra, dec = np.transpose(out)

    return ra, dec


@app.callback(
    Output("aladin-lite-div-skymap-xmatch", "run"),
    [
        Input("ra-column", "value"),
        Input("dec-column", "value"),
        Input("object-catalog", "data"),
        Input("modal_skymap_xmatch", "is_open"),
    ],
    # prevent_initial_call=True,
)
def display_skymap(ra_label, dec_label, catalog, is_open):
    """Display explorer result on a sky map (Aladin lite). Limited to 1000 sources total.

    Callbacks
    ----------
    Output: Display a sky image with MOCs
    """
    if not is_open:
        return no_update

    if catalog is None or catalog == {}:
        return no_update

    pdf = pd.read_json(io.StringIO(catalog))

    # Conversion if need be
    pdf[ra_label], pdf[dec_label] = enforce_decimal(pdf, ra_label, dec_label)

    ra0 = pdf[ra_label].to_numpy()[0]
    dec0 = pdf[dec_label].to_numpy()[0]

    # Javascript. Note the use {{}} for dictionary
    # Force redraw of the Aladin lite window
    img = """var container = document.getElementById('aladin-lite-div-skymap-xmatch');var txt = ''; container.innerHTML = txt;"""

    # Aladin lite
    img += """
    var a = A.aladin('#aladin-lite-div-skymap-xmatch',
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

    try:
        # Catalog MOC
        m1 = MOC.from_lonlat(
            pdf[ra_label].to_numpy() * u.deg,
            pdf[dec_label].to_numpy() * u.deg,
            max_norder=6,
        )
        img += """var json = {};""".format(m1.to_string(format="json"))
        img += """var moc = A.MOCFromJSON(json, {opacity: 0.25, color: 'white', lineWidth: 1, name: "user catalog"}); a.addMOC(moc);"""

        # ZTF MOC
        with open("assets/MOC.json", "r") as f:
            data = json.loads(f.read())
        img += """var json2 = {};""".format(str(data))
        img += """var moc2 = A.MOCFromJSON(json2, {opacity: 0.25, color: 'red', lineWidth: 1, name: "ZTF DR7"}); a.addMOC(moc2);"""

        # img cannot be executed directly because of formatting
        # We split line-by-line and remove comments
        img_to_show = [i for i in img.split("\n") if "// " not in i]

        return " ".join(img_to_show)
    except Exception as e:
        print(e)
        return ""


def modal_skymap():
    import visdcc

    skymap_button = dmc.Button(
        "Crossmatch Sky Map",
        id="open_modal_skymap_xmatch",
        n_clicks=0,
        leftSection=DashIconify(icon="bi:stars"),
        color="gray",
        variant="default",
        radius="xl",
        disabled=True,
        w=250,
    )

    modal = html.Div(
        [
            skymap_button,
            dbc.Modal(
                [
                    # loading(
                    dbc.ModalBody(
                        html.Div(
                            [
                                visdcc.Run_js(
                                    id="aladin-lite-div-skymap-xmatch",
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
                            id="close_modal_skymap_xmatch",
                            className="ml-auto",
                            color="gray",
                            # fullWidth=True,
                            variant="default",
                            radius="xl",
                        ),
                    ),
                ],
                id="modal_skymap_xmatch",
                is_open=False,
                size="lg",
                # fullscreen="lg-down",
            ),
        ]
    )

    return modal


clientside_callback(
    """
    function toggle_modal_skymap_xmatch(n1, n2, is_open) {
        if (n1 || n2)
            return ~is_open;
        else
            return is_open;
    }
    """,
    Output("modal_skymap_xmatch", "is_open"),
    [
        Input("open_modal_skymap_xmatch", "n_clicks"),
        Input("close_modal_skymap_xmatch", "n_clicks"),
    ],
    [State("modal_skymap_xmatch", "is_open")],
    prevent_initial_call=True,
)


@app.callback(
    Output("open_modal_skymap_xmatch", "disabled", allow_duplicate=True),
    [
        Input("ra-column", "value"),
        Input("dec-column", "value"),
    ],
    prevent_initial_call=True,
)
def update_modal(ra_label, dec_label):
    if ra_label is not None and dec_label is not None:
        return False
    return True


@callback(
    Output("stepper-xmatch-usage", "active"),
    Input("back-xmatch-usage", "n_clicks"),
    Input("next-xmatch-usage", "n_clicks"),
    State("stepper-xmatch-usage", "active"),
    prevent_initial_call=True,
)
def update(back, next_, current):
    button_id = ctx.triggered_id
    step = current if current is not None else 0
    if button_id == "back-xmatch-usage":
        step = step - 1 if step > min_step else step
    else:
        step = step + 1 if step < max_step else step
    return step


@callback(
    Output("next-xmatch-usage", "style"),
    Input("next-xmatch-usage", "n_clicks"),
    Input("stepper-xmatch-usage", "active"),
    prevent_initial_call=True,
)
def last_step(next, current):
    if current == max_step - 1 or current == max_step:
        return {"display": "none"}
    return {}


@callback(
    Output("back-xmatch-usage", "style"),
    Input("back-xmatch-usage", "n_clicks"),
    Input("stepper-xmatch-usage", "active"),
)
def first_step(back, current):
    if current == 0 or current is None:
        return {"display": "none"}
    return {}


@callback(
    Output("stepper-catalog", "color"),
    Input("object-catalog", "data"),
    Input("back-xmatch-usage", "n_clicks"),
    Input("next-xmatch-usage", "n_clicks"),
)
def update_icon_date(catalog, back_, next_):
    button_id = ctx.triggered_id
    if button_id in ["back-xmatch-usage", "next-xmatch-usage"]:
        if catalog is None or catalog == {}:
            return "red"
        return "#15284F"
    return "#15284F"


@callback(
    Output("submit_xmatch", "disabled"),
    Input("object-catalog", "data"),
    Input("date-range-picker-xmatch", "value"),
)
def disable_button(catalog, date):
    f1 = date is None or date == ""
    f2 = catalog is None or catalog == {}
    if f1 or f2:
        return True
    return False


def parse_contents(catalog, filename, date):
    pdf = pd.read_json(io.StringIO(catalog))

    # Check header? Or ask the user to provide what is RA, DEC, OID?

    return html.Div(
        [
            html.H5("{}".format(filename)),
            html.H6("Preview of the 10 first rows"),
            dash_table.DataTable(
                pdf.head(10).to_dict("records"),
                [{"name": i, "id": i} for i in pdf.columns],
            ),
        ]
    )


@callback(
    Output("output-data-upload", "children"),
    Input("object-catalog", "data"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_output(catalog, filename, date):
    if catalog is not None:
        children = parse_contents(catalog, filename, date)
        return children


@callback(
    [
        Output("object-catalog", "data"),
        Output("gauge_entry_number", "sections"),
        Output("gauge_entry_number", "label"),
    ],
    [
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
    ],
    prevent_initial_call=True,
)
def store_catalog(content, filename):
    """Store data from user"""
    if content is None:
        return no_update, no_update, no_update
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if ".csv" in filename:
            # Assume that the user uploaded a CSV file
            pdf = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif ".parquet" in filename:
            # Assume that the user uploaded a parquet file
            pdf = pd.read_parquet(io.BytesIO(decoded))
        elif ".xml" in filename:
            # Assume that the user uploaded a votable file
            table = votable.parse(io.BytesIO(decoded))
            pdf = table.get_first_table().to_table(use_names_over_ids=True).to_pandas()
    except Exception as e:
        return (
            "{}",
            [{"value": 0, "color": "grey", "tooltip": "0%"}],
            dmc.Text(e, c="red", ta="center"),
        )

    if len(pdf) > MAX_ROW:
        msg = "{:,} is too many rows! Maximum is 100,000".format(len(pdf))
        return (
            "{}",
            [{"value": 100, "color": "red", "tooltip": "100%"}],
            dmc.Text(msg, c="red", ta="center"),
        )

    sections = {
        "value": len(pdf) / MAX_ROW * 100,
        "color": "green",
        "tooltip": "{:.2f}%".format(len(pdf) / MAX_ROW * 100),
    }
    label = dmc.Text("{:,} rows".format(len(pdf)), c=COLORS_ZTF[0], ta="center")

    return pdf.to_json(), [sections], label
