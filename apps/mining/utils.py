# Copyright 2023-2024 AstroLab Software
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
import json
import logging
import traceback
import yaml
from datetime import date, timedelta, datetime, timezone
import requests
import numpy as np
import pandas as pd

from apps.utils import query_and_order_statistics, request_api, select_struct

coeffs_per_class = pd.read_parquet("assets/fclass_2022_060708_coeffs.parquet")
coeffs_per_filters = pd.read_parquet("assets/ffilters_2025_01_to_06_coeffs.parquet")

CONV = {
    "float": 4,
    "double": 8,
    "int": 4,
    "string": 8,
    "array": 4 * 60 * 60,
    "boolean": 1,
    "long": 8,
}


def upload_file_hdfs(code, webhdfs, namenode, user, filename):
    """Upload a file to HDFS

    Parameters
    ----------
    code: str
        Code as string
    webhdfs: str
        Location of the code on webHDFS in the format
        http://<IP>:<PORT>/webhdfs/v1/<path>
    namenode: str
        Namenode and port in the format
        <IP>:<PORT>
    user: str
        User name in HDFS
    filename: str
        Name on the file to be created

    Returns
    -------
    status_code: int
        HTTP status code. 201 is a success.
    text: str
        Additional information on the query (log).
    """
    try:
        response = requests.put(
            f"{webhdfs}/{filename}?op=CREATE&user.name={user}&namenoderpcaddress={namenode}&createflag=&createparent=true&overwrite=true",
            data=code,
        )
        status_code = response.status_code
        text = response.text
    except (requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
        status_code = -1
        text = e

    if status_code != 201:
        print(f"Status code: {status_code}")
        print(f"Log: {text}")

    return status_code, text


def submit_spark_job(livyhost, filename, spark_conf, job_args):
    """Submit a job on the Spark cluster via Livy (batch mode)

    Parameters
    ----------
    livyhost: str
        IP:HOST for the Livy service
    filename: str
        Path on HDFS with the file to submit. Format:
        hdfs://<path>/<filename>
    spark_conf: dict
        Dictionary with Spark configuration
    job_args: list of str
        Arguments for the Spark job in the form
        ['-arg1=val1', '-arg2=val2', ...]

    Returns
    -------
    batchid: int
        The number of the submitted batch
    response.status_code: int
        HTTP status code
    response.text: str
        Payload
    """
    headers = {"Content-Type": "application/json"}

    data = {
        "conf": spark_conf,
        "file": filename,
        "args": job_args,
    }
    response = requests.post(
        "http://" + livyhost + "/batches",
        data=json.dumps(data),
        headers=headers,
    )

    batchid = response.json()["id"]

    if response.status_code != 201:
        print(f"Batch ID {batchid}")
        print(f"Status code: {response.status_code}")
        print(f"Log: {response.text}")

    return batchid, response.status_code, response.text


def extract_type(field):
    if isinstance(field, list):
        # null, type
        return field[1]
    else:
        return field


def estimate_size_gb_ztf(content):
    """Estimate the size of the data to download

    Parameters
    ----------
    content: list
        List of selected alert fields
    """
    if content is None:
        return 0
    # Pre-defined schema
    if "Full packet" in content:
        # all nested fields, incl prv_candidates
        sizeGb = 55.0 / 1024 / 1024
    elif "Light packet" in content:
        sizeGb = 1.4 / 1024 / 1024
    elif "Medium packet" in content:
        sizeGb = 18.0 / 1024 / 1024
    else:
        # freedom on candidates + added values
        schema = request_api("/api/v1/schema", method="GET", output="json")
        sizeB = 0
        for k_out in schema.keys():
            for k_in, field in schema[k_out].items():
                if select_struct(k_in) in content:
                    sizeB += CONV[extract_type(field["type"])]
                elif select_struct(k_in, "candidate.") in content:
                    sizeB += CONV[extract_type(field["type"])]

        sizeGb = sizeB / 1024 / 1024 / 1024

    return sizeGb


def estimate_size_gb_elasticc(content):
    """Estimate the size of the data to download

    Parameters
    ----------
    content: str
        Name as given by content_tab
    """
    if "Full packet" in content:
        sizeGb = 1.4 / 1024 / 1024

    return sizeGb


def initialise_classes(class_select):
    """Add classes selected by the user

    Parameters
    ----------
    class_select: list, optional
        List of classes selected by the user.
        None is not class selected.

    Returns
    -------
    columns: str
        Comma-separated names of classes
    column_classes: list
        List of classes. Empty list if no class selected.
    """
    column_names = []
    columns = "basic:sci"
    if (class_select is not None) and (class_select != []):
        if "allclasses" not in class_select:
            for elem in class_select:
                if elem.startswith("(TNS)"):
                    continue

                # name correspondance
                if elem.startswith("(SIMBAD)"):
                    elem = elem.replace("(SIMBAD) ", "class:")
                else:
                    # prepend class:
                    elem = "class:" + elem
                columns += f",{elem}"
                column_names.append(elem)

    return columns, column_names


def get_statistics(column_names, dstart, dstop, with_class=True):
    """ """
    dic = {"basic:sci": 0}

    # Get total number of alerts for the period
    pdf = query_and_order_statistics(
        drop=False,
    )
    pdf["ISO"] = pdf["key:key"].apply(lambda x: x.split("_")[1])

    f1 = pdf["ISO"] <= dstop.strftime("%Y%m%d")
    f2 = pdf["ISO"] >= dstart.strftime("%Y%m%d")

    pdf = pdf[f1 & f2]
    dic["basic:sci"] += int(pdf["basic:sci"].sum())

    if with_class:
        # Initialise count
        for column_name in column_names:
            if column_name in pdf.columns:
                dic[column_name] = int(pdf[column_name].sum())
            else:
                dic[column_name] = 0

    return dic


def add_tns_estimation(dic, class_select):
    """Add estimation for TNS classes

    TNS statistics is not pushed in /statistics
    """
    if "allclasses" not in class_select:
        for elem in class_select:
            # name correspondance
            if elem.startswith("(TNS)"):
                filt = coeffs_per_class["fclass"] == elem

                if np.sum(filt) == 0:
                    # Nothing found. This could be because we have
                    # no alerts from this class, or because it has not
                    # yet entered the statistics. To be conservative,
                    # we do not apply any coefficients.
                    dic[elem] = 0
                else:
                    dic[elem.replace("(TNS) ", "class:")] = int(
                        dic["basic:sci"] * coeffs_per_class[filt]["coeff"].to_numpy()[0]
                    )

    return dic


def get_filter_statistics(dic, filter_select):
    """Get stastitics based on a user-defined filter

    Parameters
    ----------
    dic: dict
        Dictionnary containing counts
    filter_select: str, optional
        Filter name
    """
    id_ = coeffs_per_filters["filter"] == filter_select
    if np.sum(id_) == 1:
        dic[filter_select] = (
            coeffs_per_filters[id_]["coeff"].to_numpy()[0] * dic["basic:sci"]
        )

    return dic


def estimate_alert_number_ztf(date_range_picker, class_select, filter_select):
    """Callback to estimate the number of alerts to be transfered

    This can be improved by using the REST API directly to get number of
    alerts per class.
    """
    dstart = date(*[int(i) for i in date_range_picker[0].split("-")])
    dstop = date(*[int(i) for i in date_range_picker[1].split("-")])

    columns, column_names = initialise_classes(class_select)

    with_filter = (
        (filter_select is not None) and (filter_select != "") and (filter_select != [])
    )
    with_class = (
        (class_select is not None) and (class_select != "") and (class_select != [])
    )
    dic = get_statistics(column_names, dstart, dstop, with_class=not with_filter)

    # we check first filter, and then class
    if with_filter:
        dic = get_filter_statistics(dic, filter_select)
        total = dic["basic:sci"]
        count = np.sum([v for k, v in dic.items() if k != "basic:sci"])
    elif with_class:
        dic = add_tns_estimation(dic, class_select)
        total = dic["basic:sci"]
        count = np.sum([v for k, v in dic.items() if k != "basic:sci"])
    else:
        total = dic["basic:sci"]
        count = dic["basic:sci"]

    return total, count


def estimate_alert_number_elasticc(
    date_range_picker, class_select, elasticc_dates, elasticc_classes
):
    """Callback to estimate the number of alerts to be transfered"""
    dic = {"basic:sci": 0}
    dstart = date(*[int(i) for i in date_range_picker[0].split("-")])
    dstop = date(*[int(i) for i in date_range_picker[1].split("-")])
    delta = dstop - dstart

    # count all raw number of alerts
    for i in range(delta.days + 1):
        tmp = (dstart + timedelta(i)).strftime("%Y%m%d")
        filt = elasticc_dates["date"] == tmp
        if np.sum(filt) > 0:
            dic["basic:sci"] += int(elasticc_dates[filt]["count"].to_numpy()[0])

    # Add class estimation
    if (class_select is not None) and (class_select != []):
        if "allclasses" not in class_select:
            for elem in class_select:
                # name correspondance
                filt = elasticc_classes["classId"].astype(int) == int(elem)

                if np.sum(filt) == 0:
                    # Nothing found. This could be because we have
                    # no alerts from this class, or because it has not
                    # yet entered the statistics. To be conservative,
                    # we do not apply any coefficients.
                    dic[elem] = 0
                else:
                    coeff = (
                        elasticc_classes[filt]["count"].to_numpy()[0]
                        / elasticc_classes["count"].sum()
                    )
                    dic["class:" + str(elem)] = int(dic["basic:sci"] * coeff)
            count = np.sum([i[1] for i in dic.items() if "class:" in i[0]])
        else:
            # allclasses mean all alerts
            count = dic["basic:sci"]
    else:
        count = dic["basic:sci"]

    return dic["basic:sci"], count


def _discover_component_image(model_name, version, component):
    """Read preprocessing_image or model_image tag from MLflow model version."""
    config = yaml.load(open("config_inference.yml"), yaml.Loader)
    mlflow_uri = (config.get("MLFLOW_TRACKING_URI") or "").rstrip("/")
    username = config.get("MLFLOW_TRACKING_USERNAME") or None
    password = config.get("MLFLOW_TRACKING_PASSWORD") or None
    auth = (username, password) if username and password else None

    tag_key = "preprocessing_image" if component == "preprocessing" else "model_image"
    try:
        r = requests.get(
            f"{mlflow_uri}/api/2.0/mlflow/model-versions/get",
            params={"name": model_name, "version": version},
            auth=auth,
            timeout=5,
        )
        if r.status_code != 200:
            logging.warning(
                "[MLflow] model-versions/get %s@%s → HTTP %s",
                model_name,
                version,
                r.status_code,
            )
            return None
        mv = r.json().get("model_version", {})
        tags = {t["key"]: t["value"] for t in mv.get("tags", [])}
        image = tags.get(tag_key)
        if not image:
            logging.warning(
                "[MLflow] Tag '%s' missing for %s@%s", tag_key, model_name, version
            )
        return image
    except Exception:
        logging.warning(
            "[MLflow] Could not get image tag for %s@%s\n%s",
            model_name,
            version,
            traceback.format_exc(),
        )
        return None


def create_k8s_inference_jobs(
    input_topic, output_topic, job_id, selected_models, inf_config
):
    """Create two K8s Jobs per selected model: preprocessing + model.

    Flow per model:
      input_topic (AVRO) → preprocessing Job → intermediate (JSON) → model Job → output_topic (JSON)
    """
    namespace = inf_config.get("KUBE_NAMESPACE", "fink")
    parallelism = int(inf_config.get("INFERENCE_PARALLELISM", 1))
    dt_config = yaml.load(open("config_datatransfer.yml"), yaml.Loader)

    from kubernetes import client as k8s_client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    batch_v1 = k8s_client.BatchV1Api()

    protocol = dt_config.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    kafka_creds = [
        {
            "name": "KAFKA_BOOTSTRAP_SERVERS",
            "value": dt_config["KAFKA_BOOTSTRAP_SERVERS"],
        },
        {"name": "KAFKA_SECURITY_PROTOCOL", "value": protocol},
    ]
    if protocol not in ("PLAINTEXT", "SSL"):
        kafka_creds += [
            {
                "name": "KAFKA_SASL_USERNAME",
                "value": dt_config.get("KAFKA_SASL_USERNAME", ""),
            },
            {
                "name": "KAFKA_SASL_PASSWORD",
                "value": dt_config.get("KAFKA_SASL_PASSWORD", ""),
            },
            {
                "name": "KAFKA_SASL_MECHANISM",
                "value": dt_config.get("KAFKA_SASL_MECHANISM", "PLAIN"),
            },
        ]

    errors = []
    created = []

    for model_str in selected_models:
        parts = model_str.split("@")
        model_name = parts[0]
        version = parts[1] if len(parts) > 1 else "1"
        model_safe = (
            model_str.replace("@", "-").replace(".", "-").replace("_", "-").lower()
        )
        ts = datetime.now(timezone.utc).microsecond

        inter_topic = f"fink_ai_pre_{model_safe}_{job_id}"

        pre_image = _discover_component_image(model_name, version, "preprocessing")
        if not pre_image:
            errors.append(f"No preprocessing image found for '{model_name}@{version}'")
            continue

        pre_job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"inf-pre-{model_safe}-{ts}", "namespace": namespace},
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "parallelism": parallelism,
                "completions": parallelism,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "preprocessing",
                                "image": pre_image,
                                "env": kafka_creds
                                + [
                                    {"name": "KAFKA_ENABLED", "value": "true"},
                                    {"name": "INPUT_TOPIC", "value": input_topic},
                                    {"name": "OUTPUT_TOPIC", "value": inter_topic},
                                    {"name": "INPUT_FORMAT", "value": "avro"},
                                    {
                                        "name": "SCHEMA_TOPIC",
                                        "value": f"{input_topic}_schema",
                                    },
                                    {"name": "OUTPUT_FORMAT", "value": "json"},
                                    {"name": "SKIP_CUTOUTS", "value": "true"},
                                    {"name": "AUTO_OFFSET_RESET", "value": "earliest"},
                                    {
                                        "name": "CONSUMER_GROUP_ID",
                                        "value": f"fink-ai-pre-{model_safe}-{job_id}",
                                    },
                                    {"name": "IDLE_TIMEOUT_SECONDS", "value": "300"},
                                ],
                            }
                        ],
                    }
                },
            },
        }

        model_image = _discover_component_image(model_name, version, "model")
        if not model_image:
            errors.append(f"No model image found for '{model_name}@{version}'")
            continue

        model_job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"inf-model-{model_safe}-{ts}",
                "namespace": namespace,
            },
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "parallelism": parallelism,
                "completions": parallelism,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "model",
                                "image": model_image,
                                "env": kafka_creds
                                + [
                                    {"name": "KAFKA_ENABLED", "value": "true"},
                                    {"name": "INPUT_TOPIC", "value": inter_topic},
                                    {"name": "OUTPUT_TOPIC", "value": output_topic},
                                    {"name": "INPUT_FORMAT", "value": "json"},
                                    {"name": "OUTPUT_FORMAT", "value": "json"},
                                    {"name": "AUTO_OFFSET_RESET", "value": "earliest"},
                                    {
                                        "name": "CONSUMER_GROUP_ID",
                                        "value": f"fink-ai-model-{model_safe}-{job_id}",
                                    },
                                    {"name": "BRIDGE_NAME", "value": model_str},
                                    {"name": "IDLE_TIMEOUT_SECONDS", "value": "900"},
                                ],
                            }
                        ],
                    }
                },
            },
        }

        try:
            batch_v1.create_namespaced_job(namespace=namespace, body=pre_job)
            batch_v1.create_namespaced_job(namespace=namespace, body=model_job)
            created.append(model_str)
        except Exception:
            logging.warning(
                "[K8s] Job creation failed for %s\n%s",
                model_str,
                traceback.format_exc(),
            )
            errors.append(f"K8s Job creation failed for {model_str}")

    return created, errors
