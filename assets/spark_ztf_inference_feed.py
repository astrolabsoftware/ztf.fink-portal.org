#!/usr/bin/env python
# Feed pre-classified ZTF alerts from HDFS into a Kafka topic so that
# the K8s inference jobs (kafka_bridge) can consume them.
# Modeled after spark_ztf_transfer.py — no field selection, no TNS cross-match.
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.column import Column, _to_java_column
import pyspark.sql.functions as F
from pyspark.sql.functions import lit, struct

from fink_filters.ztf.classification import extract_fink_classification
from fink_utils.spark import schema_converter

import pandas as pd
import requests
import sys
import argparse
import logging
from logging import Logger
from time import time


def get_fink_logger(name: str = "test", log_level: str = "INFO") -> Logger:
    FORMAT = "%(asctime)-15s -Livy- %(message)s"
    logging.basicConfig(format=FORMAT, datefmt="%y/%m/%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger


def to_avro(dfcol: Column) -> Column:
    sc = SparkContext._active_spark_context
    avro = sc._jvm.org.apache.spark.sql.avro
    f = getattr(getattr(avro, "package$"), "MODULE$").to_avro
    return Column(f(_to_java_column(dfcol)))


ZTF_ALERT_COLS = ["objectId", "candid", "candidate", "prv_candidates"]


def write_to_kafka(
    sdf, key, bootstrap_servers, sasl_username, sasl_password, topic_name, npart=10
):
    """Write a DataFrame to a Kafka topic.

    Wraps all columns of *sdf* into an Avro struct (value) and sets *key* as
    the message key.  Identical to the pattern used in spark_ztf_transfer.py
    so the same function handles both schema and data writes.
    """
    df_struct = sdf.select(struct(sdf.columns).alias("struct"))
    df_kafka = df_struct.select(to_avro("struct").alias("value"))
    df_kafka = df_kafka.withColumn("key", key)
    df_kafka = df_kafka.withColumn("partition", (F.rand(seed=0) * npart).astype("int"))
    (
        df_kafka.write.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("kafka.sasl.username", sasl_username)
        .option("kafka.sasl.password", sasl_password)
        .option("topic", topic_name)
        .save()
    )


def check_path_exist(dateToCheck):
    r = requests.post(
        "https://api.ztf.fink-portal.org/api/v1/statistics",
        json={
            "date": "{}{}{}".format(*dateToCheck.split("-")),
            "columns": "basic:sci",
            "output-format": "json",
        },
    )
    return r.json() != []


def generate_spark_paths(startDate, stopDate, basePath):
    endPath = "/year={}/month={}/day={}"
    if startDate == stopDate:
        return (
            [basePath + endPath.format(*startDate.split("-"))]
            if check_path_exist(startDate)
            else []
        )
    paths = []
    for aDate in pd.date_range(start=startDate, end=stopDate).astype("str").to_numpy():
        if check_path_exist(aDate):
            paths.append(basePath + endPath.format(*aDate.split("-")))
    return paths


def add_fink_class(df):
    return df.withColumn(
        "finkclass",
        extract_fink_classification(
            df["cdsxmatch"],
            df["roid"],
            df["mulens"],
            df["snn_snia_vs_nonia"],
            df["snn_sn_vs_all"],
            df["rf_snia_vs_nonia"],
            df["candidate.ndethist"],
            df["candidate.drb"],
            df["candidate.classtar"],
            df["candidate.jd"],
            df["candidate.jdstarthist"],
            df["rf_kn_vs_nonkn"],
            df["tracklet"],
        ),
    )


def main(args):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log = get_fink_logger(__file__)

    log.info(
        "Generating data paths from {} to {}".format(args.startDate, args.stopDate)
    )
    basepath = "hdfs://vdmaster1:8020/user/julien.peloton/archive/science"
    paths = generate_spark_paths(args.startDate, args.stopDate, basepath)

    if not paths:
        log.info("No alert data found in the requested date range.")
        spark.stop()
        sys.exit(1)

    df = spark.read.format("parquet").option("basePath", basepath).load(paths)
    df = add_fink_class(df)

    # Filter by alert class if requested
    if args.fclass:
        tns_class = [c for c in args.fclass if c.startswith("(TNS)")]
        other_class = [
            c.replace("(SIMBAD) ", "") for c in args.fclass if c not in tns_class
        ]
        conditions = []
        if other_class:
            conditions.append(df["finkclass"].isin(other_class))
        if tns_class:
            sanitized = [c.replace("(TNS) ", "") for c in tns_class]
            conditions.append(df["tnsclass"].isin(sanitized))
        if conditions:
            combined = conditions[0]
            for c in conditions[1:]:
                combined = combined | c
            df = df.filter(combined)

    # Keep only the fields needed by the preprocessing container
    ztf_cols = [c for c in ZTF_ALERT_COLS if c in df.columns]
    df = df.select(ztf_cols)

    # Extract schema and publish it to the schema topic (key = schema JSON string).
    # Same pattern as spark_ztf_transfer.py so the fink-client can also read it.
    log.info("Determining data schema...")
    schema = schema_converter.to_avro(df.coalesce(1).limit(1).schema)

    # schema_converter.to_avro() incorrectly converts nullable StructType fields
    # (like `candidate`) to plain Avro records instead of [record, null] unions.
    # Spark's to_avro() serializer DOES write a union discriminator byte for these
    # fields.  Fix the published schema so consumers can decode the wire bytes.
    import json as _json

    _schema_dict = _json.loads(schema)
    for _field in _schema_dict.get("fields", []):
        _t = _field.get("type")
        if isinstance(_t, dict) and _t.get("type") == "record":
            _field["type"] = [_t, "null"]
    schema = _json.dumps(_schema_dict)

    df_schema = spark.createDataFrame(
        pd.DataFrame({"schema": ["new_schema_{}.avsc".format(time())] * 1000})
    )

    log.info("Sending schema to topic {}...".format(args.topic_name + "_schema"))
    write_to_kafka(
        df_schema,
        lit(schema),
        args.kafka_bootstrap_servers,
        args.kafka_sasl_username,
        args.kafka_sasl_password,
        args.topic_name + "_schema",
    )

    log.info("Writing {} alerts to topic {}".format(df.count(), args.topic_name))
    write_to_kafka(
        df,
        lit(args.topic_name),
        args.kafka_bootstrap_servers,
        args.kafka_sasl_username,
        args.kafka_sasl_password,
        args.topic_name,
    )
    log.info("Done.")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-startDate", type=str, required=True)
    parser.add_argument("-stopDate", type=str, required=True)
    parser.add_argument("-topic_name", type=str, required=True)
    parser.add_argument("-kafka_bootstrap_servers", type=str, required=True)
    parser.add_argument("-kafka_sasl_username", type=str, required=True)
    parser.add_argument("-kafka_sasl_password", type=str, required=True)
    parser.add_argument("-fclass", type=str, action="append", default=None)
    args = parser.parse_args()
    main(args)
