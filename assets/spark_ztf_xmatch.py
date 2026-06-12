#!/usr/bin/env python
# Copyright 2025-2026 AstroLab Software
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
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.column import Column, _to_java_column
import pyspark.sql.functions as F
from pyspark.sql.functions import struct, lit
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from pyspark.sql.utils import AnalysisException

from fink_filters.ztf.classification import extract_fink_classification
from fink_utils.spark import schema_converter

from fink_science.ztf.xmatch.utils import cross_match_astropy

from time import time
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

import sys
import argparse

import logging
from logging import Logger


def get_fink_logger(name: str = "test", log_level: str = "INFO") -> Logger:
    """Initialise python logger. Suitable for both driver and executors.

    Parameters
    ----------
    name : str
        Name of the application to be logged. Typically __name__ of a
        function or module.
    log_level : str
        Minimum level of log wanted: DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF

    Returns
    -------
    logger : logging.Logger
        Python Logger

    Examples
    --------
    >>> log = get_fink_logger(__name__, "INFO")
    >>> log.info("Hi!")
    """
    # Format of the log message to be printed
    FORMAT = "%(asctime)-15s "
    FORMAT += "-Livy- "
    FORMAT += "%(message)s"

    # Date format
    DATEFORMAT = "%y/%m/%d %H:%M:%S"

    logging.basicConfig(format=FORMAT, datefmt=DATEFORMAT)
    logger = logging.getLogger(name)

    # Set the minimum log level
    logger.setLevel(log_level)

    return logger


def add_classification(spark, df, path_to_tns):
    """Add classification from Fink & TNS

    Parameters
    ----------
    spark:
    df: DataFrame
        Spark DataFrame containing ZTF alert data
    path_to_tns: str
        Path to TNS data (parquet)

    Returns
    -------
    df: DataFrame
        Input DataFrame with 2 new columns `finkclass` and
        `tns` containing classification tags.
    """
    # extract Fink classification
    df = df.withColumn(
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

    pdf_tns_filt = pd.read_parquet(path_to_tns)
    pdf_tns_filt_b = spark.sparkContext.broadcast(pdf_tns_filt)

    @pandas_udf(StringType(), PandasUDFType.SCALAR)
    def crossmatch_with_tns(objectid, ra, dec):
        # TNS
        pdf = pdf_tns_filt_b.value
        ra2, dec2, type2 = pdf["ra"], pdf["declination"], pdf["type"]

        # create catalogs
        catalog_ztf = SkyCoord(
            ra=np.array(ra, dtype=np.float) * u.degree,
            dec=np.array(dec, dtype=np.float) * u.degree,
        )
        catalog_tns = SkyCoord(
            ra=np.array(ra2, dtype=np.float) * u.degree,
            dec=np.array(dec2, dtype=np.float) * u.degree,
        )

        # cross-match
        idx, d2d, d3d = catalog_tns.match_to_catalog_sky(catalog_ztf)

        sub_pdf = pd.DataFrame(
            {
                "objectId": objectid.to_numpy(),
                "ra": ra.to_numpy(),
                "dec": dec.to_numpy(),
            }
        )

        # cross-match
        idx2, d2d2, d3d2 = catalog_ztf.match_to_catalog_sky(catalog_tns)

        # set separation length
        sep_constraint2 = d2d2.degree < 1.5 / 3600

        sub_pdf["TNS"] = ["Unknown"] * len(sub_pdf)
        sub_pdf["TNS"][sep_constraint2] = type2.to_numpy()[idx2[sep_constraint2]]

        to_return = objectid.apply(
            lambda x: (
                "Unknown"
                if x not in sub_pdf["objectId"].to_numpy()
                else sub_pdf["TNS"][sub_pdf["objectId"] == x].to_numpy()[0]
            )
        )

        return to_return

    # remove old tns class if need be
    df = df.drop("tns")

    df = df.withColumn(
        "tns",
        crossmatch_with_tns(df["objectId"], df["candidate.ra"], df["candidate.dec"]),
    )

    return df


def to_avro(dfcol: Column) -> Column:
    """Serialize the structured data of a DataFrame column into avro data (binary).

    Note:
    Since Pyspark does not have a function to convert a column to and from
    avro data, this is a wrapper around the scala function 'to_avro'.
    Just like the function above, to be able to use this you need to have
    the package org.apache.spark:spark-avro_2.11:2.x.y in the classpath.

    Parameters
    ----------
    dfcol: Column
        A DataFrame Column with Structured data

    Returns
    -------
    out: Column
        DataFrame Column encoded into avro data (binary).
        This is what is required to publish to Kafka Server for distribution.

    Examples
    --------
    >>> from pyspark.sql.functions import col, struct
    >>> avro_example_schema = '''
    ... {
    ...     "type" : "record",
    ...     "name" : "struct",
    ...     "fields" : [
    ...             {"name" : "col1", "type" : "long"},
    ...             {"name" : "col2", "type" : "string"}
    ...     ]
    ... }'''
    >>> df = spark.range(5)
    >>> df = df.select(struct("id",\
                 col("id").cast("string").alias("id2"))\
                 .alias("struct"))
    >>> avro_df = df.select(to_avro(col("struct")).alias("avro"))
    """
    sc = SparkContext._active_spark_context
    avro = sc._jvm.org.apache.spark.sql.avro
    f = getattr(getattr(avro, "package$"), "MODULE$").to_avro
    return Column(f(_to_java_column(dfcol)))


def write_to_kafka(
    sdf,
    key,
    kafka_bootstrap_servers,
    kafka_sasl_username,
    kafka_sasl_password,
    topic_name,
    npart=10,
):
    """Send data to a Kafka cluster using Apache Spark

    Parameters
    ----------
    sdf: Spark DataFrame
        DataFrame
    key: str
        key for each Avro message
    kafka_bootstrap_servers: str
        Comma-separated list of ip:port of the Kafka machines
    kafka_sasl_username: str
        Username for writing into the Kafka cluster
    kafka_sasl_password: str
        Password for writing into the Kafka cluster
    topic_name: str
        Kafka topic (does not need to exist)
    npart: int, optional
        Number of Kafka partitions. Default is 10.
    """
    # Create a StructType column in the df for distribution.
    df_struct = sdf.select(struct(sdf.columns).alias("struct"))
    df_kafka = df_struct.select(to_avro("struct").alias("value"))
    df_kafka = df_kafka.withColumn("key", key)
    df_kafka = df_kafka.withColumn("partition", (F.rand(seed=0) * npart).astype("int"))

    # Send schema
    _ = (
        df_kafka.write.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("kafka.sasl.username", kafka_sasl_username)
        .option("kafka.sasl.password", kafka_sasl_password)
        .option("topic", topic_name)
        .save()
    )


def generate_spark_paths(startDate, stopDate, basePath):
    """Generate individual yearly data paths

    Parameters
    ----------
    startDate: str
        YYYY
    stopDate: str
        YYYY
    basePath: str
        HDFS basepath for the data

    Returns
    -------
    paths: list of str
        List of yearly paths
    """
    endPath = "/year={}"

    if startDate == stopDate:
        # easy case -- one year
        paths = [basePath + endPath.format(startDate.split("-")[0])]
    else:
        # more than one year
        dateRange = (
            pd.date_range(start=startDate, end=stopDate, freq="Y")
            .astype("str")
            .to_numpy()
        )

        paths = []
        for aDate in dateRange:
            # Keep only the year
            paths.append(basePath + endPath.format(aDate.split("-")[0]))

    return paths


def perform_xmatch(spark, df, catalog_filename, ra_col, dec_col, id_col, radius_arcsec):
    """"""
    df_other = spark.read.format("parquet").load(catalog_filename)
    pdf_other = df_other.toPandas()
    pdf_b = spark.sparkContext.broadcast(pdf_other)

    @pandas_udf(StringType(), PandasUDFType.SCALAR)
    def crossmatch(ra, dec):
        """Spark UDF for simple crossmatch"""
        pdf_cat = pdf_b.value
        ra2, dec2, id2 = pdf_cat[ra_col], pdf_cat[dec_col], pdf_cat[id_col]

        pdf = pd.DataFrame(
            {
                "ra": ra.to_numpy(),
                "dec": dec.to_numpy(),
                "candid": range(len(ra)),
            }
        )

        # create catalogs
        catalog_ztf = SkyCoord(
            ra=np.array(ra, dtype=np.float) * u.degree,
            dec=np.array(dec, dtype=np.float) * u.degree,
        )
        catalog_other = SkyCoord(
            ra=np.array(ra2, dtype=np.float) * u.degree,
            dec=np.array(dec2, dtype=np.float) * u.degree,
        )

        if (
            isinstance(radius_arcsec, dict)
            and radius_arcsec["value"] in pdf_cat.columns
        ):
            radius_col = pdf_cat[radius_arcsec["value"]]
        elif isinstance(radius_arcsec, float) or isinstance(radius_arcsec, int):
            radius_col = pd.Series([radius_arcsec])
        else:
            # FIXME: log error message
            pass

        pdf_merge, mask, idx2 = cross_match_astropy(
            pdf, catalog_ztf, catalog_other, radius_arcsec=radius_col
        )

        pdf_merge["Type"] = "Unknown"
        pdf_merge.loc[mask, "Type"] = [
            str(i).strip() for i in id2.astype(str).to_numpy()[idx2]
        ]

        return pdf_merge["Type"]

    # Keep only matches
    df = df.withColumn(
        id_col,
        crossmatch(df["candidate.ra"], df["candidate.dec"]),
    ).filter(F.col(id_col) != "Unknown")

    return df


def sanitize_fields(cnames):
    """Apply proper serialization before sending to Kafka

    Parameters
    ----------
    cnames: list
        List of fields

    Returns
    -------
    cnames: list
        List of fields, sanitized.
    """
    for cutout in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]:
        if cutout in cnames:
            cnames[cnames.index(cutout)] = "struct({}.*) as {}".format(cutout, cutout)

    if "prv_candidates" in cnames:
        cnames[cnames.index("prv_candidates")] = (
            "explode(array(prv_candidates)) as prv_candidates"
        )

    if "candidate" in cnames:
        cnames[cnames.index("candidate")] = "struct(candidate.*) as candidate"

    for lc_features in ["lc_features_g", "lc_features_r"]:
        if lc_features in cnames:
            cnames[cnames.index(lc_features)] = "struct({}.*) as {}".format(
                lc_features, lc_features
            )

    for ts in [
        "timestamp",
        "brokerEndProcessTimestamp",
        "brokerStartProcessTimestamp",
        "brokerIngestTimestamp",
    ]:
        if ts in cnames:
            cnames[cnames.index(ts)] = "cast({} as string) as {}".format(ts, ts)

    # not needed actually as cnames is changed in-place
    return cnames


def main(args):
    t0 = time()
    spark = SparkSession.builder.getOrCreate()

    # reduce Java verbosity
    spark.sparkContext.setLogLevel("WARN")

    log = get_fink_logger(__file__)

    log.info("Generating data paths...")
    paths = generate_spark_paths(args.startDate, args.stopDate, args.basePath)
    if paths == []:
        log.info(
            "No alert data found in between {} and {}".format(
                args.startDate, args.stopDate
            )
        )
        spark.stop()
        sys.exit(1)

    df = (
        spark.read.format("parquet")
        .option("basePath", args.basePath)
        .option("mergeSchema", True)
        .load(paths)
    )

    df = add_classification(spark, df, args.path_to_tns)

    # Perform the xmatch
    df = perform_xmatch(
        spark,
        df,
        args.catalog_filename,
        args.ra_col,
        args.dec_col,
        args.id_col,
        args.radius_arcsec,
    )

    # Define content
    if args.ffield is None:
        cnames = ["objectId", "candid", args.id_col]
    elif not isinstance(args.ffield, list):
        log.warning("Content has not been defined: {}".format(args.ffield))
        log.warning("Exiting.")
        spark.stop()
        sys.exit(1)
    else:
        cnames = args.ffield
        cnames.append(args.id_col)

    log.info("Selecting Fink/ZTF content {}...".format(cnames))

    # enforce proper serialisation
    cnames = sanitize_fields(cnames)

    # Check fields exist
    for cname in cnames:
        try:
            df.selectExpr(cname)
        except AnalysisException:  # noqa: PERF203
            log.warning(
                "Removing {} from the list of fields as it is not available for the selected dates. Check https://doc.ztf.fink-broker.org/en/latest/broker/science_modules/ for availability".format(
                    cname
                )
            )
            cnames.remove(cname)

    if len(cnames) == 0:
        log.error(
            "No alert fields have been selected. Restart the job by choosing at least one field."
        )
        sys.exit(1)

    # Wrap alert data
    df = df.selectExpr(cnames)

    # extract schema
    log.info("Determining data schema...")
    schema = schema_converter.to_avro(df.coalesce(1).limit(1).schema)

    log.info("Schema OK...")

    # create a fake dataframe with 100 entries
    df_schema = spark.createDataFrame(
        pd.DataFrame({"schema": ["new_schema_{}.avsc".format(time())] * 1000})
    )

    log.info("Sending the schema to Kafka...")

    # Send schema
    write_to_kafka(
        df_schema,
        lit(schema),
        args.kafka_bootstrap_servers,
        args.kafka_sasl_username,
        args.kafka_sasl_password,
        args.topic_name + "_schema",
    )

    log.info("Starting to send data to topic {}".format(args.topic_name))

    write_to_kafka(
        df,
        lit(args.topic_name),
        args.kafka_bootstrap_servers,
        args.kafka_sasl_username,
        args.kafka_sasl_password,
        args.topic_name,
    )

    log.info("Data available at topic: {}".format(args.topic_name))
    log.info("It took {:.2f} seconds".format(time() - t0))
    log.info("End.")


if __name__ == "__main__":
    """ Execute the test suite """
    parser = argparse.ArgumentParser()

    parser.add_argument("-startDate")
    parser.add_argument("-stopDate")
    parser.add_argument("-ra_col")
    parser.add_argument("-dec_col")
    parser.add_argument("-radius_arcsec")
    parser.add_argument("-id_col")
    parser.add_argument("-catalog_filename")
    parser.add_argument("-ffield", action="append")
    parser.add_argument("-basePath")
    parser.add_argument("-topic_name")
    parser.add_argument("-kafka_bootstrap_servers")
    parser.add_argument("-kafka_sasl_username")
    parser.add_argument("-kafka_sasl_password")
    parser.add_argument("-path_to_tns")

    args = parser.parse_args()
    main(args)
