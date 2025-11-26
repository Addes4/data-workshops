#!/usr/bin/env python3
from pathlib import Path
import argparse
import io

import pandas as pd
import requests

BASE_URL = "https://datasets.imdbws.com/"
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "imdb_movies.csv"
_VERBOSE = False

DTYPES = {
    "primaryTitle": "string",
    "originalTitle": "string",
    "startYear": "Int16",
    "runtimeMinutes": "Int32",
    "genres": "string",
    "averageRating": "float32",
    "numVotes": "Int32",
}


def set_verbosity(enabled):
    """Enable or disable verbose logging."""
    global _VERBOSE
    _VERBOSE = enabled


def log(message, *, detail=True):
    """Consistent console logging."""
    if detail and not _VERBOSE:
        return
    print(f"[build-imdb] {message}")


def format_count(value):
    return f"{value:,}"


def download_imdb_archive(name):
    """
    Download an IMDb TSV.GZ archive into memory and return a BytesIO handle.
    """
    url = f"{BASE_URL}{name}.tsv.gz"
    log(f"Downloading {name} archive from {url}", detail=False)

    buffer = io.BytesIO()
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        chunk_size = 8192
        for chunk in response.iter_content(chunk_size=chunk_size):
            buffer.write(chunk)

    buffer.seek(0)
    log(f"Finished downloading {name}")
    return buffer


def read_imdb_tsv(
    name,
    *,
    usecols=None,
    dtype=None,
):
    """
    Download an IMDb TSV.GZ archive and return it as a pandas DataFrame.
    """
    buffer = download_imdb_archive(name)
    log(f"Parsing {name} into DataFrame")
    df = pd.read_csv(
        buffer,
        sep="\t",
        compression="gzip",
        na_values="\\N",
        low_memory=False,
        usecols=usecols,
        dtype=dtype,
    )
    log(f"Loaded {format_count(len(df))} rows from {name}")
    return df


def load_relevant_people(relevant_ids):
    """
    Download name.basics and keep only rows whose nconst is in relevant_ids.
    Processes the file in chunks to keep memory usage manageable.
    """
    if not relevant_ids:
        log("No relevant IDs found; skipping name loading")
        return pd.DataFrame(columns=["nconst", "primaryName"]).astype(
            {"nconst": "string", "primaryName": "string"}
        )

    buffer = download_imdb_archive("name.basics")

    log("Scanning name.basics in chunks")
    reader = pd.read_csv(
        buffer,
        sep="\t",
        compression="gzip",
        na_values="\\N",
        usecols=["nconst", "primaryName"],
        dtype={"nconst": "string", "primaryName": "string"},
        chunksize=250_000,
        low_memory=False,
    )

    frames = []
    remaining_ids = set(relevant_ids)
    total_matched = 0

    for idx, chunk in enumerate(reader, start=1):
        chunk = chunk.dropna(subset=["nconst", "primaryName"])
        matches = chunk[chunk["nconst"].isin(remaining_ids)]
        if not matches.empty:
            frames.append(matches)
            matched_ids = set(matches["nconst"])
            remaining_ids.difference_update(matched_ids)
            total_matched += len(matches)
        if idx % 10 == 0:
            log(
                f"Processed {format_count(idx * 250_000)} rows; "
                f"{format_count(len(remaining_ids))} IDs still missing"
            )
        if not remaining_ids:
            log("Found all relevant names; stopping early")
            break

    if frames:
        relevant_people = pd.concat(frames, ignore_index=True)
    else:
        relevant_people = pd.DataFrame(columns=["nconst", "primaryName"])

    if remaining_ids:
        log(
            f"Warning: {format_count(len(remaining_ids))} IDs not found in name.basics",
            detail=False,
        )

    log(f"Matched {format_count(total_matched)} people rows")
    return relevant_people.astype({"nconst": "string", "primaryName": "string"})


def build_imdb_csv(force=False, verbose=False):
    """
    Build the processed CSV if it doesn't exist (or if force=True).
    Returns the path to the CSV.

    Args:
        force: Rebuild cached data when True.
        verbose: Print detailed status messages when True.
    """
    if _VERBOSE != verbose:
        set_verbosity(verbose)

    if CSV_PATH.exists() and not force:
        log(f"Using cached CSV at {CSV_PATH}", detail=False)
        return CSV_PATH

    log("Building IMDb dataset, this will take a couple of minutes :)", detail=False)
    basics = read_imdb_tsv("title.basics")
    ratings = read_imdb_tsv("title.ratings")
    crew = read_imdb_tsv("title.crew")

    log("Merging title tables")
    df = basics.merge(ratings, on="tconst", how="inner")
    df = df.merge(crew, on="tconst", how="left")

    log("Applying title filters")
    df = df[
        (df["isAdult"] == 0)
        & (df["titleType"] == "movie")
        & df["runtimeMinutes"].notna()
        & (df["startYear"] >= 1930)
        & (df["numVotes"] >= 1000)
    ].copy()

    log("Collecting director and writer IDs")
    relevant_ids = set()
    for column in ("directors", "writers"):
        ids = (
            df[column]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
            .dropna()
        )
        ids = ids[ids != ""]
        relevant_ids.update(ids)
    log(f"Unique people IDs collected: {format_count(len(relevant_ids))}")

    relevant_people = load_relevant_people(relevant_ids)

    if not relevant_people.empty:
        log("Disambiguating duplicate names")
        dup_counts = relevant_people["primaryName"].value_counts()
        dups = dup_counts[dup_counts > 1].index

        counter = {}

        def disambiguate(name):
            """Append (n) to duplicate names in order of appearance."""
            if pd.isna(name) or name not in dups:
                return name
            counter[name] = counter.get(name, 0) + 1
            return f"{name} ({counter[name]})"

        relevant_people["primaryName"] = relevant_people["primaryName"].apply(disambiguate)
    else:
        log("No relevant people found; skipping name disambiguation")

    log("Building ID → name lookup")
    name_map = (
        relevant_people[["nconst", "primaryName"]]
        .dropna()
        .set_index("nconst")["primaryName"]
    )

    def ids_to_names(series, mapping):
        """
        Convert a Series of comma-separated nconst IDs into comma-separated names.
        """
        nonnull = series.dropna()
        mapped = (
            nonnull.str.split(",")
            .explode()
            .map(mapping)
            .dropna()
            .groupby(level=0)
            .agg(",".join)
        )

        out = pd.Series(index=series.index, dtype="string")
        out.loc[mapped.index] = mapped.astype("string")
        return out

    df["directors"] = ids_to_names(df["directors"], name_map).astype("string")
    df["writers"] = ids_to_names(df["writers"], name_map).astype("string")

    for column in ("titleType", "isAdult", "endYear", "tconst"):
        if column in df.columns:
            df = df.drop(columns=column)

    df = df.astype(DTYPES, errors="ignore")

    log(f"Writing processed CSV to {CSV_PATH}", detail=False)
    df.to_csv(CSV_PATH, index=False)
    return CSV_PATH


def get_imdb_data(force=False, verbose=False):
    """
    Public entry point.
    If the processed CSV exists, load it.
    Otherwise, build it from scratch.

    Args:
        force: Rebuild cached data when True.
        verbose: Print detailed status messages when True.
    """
    set_verbosity(verbose)
    csv_path = build_imdb_csv(force=force, verbose=verbose)
    return pd.read_csv(csv_path, dtype=DTYPES, low_memory=False)


def main():
    parser = argparse.ArgumentParser(
        description="Download, preprocess, and cache an IMDb movies dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the processed CSV even if a cached copy exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    args = parser.parse_args()

    df = get_imdb_data(force=args.force, verbose=args.verbose)
    log(
        f"Finished. Rows: {format_count(len(df))}; columns: {len(df.columns)}",
        detail=False,
    )


if __name__ == "__main__":
    main()
