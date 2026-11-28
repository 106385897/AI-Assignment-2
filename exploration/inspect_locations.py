import pandas as pd


FILE_PATH = "Scats_Data_October_2006.xls"


def main():

    df = pd.read_excel(
        FILE_PATH,
        sheet_name="Data",
        header=1
    )

    columns = [
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE"
    ]

    sites = df[columns].drop_duplicates().copy()

    sites["NB_LATITUDE"] = pd.to_numeric(
        sites["NB_LATITUDE"], errors="coerce"
    )

    sites["NB_LONGITUDE"] = pd.to_numeric(
        sites["NB_LONGITUDE"], errors="coerce"
    )

    sites = sites.dropna(
        subset=["NB_LATITUDE", "NB_LONGITUDE"]
    )

    # Exclude invalid coordinates
    sites = sites[
        (sites["NB_LATITUDE"].between(-38.5, -37.0))
        & (sites["NB_LONGITUDE"].between(144.0, 146.0))
    ]

    # Create one representative point per SCATS site
    intersections = sites.groupby(
        "SCATS Number", as_index=False
    ).agg(
        Latitude=("NB_LATITUDE", "mean"),
        Longitude=("NB_LONGITUDE", "mean"),
        DetectorCount=("Location", "count")
    )

    intersections = intersections.sort_values("SCATS Number")

    print("Total SCATS intersections:", len(intersections))

    print("\nIntersection coordinates:")
    print(intersections.to_string(index=False))

    intersections.to_csv(
        "data/scats_intersections.csv",
        index=False
    )

    print("\nSaved to data/scats_intersections.csv")


if __name__ == "__main__":
    main()