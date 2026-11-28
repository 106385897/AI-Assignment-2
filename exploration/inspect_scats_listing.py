import pandas as pd


def main():

    listing = pd.read_excel(
        "data/SCATSSiteListingSpreadsheet_VicRoads.xls",
        sheet_name="SCATS Site Numbers",
        engine="calamine",
        header=None,
        skiprows=10
    )

    listing = listing.iloc[:, :5].copy()

    listing.columns = [
        "SCATS Number",
        "Description",
        "Site Type",
        "Directory",
        "Map Reference"
    ]

    listing["SCATS Number"] = pd.to_numeric(
        listing["SCATS Number"],
        errors="coerce"
    )

    listing = listing.dropna(subset=["SCATS Number"])
    listing["SCATS Number"] = listing["SCATS Number"].astype(int)

    # Keep one record for each SCATS site
    listing = listing.drop_duplicates(
        subset=["SCATS Number"],
        keep="first"
    )

    intersections = pd.read_csv(
        "data/scats_intersections.csv"
    )

    sites = intersections.merge(
        listing,
        on="SCATS Number",
        how="left",
        validate="one_to_one"
    )

    print("Total SCATS sites:", len(sites))
    print("Unique SCATS sites:", sites["SCATS Number"].nunique())
    print("Sites matched:", sites["Description"].notna().sum())
    print("Sites missing descriptions:", sites["Description"].isna().sum())

    print("\nMatched intersections:")
    print(
        sites[
            ["SCATS Number", "Description", "Latitude", "Longitude"]
        ].to_string(index=False)
    )

    sites.to_csv(
        "data/scats_sites_complete.csv",
        index=False
    )

    print("\nSaved to data/scats_sites_complete.csv")


if __name__ == "__main__":
    main()