import geopandas as gpd

from physhade.config import ROOT


def find_95th_percentile(gdf: gpd.GeoDataFrame, column: str):
    """Find the 95th percentile of a column in a GeoDataFrame."""
    return gdf[column].quantile(0.95)


if __name__ == "__main__":
    bag_gdf = gpd.read_file(str(ROOT / "Dataset/3dbag_nl.gpkg"), layer="lod12_2d")

    print(find_95th_percentile(bag_gdf, "b3_h_70p"))
