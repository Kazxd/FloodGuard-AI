import osmnx as ox

place = "Velachery, Chennai, Tamil Nadu, India"

# Try the default Overpass server first
try:
    water = ox.features_from_place(
        place,
        tags={
            "natural": ["water", "wetland"],
            "water": ["lake", "reservoir"],
        },
    )
except Exception as e:
    print("Default Overpass failed:", e)
    print("Retrying with the Kumi mirror...")
    ox.settings.overpass_endpoint = (
        "https://overpass.kumi.systems/api"
    )
    water = ox.features_from_place(
        place,
        tags={
            "natural": ["water", "wetland"],
            "water": ["lake", "reservoir"],
        },
    )

if water.empty:
    raise RuntimeError("No water features found.")

output = "data/raw/lakes_wetlands.geojson"
water.to_file(output, driver="GeoJSON")

print(f"Saved {len(water)} features to {output}")
print(water[["geometry"]].head())
