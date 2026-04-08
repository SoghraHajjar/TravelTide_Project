import reverse_geocoder as rg
import math
import pandas as pd

# Function to get country code from latitude and longitude


def get_country_code(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    try:
        # rg.search expects a tuple of (latitude, longitude)
        result = rg.search((lat, lon))
        if result and len(result) > 0:
            return result[0]['cc']  # 'cc' stands for country code
        return None
    except Exception as e:
        print(f"Error geocoding ({lat}, {lon}): {e}")
        return None

# haversine formula to calculate distance between two points on the Earth


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Erdradius in km
    # Grad -> Radiant
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance
