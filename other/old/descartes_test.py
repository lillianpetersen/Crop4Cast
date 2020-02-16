import os
import warnings
from pprint import pprint
import descarteslabs as dl
matches = dl.places.find('puerto-rico')
pprint(matches)
# The first one looks good to me, so lets make that our area of interest.
aoi = matches[0]
shape = dl.places.shape(aoi['slug'], geom='low')

import json
feature_collection = dl.metadata.search(products='landsat:LC08:PRE:TOAR', start_time='2017-03-12',
                                        end_time='2017-04-30', limit=10, place=aoi['slug'])
# As the variable name implies, this returns a FeatureCollection GeoJSON dictionary.
# Its 'features' are the available scenes.
print len(feature_collection['features'])
# The 'id' associated with each feature is a unique identifier into our imagery database.
# In this case there are two L8 scenes from adjoining WRS rows.

f0 = feature_collection['features'][0]

band_information = dl.raster.get_bands_by_key(feature_collection['features'][0]['id'])

ids = [f['id'] for f in feature_collection['features']]
# Rasterize the features.
#  * Select red, green, blue, alpha
#  * Scale the incoming data with range [0, 10000] down to [0, 4000] (40% TOAR)
#  * Choose an output type of "Byte" (uint8)
#  * Choose 60m resolution
#  * Apply a cutline of Taos county
arr, meta = dl.raster.ndarray(
    ids,
    bands=['red', 'green', 'blue', 'alpha'],
    scales=[[0,4000], [0, 4000], [0, 4000], None],
    data_type='Byte',
    resolution=60,
    cutline=shape['geometry'],
)

plt.figure(figsize=[16,16])
plt.imshow(arr)
plt.savefig('../figures/puerto-rico')
