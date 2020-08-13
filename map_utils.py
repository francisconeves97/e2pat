'''
@info utilities for map design
@author Rui Henriques
@version 1.0
'''

import folium
import dash_html_components as html


def get_lisbon_map():
    return folium.Map(location=[38.74, -9.14], zoom_start=12, tiles="cartodbpositron")


def embed_map(fmap, prefix='', height='700'):
    map_url = prefix + "temp_map.html"
    fmap.save(map_url)
    return html.Div(html.Iframe(id=prefix + 'map', srcDoc=open(map_url, 'r').read(), width='100%', height=height),
                    style={'margin-top': '20px'})
