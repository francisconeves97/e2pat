# -*- coding: utf-8 -*-

'''
@info folium utils
@author Francisco Neves
@version 1.0
'''

from branca.element import CssLink, Figure, JavascriptLink, MacroElement
from jinja2 import Template


class Draw(MacroElement):
    """
    Vector drawing and editing plugin for Leaflet.

    Parameters
    ----------
    export : bool, default False
        Add a small button that exports the drawn shapes as a geojson file.
    filename : string, default 'data.geojson'
        Name of geojson file
    position : {'topleft', 'toprigth', 'bottomleft', 'bottomright'}
        Position of control.
        See https://leafletjs.com/reference-1.5.1.html#control
    draw_options : dict, optional
        The options used to configure the draw toolbar. See
        http://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html#drawoptions
    edit_options : dict, optional
        The options used to configure the edit toolbar. See
        https://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html#editpolyoptions

    Examples
    --------
    >>> m = folium.Map()
    >>> Draw(
    ...     export=True,
    ...     filename='my_data.geojson',
    ...     position='topleft',
    ...     draw_options={'polyline': {'allowIntersection': False}},
    ...     edit_options={'poly': {'allowIntersection': False}}
    ... ).add_to(m)

    For more info please check
    https://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html

    """
    _template = Template(u"""
        {% macro script(this, kwargs) %}
            // https://stackoverflow.com/questions/30683628/react-js-setting-value-of-input
            function doEvent( obj, event ) {
                var event = new Event( event, {target: obj, bubbles: true} );
                return obj ? obj.dispatchEvent(event) : false;
            }

            var options = {
              position: {{ this.position|tojson }},
              draw: {{ this.draw_options|tojson }},
              edit: {{ this.edit_options|tojson }},
            }
            // FeatureGroup is to store editable layers.
            var drawnItems = new L.featureGroup().addTo(
                {{ this._parent.get_name() }}
            );
            options.edit.featureGroup = drawnItems;
            var {{ this.get_name() }} = new L.Control.Draw(
                options
            ).addTo( {{this._parent.get_name()}} );
            {{ this._parent.get_name() }}.on(L.Draw.Event.CREATED, function(e) {
                var layer = e.layer,
                    type = e.layerType;
                var coords = JSON.stringify(layer.toGeoJSON());
                layer.on('click', function() {
                    console.log(coords);
                });

                var el = parent.document.getElementById('{{ this.page_prefix }}geo_json');
                el.setAttribute('value', coords);
                doEvent( el, 'input' );

                drawnItems.addLayer(layer);
             });
            {{ this._parent.get_name() }}.on('draw:created', function(e) {
                drawnItems.clearLayers();
                drawnItems.addLayer(e.layer);
            });
        {% endmacro %}
        """)

    def __init__(self,
                 position='topleft', draw_options=None, edit_options=None, page_prefix=''):
        super(Draw, self).__init__()
        self._name = 'DrawControl'
        self.position = position
        self.draw_options = draw_options or {}
        self.edit_options = edit_options or {}
        self.page_prefix = page_prefix

    def render(self, **kwargs):
        super(Draw, self).render(**kwargs)

        figure = self.get_root()
        assert isinstance(figure, Figure), ('You cannot render this Element '
                                            'if it is not in a Figure.')

        figure.header.add_child(
            JavascriptLink('https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.2/leaflet.draw.js'))  # noqa
        figure.header.add_child(
            CssLink('https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.2/leaflet.draw.css'))  # noqa
