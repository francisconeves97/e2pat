import dash

app = dash.Dash(__name__, assets_folder='assets', include_assets_files=True)
# external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.config.suppress_callback_exceptions = True
