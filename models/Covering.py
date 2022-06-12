import simulated_bifurcation as sb
import numpy as np
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from math import pi

class VertexCover(sb.SBModel):

    def __init__(self, graph_connections: np.ndarray, labels = None) -> None:
        self.graph_connections = graph_connections
        self.vertices = graph_connections.shape[0]
        self.edges = int(graph_connections.sum() // 2)
        self.colored = None
        if labels is None: self.labels = [str(n) for n in range(self.vertices)]
        else: self.labels = labels

    def __str__(self) -> str: return f'{self.vertices} vertices - {self.edges} edges'

    def __to_Ising__(self) -> sb.Ising:
        J = -.5 * self.graph_connections
        h = -.5 * self.graph_connections @ np.ones((self.vertices, 1)) + .05 * np.ones((self.vertices, 1))
        return sb.Ising(J, h)

    def __from_Ising__(self, ising: sb.Ising) -> None:
        self.colored = np.where((ising.ground_state == 1).reshape(-1,))[0]

    def show(self):

        elements = []
        angle = 2 * pi / self.vertices
        radius = 100 + 5 * self.vertices

        for vertex in range(self.vertices):

            if vertex in self.colored: category = 'colored'; coeff = 1
            else: category = 'uncolored'; coeff = 1.5

            elements.append(
                {
                    'data': {'id': self.labels[vertex], 'label': self.labels[vertex]},
                    'position': {'x': coeff * radius * np.cos(angle * vertex), 'y': coeff * radius * np.sin(angle * vertex)},
                    'classes': category
                }
            )

            for edge in range(vertex + 1, self.vertices):
                if edge in self.colored or vertex in self.colored: category = 'colored'
                else: category = 'uncolored'
                if self.graph_connections[vertex, edge] == 1: elements.append({'data': {'source': self.labels[vertex], 'target': self.labels[edge]}, 'classes': category})

        app = dash.Dash(__name__)  
        app.layout = html.Div([
            cyto.Cytoscape(
                id='cytoscape-styling-1',
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '1000px'},
                elements=elements,
                stylesheet=[
                    # Group selectors
                    {
                        'selector': 'node',
                        'style': {
                            'content': 'data(label)'
                        }
                    },

                    # Class selectors
                    {
                        'selector': '.colored',
                        'style': {
                            'background-color': 'red',
                            'line-color': 'red'
                        }
                    },
                    
                ]
            )
        ])

        app.run_server(debug=True)

        