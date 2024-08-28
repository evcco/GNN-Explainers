from explainers.basic_explainer import Explainer
from explainers.meta_gnnexplainer import MetaGNNGExplainer

class GNNExplainer(Explainer):

    def __init__(self, gnn_model_path, gen_model_path=None):
        super(GNNExplainer, self).__init__(gnn_model_path, gen_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      epochs=200,
                      lr=1e-1,
                      draw_graph=0,
                      vis_ratio=0.2
                      ):

        if model == None:
            model = self.model

        explainer = MetaGNNGExplainer(model, epochs=epochs, lr=lr)
        edge_imp = explainer.explain_graph(graph)
        edge_imp = self.norm_imp(edge_imp.cpu().numpy())

        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)

        self.last_result = (graph, edge_imp)
        return edge_imp
