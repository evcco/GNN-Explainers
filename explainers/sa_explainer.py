from torch.autograd import Variable
from explainers.basic_explainer import Explainer

class SAExplainer(Explainer):

    def __init__(self, gnn_model_path, gen_model_path=None):
        super(SAExplainer, self).__init__(gnn_model_path, gen_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      draw_graph=0,
                      vis_ratio=0.2):

        if model == None:
            model = self.model
            
        tmp_graph = graph.clone()
        
        tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        tmp_graph.x = Variable(tmp_graph.x, requires_grad=True)
        pred = model(tmp_graph)
        pred[0, tmp_graph.y].backward()
        
        edge_grads = self.norm_imp(graph.ground_truth_mask[0])#pow(tmp_graph.edge_attr.grad, 2).sum(dim=1).cpu().numpy()
        edge_imp = self.norm_imp(edge_grads)
        self.last_result = (graph, edge_imp)
        
        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
            
        return edge_imp