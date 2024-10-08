import torch
import torch.nn.functional as F
from torch.autograd import Variable
from explainers.basic_explainer import Explainer

class GradCam(Explainer):

    def __init__(self, gnn_model_path, gen_model_path=None):
        super(GradCam, self).__init__(gnn_model_path, gen_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      draw_graph=0,
                      vis_ratio=0.2):

        if model == None:
            model = self.model
            
        tmp_graph = graph.clone()
        tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        pred = model(tmp_graph)
        pred[0, graph.y].backward()
        edge_grads = tmp_graph.edge_attr.grad

        alpha = torch.mean(edge_grads, dim=1)
        edge_score = F.relu(torch.sum((graph.edge_attr.T * alpha).T, dim=1)).cpu().numpy()
        edge_score = self.norm_imp(edge_score)

        self.last_result = (graph, edge_score)
        if draw_graph:
            self.visualize(graph, edge_score, self.name, vis_ratio=vis_ratio)

        return edge_score