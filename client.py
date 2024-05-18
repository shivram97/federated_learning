from centralized import fit_evaluate, load_model,train,test
from collections import OrderedDict

import flwr as fl
import torch 

def set_parameters(model,paramerters):
    params_dict = zip(model.state_dict().keys(),paramerters)
    state_dict = OrderedDict({k: torch.tensor(v) for k,v in params_dict})
    model.load_state_dict(state_dict,strict=True)
    return model

net= load_model()


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _,val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        return fit_evaluate(parameters, config)

    def evaluate(self, parameters, config):
        return fit_evaluate(parameters, config)
    
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client= FlowerClient()
)