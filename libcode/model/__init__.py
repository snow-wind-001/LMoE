from libcode.model.moe_agent import DeepSpeedAgent
# from libcode.model.openllama import LungsCancerMoE as modelmoe
# from libcode.model.openllama_tmp import OpenLLAMAPEFTModel
from libcode.model.lmoe import LungsCancerMoE as modelmoe
# from libcode.model.openllama_CLIP import OpenLLAMAPEFTModel_CLIP
from libcode.model.ImageBind import models
import importlib
def load_model(args):
    #如果是args['model'] == 'lmoe_CT'，则使用LMoEDataset
    if args['model'] == 'lmoe_CT':
        model = importlib.import_module("libcode.model.%s" % args['model'])
        modelmoe = getattr(model, 'train', None)
        model = modelmoe(args)
        agent = importlib.import_module("libcode.model.moe_agent")
        DeepSpeedAgent = getattr(agent, 'DeepSpeedAgent', None)
        agent = DeepSpeedAgent(model, args)
    if args['model'] == 'OpenLLAMAPEFTModel':
        model = importlib.import_module("libcode.model.openllama")
        OpenLLAMAPEFTModel = getattr(model, 'OpenLLAMAPEFTModel', None)
        model = OpenLLAMAPEFTModel(args)
        agent = importlib.import_module("libcode.model.agent")
        DeepSpeedAgent = getattr(agent, 'DeepSpeedAgent', None)
        agent = DeepSpeedAgent(model, args)

    # model = modelmoe(args)
    # agent = DeepSpeedAgent(model, args)
    return agent