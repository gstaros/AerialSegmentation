from .unet import UNet, UNetPlusPlus
from .vit import ViTSegmenter, ResViTSegmenter

def load_model(model_type, config):

    assert model_type in ['UNet', 'UNet++', 'ViTSegmenter', 'ResViTSegmenter']
    if model_type == 'UNet':
        parameters = config['MODEL']['UNet']
        model = UNet(
                    in_channels = parameters['in_channels'],
                    num_classes = parameters['num_classes']
                )
        
    if model_type == 'UNet++':
        parameters = config['MODEL']['UNet++']
        model = UNetPlusPlus(
                    in_channels = parameters['in_channels'],
                    num_classes = parameters['num_classes']
                )
        
    if model_type == 'ViTSegmenter':
        parameters = config['MODEL']['ViTSegmenter']
        model = ViTSegmenter(
                    image_size = parameters['image_size'],
                    patch_size = parameters['patch_size'],
                    in_channels = parameters['in_channels'],
                    num_classes = parameters['num_classes'],
                    embedding_dim = parameters['embedding_dim'],
                    num_layers = parameters['num_layers'],
                    num_heads = parameters['num_heads'],
                    hidden_dim = parameters['hidden_dim'],
                    attention_dropout = parameters['attention_dropout'],
                    fc_out_dropout = parameters['fc_out_dropout'],
                    mlp_dropout = parameters['mlp_dropout'],
                )
        
    if model_type == 'ResViTSegmenter':
        parameters = config['MODEL']['ResViTSegmenter']
        model = ResViTSegmenter(
                    image_size = parameters['image_size'],
                    patch_size_list = parameters['patch_size_list'],
                    in_channels = parameters['in_channels'],
                    num_classes = parameters['num_classes'],
                    embedding_dim = parameters['embedding_dim'],
                    num_layers = parameters['num_layers'],
                    num_heads = parameters['num_heads'],
                    hidden_dim = parameters['hidden_dim'],
                    attention_dropout = parameters['attention_dropout'],
                    fc_out_dropout = parameters['fc_out_dropout'],
                    mlp_dropout = parameters['mlp_dropout'],
                )
    

    return model