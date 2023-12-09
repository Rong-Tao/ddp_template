import torch
import torch.nn as nn
import torch.nn.functional as F

### Define your model here ###
def get_model():
    '''
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)
    '''
    model = Model_Class()
    return model

class Model_Class(nn.Module):
    def __init__(self):
        super(Model_Class, self).__init__(img_size = 512)
        # Define the layers
        self.cnn = nn.Conv2d(3,1,3,padding='same')

    def forward(self, x):
        return self.cnn(x)
