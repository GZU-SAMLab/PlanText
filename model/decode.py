import torch
from torch import nn
from model.pre_model import gpt2
import torch.nn.functional as F
from model.attention import Atten
from torch.autograd import Variable
from model.switch_attention import SwitchableAttention


class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout applied before affine transformation
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_t, h_t_1, cell_t):
        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(self.dropout(h_t_1))
        gate_t = torch.sigmoid(gate_t)

        # Sentinel embedding
        s_t = gate_t * torch.tanh(cell_t)

        return s_t


class SelectBlock(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size):
        super(SelectBlock, self).__init__()

        # Sentinel block
        self.sentinel = Sentinel(embed_size * 2, hidden_size)
        self.G_t = nn.Linear(embed_size, hidden_size)

        # Switchable Attention
        self.switch_atten = SwitchableAttention(768, 1536, 768)
        # Image Spatial Attention Block
        self.atten = Atten(hidden_size)

        # Final Caption generator
        self.mlp = nn.Linear(hidden_size, vocab_size)

        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout(0.5)

        self.hidden_size = hidden_size

    def forward(self, x, hiddens, cells, G, H):

        # hidden for sentinel should be h0-ht-1
        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)

        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens.size(1) > 1:
            hiddens_t_1 = torch.cat((h0, hiddens[:, :-1, :]), dim=1)
        else:
            hiddens_t_1 = h0

        # Get Sentinel embedding, it's calculated blockly
        sentinel = self.sentinel(x, hiddens_t_1, cells)

        # Switchable Attention
        switch_output, switch_weights = self.switch_atten(H, x, G)

        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights, beta = self.atten(switch_output, hiddens, sentinel)

        # Final score along vocabulary
        scores = self.mlp(self.dropout(c_hat + hiddens))

        return scores, atten_weights, beta, switch_weights

    def init_hidden(self, bsz):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Decoder, self).__init__()

        # word embedding
        self.embed = gpt2.wte
        embed_size = gpt2.wte.embedding_dim
        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM(embed_size * 2, hidden_size, 1, batch_first=True)

        # Save hidden_size for hidden and cell variable
        self.hidden_size = hidden_size

        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.select = SelectBlock(embed_size, hidden_size, vocab_size)

    def forward(self, V, v_g, captions, states=None):

        # Word Embedding
        embeddings = self.embed(captions)

        # x_t = [w_t;v_g]
        x = torch.cat((embeddings, v_g.unsqueeze(1).expand_as(embeddings)), dim=2)

        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda())
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size))

            # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range(x.size(1)):
            # Feed in x_t one at a time
            x_t = x[:, time_step, :]
            x_t = x_t

            h_t, states = self.LSTM(x_t, states)

            # Save hidden and cell
            hiddens[:, time_step, :] = h_t  # Batch_first
            cells[time_step, :, :] = states[1]

        # cell: Batch x seq_len x hidden_size
        cells = cells.transpose(0, 1)

        # Data parallelism for adaptive attention block
        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            adaptive_block_parallel = nn.DataParallel(self.select, device_ids=device_ids)

            scores, atten_weights, beta, switch_weights = adaptive_block_parallel(x, hiddens, cells, V, v_g)
        else:
            scores, atten_weights, beta, switch_weights = self.select(x, hiddens, cells, V, v_g)

        return scores, states, atten_weights, beta
