import torch
import torch.nn as nn

class SwitchableAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, hidden_size=58, block_size=128):
        super(SwitchableAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        # Define linear transformation layers
        self.query_linear = nn.Linear(query_size, hidden_size)
        self.key_linear = nn.Linear(key_size, hidden_size)
        self.switch = nn.Linear(block_size, 1)
        self.value_linear = nn.Linear(value_size, 512)
        self.block_size = block_size

    def forward(self, Q, K, V):
        # Linear transformations
        Q_transformed = self.query_linear(Q.unsqueeze(1))
        K_transformed = self.key_linear(K)
        V_transformed = self.value_linear(V)

        # Compute attention scores
        attn_scores = torch.matmul(Q_transformed, K_transformed.transpose(-2, -1)) / (K.size(-2) ** 0.5)
        attn_weights = torch.sigmoid(attn_scores)

        if attn_weights.size(2) < self.value_size:
            fill_weights = torch.zeros((attn_weights.size(0), attn_weights.size(1), self.block_size - attn_weights.size(2)))
            fill_weights = fill_weights.to(attn_weights.device)
            attn_weights = torch.cat((attn_weights, fill_weights), dim=-1)
        elif attn_weights.size(2) > self.block_size:
            attn_weights = attn_weights[:, :, :self.block_size]

        # Use attention weights to perform weighted sum of value vectors V
        hot = torch.sigmoid(self.switch(attn_weights))
        image_context = torch.mul(hot, V_transformed[:, :197, :])
        text_context = torch.mul(1 - hot, V_transformed[:, 197:, :])
        context = torch.cat((image_context, text_context), dim=1)
        return context, attn_weights

if __name__ == '__main__':
    # Example usage
    Q = torch.randn(2, 768)  # Query vector, dimension (batch_size, query_size)
    K = torch.randn(2, 58, 1536)  # Key vector, dimension (batch_size, num_keys, key_size)
    V = torch.randn(2, 265, 768)  # Value vector, dimension (batch_size, num_values, value_size)

    # Create self-attention module
    self_attention = SwitchableAttention(query_size=768, key_size=1536, value_size=768)

    # Run self-attention mechanism
    output, attn_scores = self_attention(Q, K, V)
    print(output.shape)  # Output shape is (batch_size, num_queries, value_size)
    print(attn_scores.shape)  # Output shape is (batch_size, num_queries, num_keys)
