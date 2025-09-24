import torch
import torch.nn as nn


class IRTembed(nn.Module):
    """
    Combines a text encoder and LLM identity embeddings.
    Projects text embeddings to a vector: b ∈ ℝ^{B × d_embed}, and returns them
    along with the corresponding LLM embeddings.
    """

    def __init__(self,
                 llm_names,
                 embedding_dim,
                 llm_embed_dim,
                 dropout=0.2,
                 device='cpu',
                 ):
        super().__init__()

        self.device = device
        self.llm_embed_dim = llm_embed_dim
        self.embedding_dim = embedding_dim


        self.llm_name_to_idx = {name: i for i, name in enumerate(llm_names)}
        self.llm_embedding_table = nn.Embedding(len(llm_names), llm_embed_dim).to(self.device)

        self.output_dim = llm_embed_dim

        self.head = nn.Sequential(
            # nn.LayerNorm(embedding_dim),
            nn.Linear(self.embedding_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*self.embedding_dim, self.output_dim)
        ).to(self.device)



    def forward(self, embeddings, llm_name_strs):
        # embeddings = nn.functional.normalize(embeddings, dim=-1)

        b = self.head(embeddings.to(self.head[0].weight.dtype))
        

        llm_ids = torch.tensor(
            [self.llm_name_to_idx[name] for name in llm_name_strs],
            dtype=torch.long,
            device=self.device
        )
        llm_embeddings = self.llm_embedding_table(llm_ids)

        return b, llm_embeddings

    def add_llms(self, new_llm_names):
        new_names = [name for name in new_llm_names if name not in self.llm_name_to_idx]
        if not new_names:
            return

        current_size = self.llm_embedding_table.num_embeddings
        new_total = current_size + len(new_names)

        new_embedding = nn.Embedding(new_total, self.llm_embed_dim).to(self.device)
        with torch.no_grad():
            new_embedding.weight[:current_size] = self.llm_embedding_table.weight

        self.llm_embedding_table = new_embedding
        self.llm_name_to_idx.update({
            name: current_size + i for i, name in enumerate(new_names)
        })


    def freeze_head(self):
        """Freezes the MLP head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False


