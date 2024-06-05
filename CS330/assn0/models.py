"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        # Check if we are going to share embeddings
        self.user_ranking_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_ranking_embeddings = ScaledEmbedding(num_items, embedding_dim)
        # Embedding functions for users and items
        if embedding_sharing:
            self.user_scoring_embeddings = self.user_ranking_embeddings
            self.item_scoring_embeddings = self.item_ranking_embeddings
        else:
            self.user_scoring_embeddings = ScaledEmbedding(num_users, embedding_dim)
            self.item_scoring_embeddings = ScaledEmbedding(num_items, embedding_dim)
        

        # Scoring model layers
        self.scoring_model_layers = []
        for i in range(len(layer_sizes) - 1):
            self.scoring_model_layers.append( nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias = True) )
        self.scoring_model_layers.append( nn.Linear(layer_sizes[-1], 1, bias = True) )
        self.activation_function = nn.ReLU()

        # Ranking model parameters
        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        #********************************************************
        #********************************************************
        #********************************************************
        
        
        

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        # print(user_ids.shape) [256]
        # print(item_ids.shape) [256]
        
        user_scoring_embeddings = self.user_scoring_embeddings(user_ids)
        item_scoring_embeddings = self.item_scoring_embeddings(item_ids)
        user_ranking_embeddings = self.user_ranking_embeddings(user_ids)
        item_ranking_embeddings = self.item_ranking_embeddings(item_ids)
        user_bias_embeddings = self.user_biases(user_ids).squeeze()
        item_bias_embeddings = self.item_biases(item_ids).squeeze()
        # print(user_scoring_embeddings.shape) [256, 32]
        
        predictions = torch.matmul(user_ranking_embeddings.unsqueeze(1), item_ranking_embeddings.unsqueeze(2)).squeeze() + user_bias_embeddings + item_bias_embeddings

        mlp_input = torch.cat(
            [
                user_scoring_embeddings,
                item_scoring_embeddings,
                torch.mul(user_scoring_embeddings, item_scoring_embeddings)
            ],
            1
        )

        for i, scoring_model_layer in enumerate(self.scoring_model_layers):
            if i == 0:
                output = scoring_model_layer(mlp_input)
            else:
                output = scoring_model_layer(output)
            output = self.activation_function(output)
            
        score = output.squeeze()


        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score