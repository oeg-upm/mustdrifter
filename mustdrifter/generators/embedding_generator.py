import torch
from tqdm import tqdm
from transformers import AutoModel

class EmbeddingsGenerator:
    """
    A class for generating text embeddings using a pretrained transformer model.

    Attributes:
    -----------
    train_device : str
        The device to use for computation ('cpu' or 'cuda').
    pretrained_model : str
        The identifier of the pretrained transformer model.
    batch_size : int
        The number of input samples to process in a single batch.
    pooling_mode : str
        The pooling strategy to apply to the embeddings. Options:
        - 'cls_pooling': Use the [CLS] token embedding.
        - 'mean_pooling': Average all token embeddings.
        - 'no_pooling': Return all token embeddings.
    device : torch.device
        The PyTorch device to which the model is loaded.
    model : AutoModel
        The transformer model loaded from the Hugging Face Transformers library.
    encoder : AutoModel
        The encoder module of the model, used for generating embeddings.
    """

    def __init__(self, pretrained_model, train_device="cpu", batch_size=8, pooling_mode="cls_pooling"):
        """
        Initializes the EmbeddingsGenerator with a pretrained model.

        Parameters:
        -----------
        pretrained_model : str
            The name or path of the pretrained transformer model.
        train_device : str, optional (default='cpu')
            The device to use for computation ('cpu' or 'cuda').
        batch_size : int, optional (default=8)
            The batch size for processing inputs.
        pooling_mode : str, optional (default='cls_pooling')
            The pooling strategy ('cls_pooling', 'mean_pooling', or 'no_pooling').
        """
        self.train_device= train_device # device to use: cpu or cuda
        self.pretrained_model= pretrained_model # pretrained model to generate embeddings
        self.batch_size = batch_size # generate embeddings in batches
        self.pooling_mode= pooling_mode # pooling strategy: cls_pooling, mean_pooling, no_pooling (each token embedding)

        self.device = torch.device(self.train_device) # Set the device

        self.model = AutoModel.from_pretrained(self.pretrained_model).to(self.device) # Load the pretrained model in the device
        self.encoder = self._get_encoder()        

    def generate_embeddings(self, inputs_tokenized):
        """
        Generate embeddings for tokenized inputs.

        Parameters:
        -----------
        inputs_tokenized : list
            A list of tokenized inputs, where each input is an encoding with 'input_ids' and 'attention_mask'.

        Returns:
        --------
        torch.Tensor
            The generated embeddings as a tensor.
        """
        all_embeddings = []

        # Process batches
        for start_idx in tqdm(range(0, len(inputs_tokenized), self.batch_size)):
            batch_embeddings = self._process_batch(inputs_tokenized, start_idx).to("cpu")
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings into a single tensor
        return torch.cat(all_embeddings, dim=0).to("cpu")

    def _process_batch(self, inputs_tokenized, start_idx):
        """
        Process a single batch of inputs and generate embeddings.

        Parameters:
        -----------
        inputs_tokenized : list
            A list of tokenized inputs.
        start_idx : int
            The starting index of the batch.

        Returns:
        --------
        torch.Tensor
            The processed batch embeddings.
        """
        end_idx = start_idx + self.batch_size
        batch_inputs_tokenized = torch.stack([_encoding.input_ids for _encoding in inputs_tokenized[start_idx:end_idx]], dim=0).to(self.device)
        
        batch_inputs_attention_mask = torch.stack([_encoding.attention_mask for _encoding in inputs_tokenized[start_idx:end_idx]], dim=0).to(self.device)

        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=batch_inputs_tokenized, attention_mask=batch_inputs_attention_mask)
            embeddings = encoder_outputs.last_hidden_state

        # Pool embeddings as per the specified pooling mode
        return self._apply_pooling(embeddings)

    def _get_encoder(self):
        """
        Retrieve the encoder module of the transformer model.

        Returns:
        --------
        AutoModel
            The encoder model.
        """
        if self.model.config.is_encoder_decoder:
            self.encoder= self.model.encoder 
        else:
            self.encoder= self.model
        return self.encoder

    def _apply_pooling(self, embeddings):
        """
        Apply the specified pooling strategy to the embeddings.

        Parameters:
        -----------
        embeddings : torch.Tensor
            The token embeddings from the transformer model.

        Returns:
        --------
        torch.Tensor
            The pooled embeddings.
        """
        if self.pooling_mode == "mean_pooling":
            return torch.mean(embeddings, dim=1)
        elif self.pooling_mode == "cls_pooling":
            return embeddings[:, 0, :]
        elif self.pooling_mode == "no_pooling":
            return embeddings
        else:
            raise ValueError(f"Invalid pooling mode: {self.pooling_mode}")
