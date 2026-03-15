from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from collections import namedtuple

class TokenGenerator:
    """
    A class for tokenizing text inputs using a pretrained transformer model.

    Attributes:
    -----------
    pretrained_model : str
        The identifier of the pretrained transformer model.
    batch_size : int
        The number of text samples to process in a single batch.
    max_length : int
        The maximum sequence length for tokenization.
    padding : bool or str
        The padding strategy ('max_length', 'longest' (or True)).
    truncation : bool
        Whether to truncate inputs to max_length.
    tokenizer : AutoTokenizer
        The tokenizer instance loaded from the Hugging Face Transformers library.
    Encoding : namedtuple
        A named tuple structure for storing tokenized inputs (input_ids, attention_mask).
    """
    
    def __init__(self, pretrained_model, batch_size=8, tokenizer_max_len=512, tokenizer_padding="max_length", tokenizer_truncation=True):
        """
        Initialize with tokenizer and configuration settings.

        Parameters:
        -----------
        pretrained_model : str
            The name or path of the pretrained transformer model.
        batch_size : int, optional (default=8)
            The batch size for processing text inputs.
        tokenizer_max_len : int, optional (default=512)
            The maximum length for tokenized sequences.
        tokenizer_padding : bool or str, optional (default=`max_length`)
            The padding strategy ('max_length', 'longest' (or True)).
        tokenizer_truncation : bool, optional (default=True)
            Whether to truncate inputs to max_length.
        """

        self.pretrained_model= pretrained_model # pretrained model to generate embeddings
        self.batch_size = batch_size # generate embeddings in batches
        self.max_length =       tokenizer_max_len
        self.padding =          tokenizer_padding
        self.truncation =       tokenizer_truncation

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token= self.tokenizer.eos_token
        self.Encoding = namedtuple('Encoding', ['input_ids', 'attention_mask'])
        
        

    def tokenize_texts(self, texts, output=list):
        """
        Tokenize a list of text inputs in batches.

        Parameters:
        -----------
        texts : list of str
            A list of text strings to tokenize.
        output : type, optional (default=list)
            The type of the output (dict or list of namedtuples).

        Returns:
        --------
        dict or list
            - If output is dict: A dictionary with 'input_ids' and 'attention_mask' as tensor lists.
            - If output is list: A list of namedtuples with 'input_ids' and 'attention_mask' attributes.
        """

        if output == dict:
            all_tokenized = {'input_ids': [], 'attention_mask': []}

            # Process batches of texts
            for start_idx in tqdm(range(0, len(texts), self.batch_size)):
                batch_tokenized = self._process_batch(texts, start_idx)
                all_tokenized['input_ids'].extend(batch_tokenized['input_ids'])
                all_tokenized['attention_mask'].extend(batch_tokenized['attention_mask'])

            # Convert lists to tensors for compatibility with model inputs
            all_tokenized['input_ids'] = torch.tensor(all_tokenized['input_ids'])
            all_tokenized['attention_mask'] = torch.tensor(all_tokenized['attention_mask'])

            return all_tokenized
        elif output == list:
            all_tokenized = []

            # Process batches of texts
            for start_idx in tqdm(range(0, len(texts), self.batch_size)):
                batch_tokenized = self._process_batch(texts, start_idx)
                # Create TokenizedText instances for each item in the batch
                for i in range(len(batch_tokenized['input_ids'])):
                    all_tokenized.append(
                        self.Encoding(  input_ids=batch_tokenized['input_ids'][i], 
                                        attention_mask=batch_tokenized['attention_mask'][i])
                    )

            return all_tokenized


    def _process_batch(self, texts, start_idx):
        """
        Tokenize a single batch of texts.

        Parameters:
        -----------
        texts : list of str
            A list of all text strings.
        start_idx : int
            The starting index of the batch.

        Returns:
        --------
        dict
            A dictionary containing 'input_ids' and 'attention_mask' for the tokenized batch.
        """
        end_idx = start_idx + self.batch_size
        batch_texts = texts[start_idx:end_idx]

        # Tokenize the batch
        return self.tokenizer(
            batch_texts,
            max_length= self.max_length,
            padding=    self.padding,
            truncation= self.truncation,
            return_tensors= "pt"  # Keeps output as lists for flexibility
        ).to("cpu")
