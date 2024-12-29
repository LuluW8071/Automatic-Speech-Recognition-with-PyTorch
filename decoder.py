from torchaudio.models.decoder import ctc_decoder


class CTCBeamDecoder:
    """
    Class for performing CTC beam decoding with a language model.

    Args:
        beam_size (int): Beam size for beam search decoding.
        blank_id (int): Index of the blank label in the vocabulary.
        kenlm_path (str): Path to the KenLM language model file.
    """

    def __init__(self, beam_size=50, token_path=None, lexicon_path=None, kenlm_path=None):
        """
        Initializes the CTCBeamDecoder.

        Args:
            beam_size (int): Beam size for beam search decoding.
            blank_id (int): Index of the blank label in the vocabulary.
            kenlm_path (str): Path to the KenLM language model file.
            alpha (float): Scaling factor for language model score.
            beta (float): Scaling factor for length normalization.
        """
        print("loading beam search with lm...")

        self.decoder = ctc_decoder(
            lexicon=lexicon_path,
            tokens=token_path,
            lm=kenlm_path,
            nbest=1,
            beam_size=beam_size,
            beam_threshold=25,
            lm_weight=0.15,
            word_score=-0.26,
        )

    def __call__(self, output):
        """
        Perform beam search decoding on the given output.

        Args:
            output (torch.Tensor): Output tensor from the neural network.

        Returns:
            str: Decoded text sequence.
        """
        # Perform beam search decoding
        beam_result = self.decoder(output)
        return " ".join(beam_result[0][0].words).strip()