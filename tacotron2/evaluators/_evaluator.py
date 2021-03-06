import torch


class BaseEvaluator(object):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, tokenizer, denoiser=None, device='cpu'):

        self.encoder = encoder
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.denoiser = denoiser

        self.device = device

    def synthesize(self, text):

        with torch.no_grad():
            self.encoder.eval()
            self.vocoder.eval()
            sequence = torch.LongTensor(self.tokenizer.encode(text))\
                            .unsqueeze(0)\
                            .to(self.device)

            mel_outputs, mel_outputs_postnet, gates, alignments = self.encoder.inference(sequence)
            audio = self.vocoder.infer(mel_outputs_postnet, sigma=0.9)
            if self.denoiser:
                audio = self.denoiser(audio, strength=0.01)[:, 0]

            return audio, (mel_outputs_postnet, gates, alignments)


class EmbeddingEvaluator(BaseEvaluator):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, tokenizer, denoiser=None, device='cpu'):
        super().__init__(encoder, vocoder, tokenizer, denoiser, device)

    def synthesize(self, text, embedding):

        with torch.no_grad():
            self.encoder.eval()
            self.vocoder.eval()
            sequence = torch.LongTensor(self.tokenizer.encode(text))\
                            .unsqueeze(0)\
                            .to(self.device)

            embedding = torch.FloatTensor(embedding)\
                             .unsqueeze(0)\
                             .to(self.device)

            mel_outputs, mel_outputs_postnet, gates, alignments = self.encoder.inference(sequence, embedding)
            audio = self.vocoder.infer(mel_outputs_postnet, sigma=0.9)
            if self.denoiser:
                audio = self.denoiser(audio, strength=0.05)[:, 0]

            return audio, (mel_outputs_postnet, gates, alignments)