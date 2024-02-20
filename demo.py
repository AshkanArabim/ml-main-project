from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from itertools import product
import os


if __name__ == "__main__":
    encoder = VoiceEncoder()

    # make dictionary of audio files
    # note: put audio files in the root directory, and make sure they're `.wav` files
    voices = {filename[:-4]: filename for filename in os.listdir('.') if ".wav" in filename}

    # load and preprocess audio tracks
    for name in voices:
        voices[name] = preprocess_wav(voices[name])

    # embed speaker (actual physical voice)
    for name in voices:
        voices[name] = encoder.embed_speaker([voices[name]])

    people_combos = {frozenset((s1, s2)) for s1, s2 in product(voices.keys(), voices.keys()) if s1 != s2}
        
    for p1, p2 in people_combos:
        similarity = np.dot(voices[p1], voices[p2])  # using dot product
        # similarity = np.dot(p1, p2) /  # using cosine similarity
        print(f"similarity of {p1} and {p2}:\t{similarity}")
