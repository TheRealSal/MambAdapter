import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor
import torchaudio, torch


class Urban_Sound_8k(Dataset):
    """
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
        air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot,
        jackhammer, siren, and street_music.
    """

    def __init__(
        self,
        data_path,
        max_len_AST,
        split,
        train_fold_nums=[1,2,3,4,5,6,7,8,9],
        test_fold_nums=[10],
        apply_SpecAug=False,
        few_shot=False,
        samples_per_class=1,
        cache_dir=None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"`train` arg ({split}) must be 'train' or 'test'.")

        self.data_path = os.path.expanduser(data_path)
        self.max_len_AST = max_len_AST
        self.split = split
        self.train_fold_nums = train_fold_nums
        self.test_fold_nums = test_fold_nums

        self.apply_SpecAug = apply_SpecAug
        self.freq_mask = 24
        self.time_mask = 80

        self.processor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            max_length=self.max_len_AST,
            cache_dir=cache_dir,
        )

        self.x, self.y = self.get_data()

        if few_shot:
            self.x, self.y = self.get_few_shot_data(samples_per_class)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x_i = self.x[index]  # torch.Tensor
        y_i = self.y[index]  # torch.LongTensor scalar or int

        if self.apply_SpecAug:
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            timem = torchaudio.transforms.TimeMasking(self.time_mask)

            # x_i should be (time, freq); transpose to (freq, time)
            if x_i.ndim == 2:
                fbank = torch.transpose(x_i, 0, 1)
            elif x_i.ndim == 3:
                fbank = torch.transpose(x_i[0], 0, 1)
            else:
                raise RuntimeError(f"Unexpected feature shape {tuple(x_i.shape)}")

            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)  # back to (time, freq)
            return fbank, int(y_i) if isinstance(y_i, (int, np.integer)) else y_i

        return x_i, int(y_i) if isinstance(y_i, (int, np.integer)) else y_i

    def get_few_shot_data(self, samples_per_class: int):
        x_few, y_few = [], []
        total_classes = torch.unique(self.y).tolist() if isinstance(self.y, torch.Tensor) else sorted(set(self.y))

        counts = {c: 0 for c in total_classes}
        for xi, yi in zip(self.x, self.y):
            c = int(yi) if not isinstance(yi, int) else yi
            if counts[c] < samples_per_class:
                x_few.append(xi)
                y_few.append(c)
                counts[c] += 1
            if all(v >= samples_per_class for v in counts.values()):
                break

        return x_few, torch.tensor(y_few, dtype=torch.long)

    def get_data(self):
        folds = self.train_fold_nums if self.split == "train" else self.test_fold_nums

        x_list, y_list = [], []

        meta_path = os.path.join(self.data_path, "UrbanSound8K", "metadata", "UrbanSound8K.csv")
        with open(meta_path) as f:
            lines = f.readlines()[1:]  # skip header

        for line in lines:
            items = line.rstrip("\n").split(",")
            fold = int(items[-3])
            if fold not in folds:
                continue

            wav_path = os.path.join(self.data_path, "UrbanSound8K", "audio", f"fold{fold}", items[0])
            wav, sr = soundfile.read(wav_path)

            # mono
            if wav.ndim > 1:
                wav = wav[:, 0]

            # resample to 16 kHz
            if sr != 16000:
                wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)

            proc = self.processor(wav, sampling_rate=16000, return_tensors="pt")
            feat = proc["input_values"].squeeze(0)

            if feat.ndim == 1:
                feat = feat.unsqueeze(1)  # (time, 1)

            x_list.append(feat.float())
            y_list.append(self.class_ids[items[-1]])

        y_tensor = torch.tensor(y_list, dtype=torch.long)
        return x_list, y_tensor

    @property
    def class_ids(self):
        return {
            "air_conditioner": 0,
            "car_horn": 1,
            "children_playing": 2,
            "dog_bark": 3,
            "drilling": 4,
            "engine_idling": 5,
            "gun_shot": 6,
            "jackhammer": 7,
            "siren": 8,
            "street_music": 9,
        }