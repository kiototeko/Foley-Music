from pyhocon import ConfigTree
from Torchpie.torchpie.logging import logger

import torch
from torch import nn, Tensor


class ModelFactory:

    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg

    def build(self, device=torch.device('cpu'), wrapper=lambda x: x):
        emb_dim = self.cfg.get_int('model.emb_dim')
        hid_dim = self.cfg.get_int('model.hid_dim', 512)
        duration = self.cfg.get_float('dataset.duration')
        fps = self.cfg.get_float('dataset.fps')
        layout = self.cfg.get_string('dataset.pose_layout')
        events_per_sec = self.cfg.get_int('dataset.events_per_sec')
        ckpt = self.cfg.get_string('ckpt')
        streams = self.cfg.get_config('dataset.streams')
        num_heads = self.cfg.get_int('model.num_heads')
        audio_duration = duration

        if self.cfg.get_string('model.name') == 'music_transformer':
            from .music_transformer_dev.music_transformer import music_transformer_dev_baseline
            pose_seq2seq = music_transformer_dev_baseline(
                243,#123,#240 + 3, #88*2 for note_on/off, 32 velocity, 32 time_shift, 3 pad
		num_heads=num_heads,
                d_model=emb_dim,
                dim_feedforward=emb_dim * 2,
                encoder_max_seq=int(duration * fps),
                decoder_max_seq=self.cfg.get_int('model.decoder_max_seq', 480),
                layout=layout,
                num_encoder_layers=self.cfg.get_int('model.num_encoder_layers', 0),
                num_decoder_layers=self.cfg.get_int('model.num_decoder_layers', 3),
                rpr=self.cfg.get_bool('model.rpr', True),
                use_control='control' in self.cfg.get_config('dataset.streams'),
                rnn=self.cfg.get_bool('model.rnn'),
                layers=self.cfg.get_int('model.pose_net_layers'),
                fps=self.cfg.get_int('dataset.fps'),
                img_height=self.cfg.get_int('dataset.sensor_channels'),
                batch_size=self.cfg.get_int('batch_size'),
                duration=duration,
                in_channels=self.cfg.get_int('dataset.num_sensors'),
                convTransformer=self.cfg.get_bool('model.convTransformer'),
                hidden_dim=self.cfg.get_int('model.hidden_dim')
            )
            if ckpt != 'ckpt':
                pass
                # TODO load weight for finetune

        else:
            raise Exception

        pose_seq2seq = pose_seq2seq.to(device)
        pose_seq2seq = wrapper(pose_seq2seq)

        return pose_seq2seq
