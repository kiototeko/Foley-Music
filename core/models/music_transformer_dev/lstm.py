# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import pdb


class AudioToKeypointRNN(nn.Module):

    def __init__(self, options):
        super(AudioToKeypointRNN, self).__init__()

        # Instantiating the model
        self.init = None

        hidden_dim = options['hidden_dim']
        if options['trainable_init']:
            device = options['device']
            batch_sz = options['batch_size']
            # Create the trainable initial state
            h_init = \
                init.constant_(torch.empty(1, batch_sz, hidden_dim, device=device), 0.0)
            c_init = \
                init.constant_(torch.empty(1, batch_sz, hidden_dim, device=device), 0.0)
            h_init = Variable(h_init, requires_grad=True)
            c_init = Variable(c_init, requires_grad=True)
            self.init = (h_init, c_init)

        # Declare the model
        self.lstm = nn.LSTM(options['input_dim'], hidden_dim, 1)
        self.dropout = nn.Dropout(options['dropout'])
        self.fc = nn.Linear(hidden_dim, options['output_dim'])

        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.lstm, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        # perform the Forward pass of the model
        pdb.set_trace()
        output, (h_n, c_n) = self.lstm(inputs, self.init)
        #output = output.view(-1, output.size()[-1])  # flatten before FC
        #dped_output = self.dropout(output)
        predictions = self.fc(output)
        return predictions
