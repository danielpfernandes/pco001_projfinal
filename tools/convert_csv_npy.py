#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ******************************************************
# * Este script conver arquivos .csv em arquivos .npy  *
# *                                                    *
# * Autores: Daniel P Fernandes, Natalia S Sanchez e   *
# *          Alexandre L Sousa                         *
# *                                                    *
# ******************************************************
#
# Copyright 2021 Daniel P Fernandes, Natalia S Sanches & Alexandre L Sousa
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--input", action='store', help="Input file.", type=str)

    return parser.parse_args()

def csv_npy(args):
    data = np.genfromtxt(fname=args.input, delimiter=';')
    print(data)
    print(data.ndim)
    out_path = os.path.splitext(args.input)[0] + ".npy"
    np.save(out_path, data)

def main():
    args = parse_args()
    csv_npy(args)

if __name__ == "__main__":
    main()