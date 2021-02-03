from __future__ import absolute_import

import argparse
import logging

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io.fileio import WriteToFiles
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def get_embed(input):
  module = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
  embed = module(input)
  return embed

def run(argv=None, save_main_session=True):
  parser = argparse.ArgumentParser()
  
  # a file with one text per line on google storage
  parser.add_argument(
      '--input',
      dest='input',
      required=True,
      help='Input file to process.')

  # google storage output
  parser.add_argument(
      '--output',
      dest='output',
      required=True,
      help='Output file to write results to.')

  known_args, pipeline_args = parser.parse_known_args(argv)

  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  
  with beam.Pipeline(options=pipeline_options) as p:
    # read in data
    lines = p | 'Read' >> ReadFromText(known_args.input)
    
    embeds = get_embed(lines)
    output = np.array(embeds)
    
    # write output
    output | 'Write' >> WriteToFiles(known_args.output)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
