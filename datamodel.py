import csv
import sys
import statistics as stat


class DataModel:
  """Class that parses raw data"""
  def __init__(self):
    self.data_file = 'data/bids.csv'
    self.train_file = 'data/train.csv'
    self.test_file = 'data/test.csv'
    self.reader = csv.DictReader(open('data/bids.csv', 'r'), delimiter=',')
    self.data = {}
    self.train = []
    self.labels = []
    self.test = []


  def get_header_index(self, header_name):
    for i in range(len(self.headers)):
      if self.headers[i] == header_name.lower().strip():
        return i;
    return -1


  def get_header_name(self, header_index):
    if header_index in range(len(self.headers)):
      return self.headers[header_index]
    return None


  def get_data(self, num):
    req_num = num
    data_count = len(self.data)
    if data_count < num or num < 0:
      num = num - self.reader.line_num
      while num != 0:
        try:
          row = next(self.reader)
          if row['bidder_id'] not in self.data:
            self.data[row['bidder_id']] = []
          key = row.pop('bidder_id', None)
          self.data[key].append(row)
          num = num - 1
        except StopIteration as e:
          break

    with open(self.train_file) as train_file:
      reader = csv.DictReader(train_file)
      for row in reader:
        self.train.append(row['bidder_id'])
        self.labels.append(int(float(row['outcome'])))

    with open(self.test_file) as test_file:
      reader = csv.DictReader(test_file)
      for row in reader:
        self.test.append(row['bidder_id'])

