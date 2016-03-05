import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn


class CharRNN(object):
  """Character RNN model."""
  
  def __init__(self, is_training, batch_size, num_unrollings, vocab_size, 
               hidden_size, max_grad_norm, embedding_size, num_layers):
    self.batch_size = batch_size
    self.num_unrollings = num_unrollings
    if not is_training:
      self.batch_size = 1
      self.num_unrollings = 1
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.embedding_size = embedding_size
    self.model_size = (embedding_size * vocab_size + # embedding parameters
                       # lstm parameters
                       4 * hidden_size * (hidden_size + embedding_size + 1) +
                       # softmax parameters
                       vocab_size * (hidden_size + 1) +
                       # multilayer lstm parameters for extra layers.
                       (num_layers - 1) * 4 * hidden_size *
                       (hidden_size + hidden_size + 1))
    
    # Placeholder to feed in input and targets/labels data.
    self.input_data = tf.placeholder(tf.int64,
                                     [self.batch_size, self.num_unrollings],
                                     name='inputs')
    self.targets = tf.placeholder(tf.int64,
                                  [self.batch_size, self.num_unrollings],
                                  name='targets')

    # with tf.variable_scope('rnn_cell') as rnn_cell:
    # Create multilayer LSTM cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                             input_size=self.embedding_size,
                                             forget_bias=0.0)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)

    with tf.name_scope('initial_state'):
      # zero_state is used to compute the intial state for cell.
      self.zero_state = cell.zero_state(self.batch_size, tf.float32)
      
      # Placeholder to feed in initial state.
      self.initial_state = tf.placeholder(tf.float32,
                                          [self.batch_size, cell.state_size],
                                          'initial_state')

    # Embeddings layers.
    with tf.name_scope('embedding_layer'):
      with tf.device("/cpu:0"):
        self.embedding = tf.get_variable("embedding",
                                         [self.vocab_size, self.embedding_size])
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

    with tf.name_scope('slice_inputs'):
      # Slice inputs into a list of shape [batch_size, 1] data colums.
      sliced_inputs = [tf.reshape(input_, [self.batch_size, self.embedding_size])
                       for input_ in tf.split(1, self.num_unrollings, inputs)]
    
    # print(sliced_inputs[0].get_shape())
    # Copy cell to do unrolling and collect outputs.
    outputs, final_state = rnn.rnn(cell, sliced_inputs, initial_state=self.initial_state)
    self.final_state = final_state

    with tf.name_scope('flatten_ouput_and_target'):
      # Reshape outputs into one dimension.
      flat_outputs = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
      # Reshape the targets too.      
      flat_targets = tf.reshape(tf.concat(1, self.targets), [-1])

    
    # Create softmax parameters, weights and bias.
    with tf.variable_scope('softmax') as sm_vs:
      softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])
      logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
      self.probs = tf.nn.softmax(logits)


    with tf.name_scope('loss'):
      # Compute mean cross entropy loss for each output.
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, flat_targets)
      
      self.mean_loss = tf.reduce_sum(loss) / (self.batch_size * self.num_unrollings)


    with tf.name_scope('loss_monitor'):
      # Count the number of elements and the sum of mean_loss
      # from each batch to compute the average loss.
      count = tf.Variable(1.0, name='count')
      sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
      
      self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                         count.assign(0.0),
                                         name='reset_loss_monitor')
      self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss +
                                                               self.mean_loss),
                                          count.assign(count + 1),
                                          name='update_loss_monitor')
      with tf.control_dependencies([self.update_loss_monitor]):
        self.average_loss = sum_mean_loss / count
        self.ppl = tf.exp(self.average_loss)

      # Monitor the loss.
      if is_training:
        loss_summary_name = "average training loss"
        ppl_summary_name = "training perplexity"
      else:
        loss_summary_name = "average evaluation loss"
        ppl_summary_name = "evaluation perplexity"
      average_loss_summary = tf.scalar_summary(loss_summary_name, self.average_loss)
      ppl_summary = tf.scalar_summary(ppl_summary_name, self.ppl)

    # Monitor the loss.
    self.summaries = tf.merge_summary([average_loss_summary, ppl_summary],
                                      name='loss_monitor')
    
    tf.Variable(1.0, name='just_testing_3')

    self.global_step = tf.get_variable('global_step', [],
                                       initializer=tf.constant_initializer(0.0))
    if is_training:
      learning_rate = tf.train.exponential_decay(1.0, self.global_step,
                                                 5000, 0.1, staircase=True)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                        self.max_grad_norm)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
      self.saver = tf.train.Saver()
    
      
  def run_epoch(self, session, data_size, batch_generator, is_training,
                verbose=False, frequency=10, summary_writer=None, debug=False):
    """Runs the model on the given data for one full pass."""
    epoch_size = ((data_size // self.batch_size) - 1) // self.num_unrollings
    
    print('epoch_size: %d' % epoch_size)
    print('data_size: %d' % data_size)
    print('num_unrollings: %d' % self.num_unrollings)
    print('batch_size: %d' % self.batch_size)

    if is_training:
      extra_op = self.train_op
    else:
      extra_op = tf.no_op()

    # Prepare initial state and reset the average loss
    # computation.
    state = self.zero_state.eval()
    self.reset_loss_monitor.run()
    start_time = time.time()
    for step in range(epoch_size):
      # Generate the batch and use [:-1] as inputs and [1:] as targets.
      data = batch_generator.next()
      inputs = np.array(data[:-1]).transpose()
      targets = np.array(data[1:]).transpose()

      ops = [self.average_loss, self.final_state, extra_op,
             self.summaries, self.global_step]
      
      feed_dict = {self.input_data: inputs, self.targets: targets,
                   self.initial_state: state}

      results = session.run(ops, feed_dict)
      average_loss, state, _, summary_str, global_step = results
      
      ppl = np.exp(average_loss)
      if verbose and ((step+1) % frequency == 0):
        print("%.1f%%, step:%d, perplexity: %.3f, speed: %.0f wps" %
              (step * 1.0 / epoch_size * 100, step, ppl,
               step * self.batch_size * self.num_unrollings /
               (time.time() - start_time)))

    print("final ppl: %.3f, speed: %.0f wps" %
          (ppl, step * self.batch_size * self.num_unrollings /
           (time.time() - start_time)))
    return ppl, summary_str, global_step

  def sample_seq(self, session, length, start_text, vocab_index_dict,
                 index_vocab_dict, max_prob=True):

    state = self.zero_state.eval()

    # use start_text to warm up the RNN.
    if start_text is not None:
      seq = list(start_text)
      for char in start_text[:-1]:
        x = np.array([[char2id(char, vocab_index_dict)]])
        state = session.run(self.final_state,
                            {self.input_data: x,
                             self.initial_state: state})
      x = np.array([[char2id(start_text[-1], vocab_index_dict)]])

    for i in range(length):
      state, probs = session.run([self.final_state,
                                  self.probs],
                                 {self.input_data: x,
                                  self.initial_state: state})
      if max_prob:
        sample = np.argmax(probs[0])
      else:
        sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]

      seq.append(id2char(sample, index_vocab_dict))
      x = np.array([[sample]])
    return ''.join(seq)
      
        
class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, text, batch_size, n_unrollings, vocab_size,
                 vocab_index_dict, index_vocab_dict):
      self._text = text
      self._text_size = len(text)
      self._batch_size = batch_size
      self.vocab_size = vocab_size
      self._n_unrollings = n_unrollings
      self.vocab_index_dict = vocab_index_dict
      self.index_vocab_dict = index_vocab_dict
      
      segment = self._text_size // batch_size

      # number of elements in cursor list is the same as
      # batch_size.  each batch is just the collection of
      # elements in where the cursors are pointing to.
      self._cursor = [ offset * segment for offset in range(batch_size)]
      self._last_batch = self._next_batch()
      
    def _next_batch(self):
      """Generate a single batch from the current cursor position in the data."""
      batch = np.zeros(shape=(self._batch_size), dtype=np.float)
      for b in range(self._batch_size):
        batch[b] = char2id(self._text[self._cursor[b]], self.vocab_index_dict)
        self._cursor[b] = (self._cursor[b] + 1) % self._text_size
      return batch

    def next(self):
      """Generate the next array of batches from the data. The array consists of
      the last batch of the previous array, followed by num_unrollings new ones.
      """
      batches = [self._last_batch]
      for step in range(self._n_unrollings):
        batches.append(self._next_batch())
      self._last_batch = batches[-1]
      return batches


# Utility functions
def batches2string(batches, index_vocab_dict):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, id2char_list(b, index_vocab_dict))]
  return s


def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]


def char2id(char, vocab_index_dict):
  try:
    return vocab_index_dict[char]
  except KeyError:
    print('Unexpected char')
    return 0


def id2char(index, index_vocab_dict):
  return index_vocab_dict[index]

    
def id2char_list(lst, index_vocab_dict):
  return [id2char(i, index_vocab_dict) for i in lst]

