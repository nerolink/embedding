import model_with_word_to_vec as md

projects = {
    'camel': ['camel-1.2', 'camel-1.4', 'camel-1.6'],
    'forrest': ['forrest-0.6', 'forrest-0.7', 'forrest-0.8'],
    'jedit': ['jedit-3.2.1', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
    'ivy': ['ivy-1.1', 'ivy-1.4', 'ivy-2.0'],
    'log4j': ['log4j-1.0', 'log4j-1.1', 'log4j-1.2'],
    'lucene': ['lucene-2.0', 'lucene-2.2', 'lucene-2.4'],
    'poi': ['poi-1.5', 'poi-2.0', 'poi-2.5.1', 'poi-3.0'],
    'prop': ['prop-1', 'prop-2', 'prop-4', 'prop-5', 'prop-6'],
    'synapse': ['synapse-1.0', 'synapse-1.1', 'synapse-1.2'],
    'velocity': ['velocity-1.4', 'velocity-1.5', 'velocity-1.6.1'],
    'xalan': ['xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
    'xerces': ['xerces-1.2', 'xerces-1.3', 'xerces-1.4.4']
}

if __name__ == '__main__':
    # md.train_and_test_cnn_p('forrest', projects['forrest'][0], projects['forrest'][1])
    # md.train_and_test_cnn('forrest', projects['forrest'][0], projects['forrest'][1])
    # md.train_and_test_cnn_p('jedit', projects['jedit'][0], projects['jedit'][1])
    # md.train_and_test_cnn('jedit', projects['jedit'][0], projects['jedit'][1])
    md.train_and_test_cnn_p('lucene', projects['lucene'][0], projects['lucene'][1])
