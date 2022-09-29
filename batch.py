class Batch:
    def __init__(self, train_batch_size, test_batch_size, sample_gen_function):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_batch = []
        self.test_batch = []
        self.sample_gen_function = sample_gen_function

    def gen_train_batch(self):
        self.train_batch = [self.sample_gen_function() for i in range(self.train_batch_size)]

    def gen_test_batch(self):
        # check if a test batch has already been generated
        if not self.test_batch:
            self.test_batch = [self.sample_gen_function() for i in range(self.test_batch_size)]
        else:
            print("Test Batch already generated")