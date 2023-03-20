import numpy as np
import math
import pickle
from textwrap import dedent
from trainer import Trainer

class TestResult:
    def __init__(self, network_pred):
        self.network_pred = network_pred.astype(np.int64)

        self.tests_count = network_pred.sum()
        self.correct_count = network_pred.trace()
        self.false_count = self.tests_count - self.correct_count
        self.correct_portion = self.correct_count / self.tests_count
        self.false_portion = self.false_count / self.tests_count

        self.label_count = network_pred.sum(axis=1)
        self.label_correct_count = network_pred.diagonal()
        self.label_false_count = self.label_count - self.label_correct_count
        self.label_correct_portion = self.label_correct_count / self.label_count
        self.label_false_portion = self.label_false_count / self.label_count

    
    def __str__(self):
        return dedent("""\
        Test Result:
        (true Label, network prediction)-table:
        {}
        
        Tests count:        {}
        Correct count:      {}
        False count:        {}
        Correct portion:    {}
        False portion:      {}
        
        Labels count:           {}
        Lables correct count:   {}
        Lables false count:     {}
        Lables correct portion: {}
        Lables false portion:   {}""").format(self.network_pred, 
            self.tests_count, self.correct_count, self.false_count, self.correct_portion, self.false_portion,
            self.label_count, self.label_correct_count, self.label_false_count, np.round(self.label_correct_portion, 2), np.round(self.label_false_portion, 2))
        

class Analysis:
    def __init__(self, trainer, test_images, test_labels, num_labels):
        self.trainer = trainer
        self.test_images = test_images
        self.test_labels = test_labels
        self.num_labels = num_labels
        self.test_results_epoch = []
        self.test_results_batch = []

        initial_test_result = self.perform_test()
        self.test_results_epoch.append(initial_test_result)
        self.test_results_batch.append(initial_test_result)

    @staticmethod
    def load(filepath, train_images, train_labels, test_images, test_labels):
        f = open(filepath, "rb")
        analysis = pickle.load(f)
        f.close()

        analysis.trainer.set_training_data(train_images, train_labels)
        analysis.test_images = test_images
        analysis.test_labels = test_labels

        return analysis

    def perform_analysis(self, epochs, batches_per_test=math.inf, num_tests_batch=math.inf, num_tests_epoch=math.inf):
        batches_per_test = min(batches_per_test, self.trainer.epoch_size)
        num_tests_batch = min(num_tests_batch, len(self.test_images))
        num_tests_epoch = min(num_tests_epoch, len(self.test_images))
        for _ in range(epochs):
            batch_counter = 0
            while True:
                self.trainer.finish_batch()
                batch_counter += 1
                if batch_counter >= batches_per_test:
                    test_result = self.perform_test(num_tests_batch)
                    self.test_results_batch.append(test_result)
                    batch_counter = 0

                if self.trainer.epoch_counter == 0:
                    test_result = self.perform_test(num_tests_epoch)
                    self.test_results_epoch.append(test_result)
                    break

    def perform_test(self, num_tests=math.inf):
        network_pred = np.zeros(shape=(self.num_labels, self.num_labels))

        for j in range(min(num_tests, len(self.test_images))):
            pred_enc = self.trainer.best_network.forward(self.test_images[j])
            pred = np.argmax(pred_enc)
            network_pred[self.test_labels[j]][pred] += 1

        return TestResult(network_pred)


    def save_to_file(self, file):
        if isinstance(file, str):
            with open(file, "wb") as f:
                return self.save_to_file(f)
        
        self.trainer.save_to_file(file)
        pickle.dump(
            (
                self.num_labels,
                self.test_results_epoch,
                self.test_results_batch,
            ),
            file
        )

    @staticmethod
    def load_from_file(file, train_images, train_labels, test_images, test_labels):
        if isinstance(file, str):
            with open(file, "rb") as f:
                return Analysis.load_from_file(f, train_images, train_labels, test_images, test_labels)
        
        analysis = Analysis.__new__(Analysis)

        analysis.trainer = Trainer.load_from_file(file, train_images, train_labels)
        (
            analysis.num_labels,
            analysis.test_results_epoch,
            analysis.test_results_batch,
        ) = pickle.load(file)

        analysis.test_images = test_images
        analysis.test_labels = test_labels

        return analysis