import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2_es_avg
net = network2_es_avg.Network([784, 30, 10],
                              cost=network2_es_avg.CrossEntropyCost)
net.SGD(training_data, 10, 0.5,
        lmbda=5.0,
        es=10,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False
)
