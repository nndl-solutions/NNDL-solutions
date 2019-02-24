import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2_es
net = network2_es.Network([784, 30, 10], cost=network2_es.CrossEntropyCost)
net.SGD(training_data, 10, 0.5,
        lmbda=5.0,
        es=5,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False
)
