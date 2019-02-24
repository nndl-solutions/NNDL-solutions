import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2_momentum
net = network2_momentum.Network([784, 30, 10], cost=network2_momentum.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5,
        lmbda=5.0,
        mu=0.01,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False
)
