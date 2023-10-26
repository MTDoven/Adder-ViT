import torchvision, torch
import torchvision.transforms as transforms
from .autoaugment import ImageNetPolicy
from .randomaug import RandAugment
from torch.utils.data import DataLoader

mnist_path = r'D:\Project\_datasets\MNIST'
cifar10_path = r'D:\Project\_datasets\CIFAR10'
cifar100_path = r'D:\Project\_datasets\CIFAR100'
imagenet_path = None #'/opt/data/private/dataset/ImageNet'


def get_mnist(imgsize, batchsize, num_workers):
    # Prepare dataset
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),])
    # Prepare dataset
    trainset = torchvision.datasets.MNIST(root=mnist_path, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    testset = torchvision.datasets.MNIST(root=mnist_path, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    return trainloader, testloader, trainset, testset, None


def get_cifar100(imgsize, batchsize, num_workers):
    # Prepare dataset
    transform_train = transforms.Compose([
            RandAugment(2,14),
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    # Prepare dataset
    trainset = torchvision.datasets.CIFAR100(root=cifar100_path, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    testset = torchvision.datasets.CIFAR100(root=cifar100_path, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, trainset, testset, classes


def get_cifar10(imgsize, batchsize, num_workers):
    # Prepare dataset
    transform_train = transforms.Compose([
            RandAugment(2,14),
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, trainset, testset, classes


def get_imagenet(imgsize, batchsize, num_workers, enhance=None):
    transform_train = transforms.Compose([
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imgsize,imgsize)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    # Prepare dataset
    trainset = torchvision.datasets.ImageFolder(root=imagenet_path+'/train',transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    testset = torchvision.datasets.ImageFolder(root=imagenet_path+'/val',transform=transform_train)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    return trainloader, testloader, trainset, testset, None
