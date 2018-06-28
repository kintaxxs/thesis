import torch
import fire
from time import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from config import opt
import torchvision
import tensorly as tl
import torchvision.transforms as transforms
import os
import torch.backends.cudnn as cudnn
import copy
import numpy as np
import time
import json
import onnx
from PIL import Image
#from utils.filter_prunner import FilterPrunner
from utils.prune import *
from heapq import nsmallest
from operator import itemgetter
from utils.utils import progress_bar
from utils.estimator import Estimator
from utils.deploy import export_onnx_model, deploy_by_rpc
from utils.fine_tune import fine_tune_test, fine_tune
from utils import bc

tl.set_backend('pytorch')
dataset = 'cifar'
runtime_budget = 6 #ms

class FilterPrunner:
        def __init__(self, model):
                self.model = model
                self.reset()

        def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
                self.filter_ranks = {}

        def forward(self, x):
                self.activations = []
                self.gradients = []
                self.grad_index = 0
                self.activation_to_layer = {}

                activation_index = 0
                for layer, (name, module) in enumerate(self.model.features._modules.items()):
                    x = module(x)
                    if isinstance(module, torch.nn.modules.conv.Conv2d):
                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                        self.activation_to_layer[activation_index] = layer
                        activation_index += 1

                return self.model.classifier(x.view(x.size(0), -1))

        def compute_rank(self, grad):
                activation_index = len(self.activations) - self.grad_index - 1
                activation = self.activations[activation_index]
                values = \
                        torch.sum((activation * grad), dim = 0, keepdim=True).\
                                sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

                # Normalize the rank by the filter dimensions
                values = \
                        values / (activation.size(0) * activation.size(2) * activation.size(3))

                if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = \
                                torch.FloatTensor(activation.size(1)).zero_().cuda()

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

        def lowest_ranking_filters(self, num):
                data = []
                for i in sorted(self.filter_ranks.keys()):
                        for j in range(self.filter_ranks[i].size(0)):
                                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                return nsmallest(num, data, itemgetter(2))

        def normalize_ranks_per_layer(self):
                for i in self.filter_ranks:
                        v = torch.abs(self.filter_ranks[i])
                        sqrt_v = np.sqrt(torch.sum(v * v))
                        v = v / sqrt_v.type(torch.cuda.FloatTensor)
                        self.filter_ranks[i] = v.cpu()

        def get_prunning_plan(self, num_filters_to_prune):
                filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
                #print('Travis: ', filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
                filters_to_prune_per_layer = {}
                for (l, f, _) in filters_to_prune:
                        if l not in filters_to_prune_per_layer:
                                filters_to_prune_per_layer[l] = []
                        filters_to_prune_per_layer[l].append(f)

                for l in filters_to_prune_per_layer:
                        filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                        for i in range(len(filters_to_prune_per_layer[l])):
                                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

                filters_to_prune = []
                for l in filters_to_prune_per_layer:
                        for i in filters_to_prune_per_layer[l]:
                                filters_to_prune.append((l, i))

                return filters_to_prune

        def get_layer_prunning_rank(self, num_filters_to_prune):
            filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
            return filters_to_prune
        def get_layer_prunning_plan(self, filters_to_prune):

            filters_to_prune_per_layer = {}
            for (l, f, _) in filters_to_prune:
                    if l not in filters_to_prune_per_layer:
                            filters_to_prune_per_layer[l] = []
                    filters_to_prune_per_layer[l].append(f)

            for l in filters_to_prune_per_layer:
                    filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                    for i in range(len(filters_to_prune_per_layer[l])):
                            filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

            filters_to_prune = []
            for l in filters_to_prune_per_layer:
                    for i in filters_to_prune_per_layer[l]:
                            filters_to_prune.append((l, i))

            return filters_to_prune
        def modify_model(self, model):
            self.model = model

class PrunningFineTuner_VGG16:
        def __init__(self, model):

                # Preparing data
                print('==> Preparing data..')
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
                self.train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
                testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
                self.test_data_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

                self.model = model
                self.criterion = torch.nn.CrossEntropyLoss()
                self.prunner = FilterPrunner(self.model)
                self.model.train()
                self.perf_model = Estimator()
                self.initialization = 0.5
                self.decay_rate = 0.96

        def test(self):
                self.model.eval()
                correct = 0
                total = 0

                for i, (batch, label) in enumerate(self.test_data_loader):
                        batch = batch.cuda()
                        #print(Variable(batch).size())
                        output = self.model(Variable(batch))
                        pred = output.data.max(1)[1]
                        correct += pred.cpu().eq(label).sum()
                        total += label.size(0)
                print("Accuracy :", float(correct) / total)
                self.model.train()

        def train(self, optimizer = None, epoches = 10):
                if optimizer is None:
                        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

                for i in range(epoches):
                        print("Epoch: ", i)
                        self.train_epoch(optimizer)
                        self.test()
                print("Finished fine tuning.")


        def train_batch(self, optimizer, batch, label, rank_filters):
                self.model.zero_grad()
                input = Variable(batch)

                if rank_filters:
                        output = self.prunner.forward(input)
                        self.criterion(output, Variable(label)).backward()
                else:
                        self.criterion(self.model(input), Variable(label)).backward()
                        optimizer.step()
        def train_epoch(self, optimizer = None, rank_filters = False):
                for batch, label in self.train_data_loader:
                        self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

        def get_candidates_to_prune(self, num_filters_to_prune):
                self.prunner.reset()

                self.train_epoch(rank_filters = True)

                self.prunner.normalize_ranks_per_layer()

                return self.prunner.get_prunning_plan(num_filters_to_prune)

        def get_rank_to_prune(self, num_filters_to_prune):
                self.prunner.reset()

                self.train_epoch(rank_filters = True)

                self.prunner.normalize_ranks_per_layer()

                return self.prunner.get_layer_prunning_rank(num_filters_to_prune)

        def total_num_filters(self):
                filters = 0
                for name, module in self.model.features._modules.items():
                    if isinstance(module, torch.nn.modules.conv.Conv2d):
                        filters = filters + module.out_channels
                return filters

        def prune(self):

                print('\n{}NetAdapt Begin{}'.format(bc.yellow, bc.end))
                print('{}↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓{}'.format(bc.yellow, bc.end))
                print("\n{}Runtime_budget is {}{}".format(bc.green, runtime_budget, bc.end))

                #Get the accuracy before prunning
                self.test()
                self.model.train()

                #Make sure all the layers are trainable
                for param in self.model.features.parameters():
                        param.requires_grad = True

                model_image_size = {}
                if(dataset == 'cifar'):
                    in_image_size = 32
                    for i, key in enumerate(self.model.features._modules.keys()):
                        if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                            conv_layer = self.model.features._modules[key]
                            after_image_size = ((in_image_size - conv_layer.kernel_size[0] + 2*conv_layer.padding[0]) // conv_layer.stride[0] )+ 1
                            model_image_size[key] = [in_image_size, after_image_size]
                            in_image_size = after_image_size
                        elif isinstance(self.model.features._modules[key], torch.nn.modules.MaxPool2d):
                            maxpool_layer = self.model.features._modules[key]
                            after_image_size = ((in_image_size - maxpool_layer.kernel_size) // maxpool_layer.stride )+ 1
                            model_image_size[key] = [in_image_size, after_image_size]
                            in_image_size = after_image_size

                print(model_image_size)

                #Get the index of each layer
                self.layer_index = []
                for i, key in enumerate(self.model.features._modules.keys()):
                    if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                        self.layer_index.append(key)

                #Get the runtime before prunning
                self.origin_predict_layer_runtime = {}
                origin_predict_runtime = 0.0
                for i, key in enumerate(self.model.features._modules.keys()):
                    if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                        conv_layer = self.model.features._modules[key]
                        tmp_perf_runtime = self.perf_model.conv_predict(model_image_size[key][0], conv_layer.in_channels, conv_layer.out_channels ,\
                                conv_layer.kernel_size[0], conv_layer.stride[0], conv_layer.padding[0])
                        self.origin_predict_layer_runtime['conv_'+key] = tmp_perf_runtime
                        origin_predict_runtime += tmp_perf_runtime
                for i, key in enumerate(self.model.classifier._modules.keys()):
                    if isinstance(self.model.classifier._modules[key], torch.nn.modules.linear.Linear):
                        fc_layer = self.model.classifier._modules[key]
                        tmp_perf_runtime = self.perf_model.fc_predict(fc_layer.in_features, fc_layer.out_features)
                        self.origin_predict_layer_runtime['fc_'+key] = tmp_perf_runtime
                        origin_predict_runtime += tmp_perf_runtime
                print('{}Predict_Origin_predict_runtime{}: {}ms'.format(bc.green, bc.end, origin_predict_runtime))
                print('{}Predict_Origin_predict_layer_runtime{}: {}\n'.format(bc.green, bc.end, self.origin_predict_layer_runtime))

                self.decomposed_predict_layer_runtime = copy.deepcopy(self.origin_predict_layer_runtime)

                origin_model = copy.deepcopy(self.model)
                #model_runtime = origin_runtime * 1000
                perf_model_runtime = origin_predict_runtime
                iteration_count = 1


                while(perf_model_runtime > runtime_budget):

                    print('{}Iteration {}{}'.format(bc.red, iteration_count, bc.end))
                    print('{}--------------------------------------------{}'.format(bc.red, bc.end))

                    if(iteration_count > 100):
                        import sys
                        print('iteration > 100')
                        sys.exit(1)

                    number_of_filters = self.total_num_filters()
                    print("Ranking filters.. ")
                    prune_targets = self.get_rank_to_prune(number_of_filters)
                    print('{}Number_of_filters{}: {}'.format(bc.green, bc.end, number_of_filters))
                    print('{}Initialization{}: {}, {}Decay{}: {}'.format(bc.green, bc.end, self.initialization, bc.green, bc.end, self.decay_rate))
                    #print('Travis model: ', self.model)
                    #print('Travis prune_targets: ', prune_targets)
                    print('')

                    layer_record = {}
                    model_record = {}

                    #tmp_model = copy.deepcopy(self.model)
                    for i, key in enumerate(self.model.features._modules.keys()):
                        if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                            prune_conv_layer = self.model.features._modules[key]

                            if(self.layer_index.index(key) != len(self.layer_index)-1):
                                next_conv_layer = self.model.features._modules[self.layer_index[self.layer_index.index(key)+1]]
                            else:
                                next_conv_layer = None
                            print('{}Convolution Layer {}{}'.format(bc.light_blue, key, bc.end))
                            print('Curr Layer: {}, Next Layer: {}'.format(prune_conv_layer, next_conv_layer))
                            tmp_model = copy.deepcopy(self.model)
                            #print('Travis tmp_model: {}'.format(tmp_model))
                            layer_all_filter = []
                            for i in prune_targets:
                                if(str(i[0]) == key):
                                    layer_all_filter.append(i)

                            tmp_perf_runtime = self.perf_model.conv_predict(model_image_size[key][0], prune_conv_layer.in_channels, prune_conv_layer.out_channels ,\
                                    prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0], prune_conv_layer.padding[0])
                            if(tmp_perf_runtime <= self.initialization):
                                print('{}tmp_perf_runtime <= self.initialization{}'.format(bc.purple, bc.end))
                                print('tmp_perf_runtime: {}, self.initialzation: {}'.format(tmp_perf_runtime, self.initialization))
                                print('image_size: {}, in_channels: {}, out_channels: {}, kernel_size: {}, stride: {}, padding: {}'.format(model_image_size[key][0],\
                                        prune_conv_layer.in_channels, prune_conv_layer.out_channels, prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0],\
                                        prune_conv_layer.padding[0]))
                                layer_record[key] = [0, None, None]
                                print('')
                                continue

                            if(next_conv_layer != None):
                                prune_layer_runtime_1 = self.perf_model.conv_predict(model_image_size[key][0], prune_conv_layer.in_channels, prune_conv_layer.out_channels ,\
                                        prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0], prune_conv_layer.padding[0])
                                prune_layer_runtime_2 = self.perf_model.conv_predict(model_image_size[self.layer_index[self.layer_index.index(key)+1]][0], next_conv_layer.in_channels, next_conv_layer.out_channels ,\
                                        next_conv_layer.kernel_size[0], next_conv_layer.stride[0], next_conv_layer.padding[0])
                                prune_layer_runtime = prune_layer_runtime_1 + prune_layer_runtime_2
                            else:
                                prune_layer_runtime = self.perf_model.conv_predict(model_image_size[key][0], prune_conv_layer.in_channels, prune_conv_layer.out_channels ,\
                                        prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0], prune_conv_layer.padding[0])

                            for tmp_out_channel in range(prune_conv_layer.out_channels-1, 1, -1):
                                if(next_conv_layer != None):
                                    tmp_runtime_1 = self.perf_model.conv_predict(model_image_size[key][0], prune_conv_layer.in_channels, tmp_out_channel ,\
                                        prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0], prune_conv_layer.padding[0])
                                    tmp_runtime_2 = self.perf_model.conv_predict(model_image_size[self.layer_index[self.layer_index.index(key)+1]][0], tmp_out_channel, next_conv_layer.out_channels ,\
                                        next_conv_layer.kernel_size[0], next_conv_layer.stride[0], next_conv_layer.padding[0])
                                    tmp_runtime = tmp_runtime_1 + tmp_runtime_2
                                else:
                                    tmp_runtime_1 = self.perf_model.conv_predict(model_image_size[key][0], prune_conv_layer.in_channels, tmp_out_channel ,\
                                        prune_conv_layer.kernel_size[0], prune_conv_layer.stride[0], prune_conv_layer.padding[0])
                                    tmp_runtime = tmp_runtime_1

                                #print('Travis tmp_out_channel: {}, tmp_runtime: {}, prune_layer_runtime: {}'.format(tmp_out_channel, tmp_runtime, prune_layer_runtime))
                                if((prune_layer_runtime - tmp_runtime) >= self.initialization):
                                    print('Travis prune_layer_runtime: {}, tmp_runtime: {}, prune_layer_runtime-tmp_runtime: {}'.format(prune_layer_runtime, \
                                                                                                tmp_runtime, prune_layer_runtime-tmp_runtime))
                                    num_filter_to_prune = prune_conv_layer.out_channels - tmp_out_channel
                                    break

                            layer_prune_target = layer_all_filter[0:num_filter_to_prune]
                            #for i in layer_prune_target:
                            #    print(i)
                            prune_plan = self.prunner.get_layer_prunning_plan(layer_prune_target)
                            #print(layer_prune_target)
                            #print(prune_plan)

                            layers_prunned = {}
                            for layer_index, filter_index in prune_plan:
                                    if layer_index not in layers_prunned:
                                            layers_prunned[layer_index] = 0
                                    layers_prunned[layer_index] = layers_prunned[layer_index] + 1

                            print("Layers that will be prunned", layers_prunned)

                            print("Prunning filters.. ")
                            cpu_model = tmp_model.cpu()
                            for layer_index, filter_index in prune_plan:
                                    model = prune_vgg16_conv_layer(cpu_model, layer_index, filter_index)

                            tmp_model = cpu_model.cuda()
                            tmp_model, tmp_acc = fine_tune(tmp_model, True)

                            layer_record[key] = [tmp_acc, prune_plan, tmp_model]
                            print('Acc after Fine_Tune {}'.format(tmp_acc))
                            print('')
                            #print('Travis Model', self.model)

                    acc_max = [0, -1]
                    for i, key in enumerate(layer_record.keys()):
                        if layer_record[key][0] > acc_max[1]:
                            acc_max = [key, layer_record[key][0]]
                    print('{}Pick max acc..{} key: {}, acc: {}'.format(bc.blue, bc.end, acc_max[0], acc_max[1]))

                    ## Travis Test
                    self.model = layer_record[acc_max[0]][2]
                    self.prunner.modify_model(layer_record[acc_max[0]][2])

                    print(self.model)
                    self.test()

                    tmp_latency = 0.0
                    tmp_layer_latency = {}
                    for i, key in enumerate(self.model.features._modules.keys()):
                        if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                            tmp_conv_layer = self.model.features._modules[key]
                            tmp_perf_runtime = self.perf_model.conv_predict(model_image_size[key][0], tmp_conv_layer.in_channels, tmp_conv_layer.out_channels ,\
                                    tmp_conv_layer.kernel_size[0], tmp_conv_layer.stride[0], tmp_conv_layer.padding[0])
                            tmp_layer_latency['conv_'+key] = tmp_perf_runtime
                            tmp_latency += tmp_perf_runtime
                    for i, key in enumerate(self.model.classifier._modules.keys()):
                        if isinstance(self.model.classifier._modules[key], torch.nn.modules.linear.Linear):
                            fc_layer = self.model.classifier._modules[key]
                            tmp_perf_runtime = self.perf_model.fc_predict(fc_layer.in_features, fc_layer.out_features)
                            tmp_layer_latency['fc_'+key] = tmp_perf_runtime
                            tmp_latency += tmp_perf_runtime
                    print('{}Predict Runtime after iteration {}: {}ms, reduction: {}ms{}'.format('\033[32m', iteration_count, tmp_latency, perf_model_runtime-tmp_latency, '\033[0m'))
                    print('Runtime for each layer: {}'.format(tmp_layer_latency))

                    perf_model_runtime = tmp_latency

                    #torch.save(self.model, "result/prunned_model/model_" + str(runtime_budget) + '_' + str(iteration_count) +  "_prunned")

                    ## Get runtime after one iteration
                    tmp_save_model_name = export_onnx_model(self.model)
                    tmp_model_runtime = deploy_by_rpc(tmp_save_model_name) * 1000
                    print('{}Real Runtime after iteration {}: {}ms{}'.format('\033[32m', iteration_count, tmp_model_runtime, '\033[0m'))
                    os.remove(tmp_save_model_name)

                    iteration_count += 1
                    self.initialization *= self.decay_rate
                    print('')

                print("{}Finished. Going to fine tune the model a bit more{}".format('\033[33m', '\033[0m'))
                optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
                self.train(optimizer, epoches = 15)
                print('Travis self.model', self.model)
                final_acc = fine_tune_test(self.model, self.test_data_loader, True)

                ## Get runtime after one iteration
                tmp_save_model_name = export_onnx_model(self.model)
                tmp_model_runtime = deploy_by_rpc(tmp_save_model_name) * 1000
                print('Runtime after pruning: {}ms'.format(tmp_model_runtime))
                os.remove(tmp_save_model_name)
                model_runtime_after_pruning = tmp_model_runtime
                
                state = {
                    'net': self.model,
                    'acc': final_acc,
                    'iteration_count': iteration_count,
                    'model_runtime': model_runtime_after_pruning,
                }
                torch.save(state, "result/prunned_model/model_" + str(runtime_budget)  +  "_prunned")

def pruning(**kwargs):

    global runtime_budget

    opt.parse(kwargs)
    #tl.set_backend('pytorch')

    # Model, Load checkpoint
    print('==> Load checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if opt.load_model_path:
        checkpoint = torch.load(opt.load_model_path)
    else:
        import sys 
        print('set the load_model_path')
        sys.exit(1)
    model = checkpoint['net']
    model = model.cuda()

    print(model)
    print('Before Prunning Acc: {}'.format(checkpoint['acc']))

    '''
    for i in range(10, 7, -1):
        runtime_budget = i
        fine_tuner = PrunningFineTuner_VGG16(model)
        fine_tuner.prune()
    '''
    fine_tuner = PrunningFineTuner_VGG16(model)
    fine_tuner.prune()


def get_model_mac_weight(model):

    total_mac = 0
    total_weight = 0

    model_image_size = {}
    if(dataset == 'cifar'):
        in_image_size = 32
        for i, key in enumerate(model.features._modules.keys()):
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                after_image_size = ((in_image_size - conv_layer.kernel_size[0] + 2*conv_layer.padding[0]) // conv_layer.stride[0] )+ 1
                model_image_size[key] = [in_image_size, after_image_size]
                in_image_size = after_image_size
            elif isinstance(model.features._modules[key], torch.nn.modules.MaxPool2d):
                maxpool_layer = model.features._modules[key]
                after_image_size = ((in_image_size - maxpool_layer.kernel_size) // maxpool_layer.stride )+ 1
                model_image_size[key] = [in_image_size, after_image_size]
                in_image_size = after_image_size

    for i, key in enumerate(model.features._modules.keys()):
        if isinstance(model.features._modules[key], torch.nn.modules.container.Sequential):
            sequential = model.features._modules[key]
            for conv_layer in sequential:
                assert isinstance(conv_layer, torch.nn.modules.conv.Conv2d)
                total_mac += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*(model_image_size[key][1]**2)*conv_layer.out_channels
                total_weight += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*conv_layer.out_channels
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = model.features._modules[key]
            total_mac += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*(model_image_size[key][1]**2)*conv_layer.out_channels
            total_weight += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*conv_layer.out_channels

    for i, key in enumerate(model.classifier._modules.keys()):
        if isinstance(model.classifier._modules[key], torch.nn.modules.container.Sequential):
            sequential = model.classifier._modules[key]
            for fc_layer in sequential:
                assert isinstance(fc_layer, torch.nn.modules.linear.Linear)
                total_mac += fc_layer.in_features * fc_layer.out_features
                total_weight += fc_layer.in_features * fc_layer.out_features
        if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
            fc_layer = model.classifier._modules[key]
            total_mac += fc_layer.in_features * fc_layer.out_features
            total_weight += fc_layer.in_features * fc_layer.out_features

    return total_mac, total_weight

def test():

    layer_index = ['0', '3', '6', '8', '10']
    #perf_model = Estimator()
    #print(perf_model.conv_predict(4, 192, 384, 3, 1, 1))
    #print(perf_model.fc_predict(4096, 4096))
    model = torch.load('checkpoint/alexnet_model')['net']
    print(model)
    for i, key in enumerate(model.features._modules.keys()):
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            print(model.features._modules[key], layer_index.index(key), len(layer_index))
            if(layer_index.index(key) == len(layer_index)-1):
                print('out of range')
                break
            print(model.features._modules[layer_index[layer_index.index(key)+1]])
            #print(model.features._modules[list(model.features._modules.keys())[i+1]])
            print('')

def help():

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example
        python {0} train --env='env1025' --lr=0.01
        python {0} test --dataset='path/to/detaset/root/'
        python {0} help
    available args:'''.format(__file__))

    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)

if __name__ == '__main__':
    fire.Fire()
