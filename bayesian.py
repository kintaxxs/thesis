import fire
from config import opt
import torch
import logging
import tensorly as tl
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils.utils import progress_bar
from utils.decompositions import tucker_decomposition_conv_layer_without_rank, tucker_decomposition_fc_layer_without_rank, tucker_decomposition_conv_layer, tucker_decomposition_fc_layer, estimate_ranks, estimate_svd_ranks
from bayes_opt import BayesianOptimization
from utils.estimator import Estimator
import copy
import math
import time
import onnx
from PIL import Image
import numpy as np
import os
from hyperopt import fmin, tpe, hp, rand, space_eval
from utils.deploy import export_onnx_model, deploy_by_rpc
from utils.fine_tune import fine_tune_test, fine_tune

class Decomposer():
    def __init__(self, **kwargs):
        opt.parse(kwargs)
        tl.set_backend('pytorch')
        self.dataset = "cifar"
        self.decomposed_layer_info = {'key': -1, 'image_size': -1, 'kernel_size': -1, 'stride': -1, 'padding': -1}
        self.layer_budget = {}
        self.origin_layer_runtime = {}
        self.origin_model_runtime = 0.0
        self.VBMF_layer_runtime = {}
        self.VBMF_model_runtime = 0.0
        self.VBMF_layer_rank = {}
        self.constrain = opt.constrain
        self.conv_target_rate = 0.0
        self.fc_target_rate = 0.0
        self.user_budget = 1
        self.real_model_runtime = 0.0
        self.remain_budget = 0.0

        # Configure Logger
        self.logger = logging.getLogger()
        log_file = logging.FileHandler('result/log/test.log')
        self.logger.addHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        log_file.setFormatter(formatter)
        self.logger.setLevel(logging.DEBUG)

        # Load Perf Model
        self.perf_model = Estimator()

        # Load Pre-trained Model
        if(opt.load_model_path is None):
            import sys
            print('set the model path')
            sys.exit(-1)
        else:
            checkpoint = torch.load(opt.load_model_path)
            if(type(checkpoint) is dict):
                checkpoint = checkpoint['net']
        self.model = checkpoint.cuda()
        print(self.model)

        # Preparing data
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
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

        print('\n{}Model Info'.format('\033[33m'))
        print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓{}\n'.format('\033[0m'))

        # Set Criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # Calculate Image_size for each layer
        self.model_image_size = {}
        if(self.dataset == 'cifar'):
            in_image_size = 32
        for i, key in enumerate(self.model.features._modules.keys()):
            if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = self.model.features._modules[key]
                after_image_size = ((in_image_size - conv_layer.kernel_size[0] + 2*conv_layer.padding[0]) // conv_layer.stride[0] )+ 1
                self.model_image_size[key] = [in_image_size, after_image_size]
                in_image_size = after_image_size
            elif isinstance(self.model.features._modules[key], torch.nn.modules.MaxPool2d):
                maxpool_layer = self.model.features._modules[key]
                after_image_size = ((in_image_size - maxpool_layer.kernel_size) // maxpool_layer.stride )+ 1
                self.model_image_size[key] = [in_image_size, after_image_size]
                in_image_size = after_image_size
        print('{}Image_Size{}: {}'.format('\033[36m', '\033[0m', self.model_image_size))

        # Get Origin MAC and Weight and runtime
        self.origin_mac, self.origin_weight = self.get_model_mac_weight(self.model)

        self.origin_model_runtime, self.origin_layer_runtime = self.get_model_predict_runtime(self.model)
        #print('self.origin_model_runtime: {}, self.get_model_predict_runtime: {}'.format(self.origin_model_runtime, self.get_model_predict_runtime(self.model)))

	#deploy to target
        save_model_name = export_onnx_model(self.model)
        decomp_runtime = deploy_by_rpc(save_model_name)
        self.real_model_runtime = decomp_runtime * 1000
        os.remove(save_model_name)

        #print('Origin_MAC: {}, Origin_Weight: {}, Pred_Origin_Runtime: {}, Real_Origin_Runtime: {}'.format(self.origin_mac, \
        #                                        self.origin_weight, self.origin_model_runtime, self.real_model_runtime))
        print('{}Origin_MAC{}: {}'.format('\033[36m', '\033[0m', self.origin_mac))
        print('{}Origin_Weight{}: {}'.format('\033[36m', '\033[0m', self.origin_weight))
        print('{}Pred_Origin_Runtime{}: {}'.format('\033[36m', '\033[0m', self.origin_model_runtime))
        print('{}Real_Origin_Runtime{}: {}'.format('\033[36m', '\033[0m', self.real_model_runtime))
        print('{}Origin_Layer_Runtime{}: {}'.format('\033[36m', '\033[0m', self.origin_layer_runtime))
        #print('Origin Layer Runtime: {}'.format(self.origin_layer_runtime))

        # Calculate MAC for each layer
        self.layer_mac = {}
        self.layer_weight = {}
        N_classifier = len(self.model.classifier._modules.keys())
        for i, key in enumerate(self.model.features._modules.keys()):
            if i == 0:
                continue
            if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = self.model.features._modules[key]
                self.layer_mac['conv_'+key] = (conv_layer.kernel_size[0] * conv_layer.kernel_size[0] * conv_layer.in_channels) * \
					(self.model_image_size[key][1] * self.model_image_size[key][1] * conv_layer.out_channels)
                self.layer_weight[key] = (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*conv_layer.out_channels
        for i, key in enumerate(self.model.classifier._modules.keys()):
            if i >= N_classifier - 2:
                break
            if isinstance(self.model.classifier._modules[key], torch.nn.modules.linear.Linear):
                fc_layer = self.model.classifier._modules[key]
                self.layer_mac['fc_'+key] = fc_layer.in_features * fc_layer.out_features
                self.layer_weight[key] = fc_layer.in_features * fc_layer.out_features

        print('{}Layer MAC{}: {}'.format('\033[36m', '\033[0m', self.layer_mac))

        # Calculate importance for each layer

        self.layer_importance = self.get_layer_importance()
        '''
        tmp_total_importance = sum([tmp_mac for tmp_mac in self.layer_mac.values()])
        for key, value in self.layer_mac.items():
            self.layer_importance[key] = float(value/tmp_total_importance)
        '''
        print('{}Layer Importance{}: {}'.format('\033[36m', '\033[0m', self.layer_importance))

        # Get Layer Budget

        self.layer_budget, self.VBMF_layer_runtime, self.VBMF_model_runtime, self.VBMF_layer_rank = self.get_layer_budget()
        print('{}Layer Budget{}: {}'.format('\033[36m', '\033[0m', self.layer_budget))
        #print('Travis VBMF Layer Runtime: {}'.format(self.VBMF_layer_runtime))
        #print('Travis VBMF Model Runtime: {}'.format(self.VBMF_model_runtime))


    def decompose(self):
        
        print('\n{}Bayesian Begin'.format('\033[33m'))
        print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓{}\n'.format('\033[0m'))

        self.conv_decomposition()
        self.fc_decomposition()

        print('-------------> Decomposition Finish')
        print('Final Fine_tune ...')
        fine_tune(self.model, 30)
        fine_tune_test(self.model, self.testloader, True)
        print('The Decomposed Model ...')
        print(self.model)
        mac, weight = self.get_model_mac_weight(self.model)
	#deploy to target
        save_model_name = export_onnx_model(self.model)
        decomp_runtime = deploy_by_rpc(save_model_name)
        self.decomp_runtime_ms = decomp_runtime * 1000
        os.remove(save_model_name)

        tmp_decomp_predict_runtime, tmp_decomp_layer_runtime = self.get_model_predict_runtime(self.model)

        print('Origin_MAC: {}, Origin_Weight: {}, Origin_Runtime: {}, Origin_Predict_Runtime: {}'.format(self.origin_mac, self.origin_weight, self.real_model_runtime, self.origin_model_runtime))
        print('Decomp_MAC: {}, Decomp_Weight: {}, Decomp_Runtime: {}, Decomp_Predict_Runtime: {}'.format(mac, weight, self.decomp_runtime_ms, tmp_decomp_predict_runtime))
        print('Speedup : {}'.format(float(self.real_model_runtime/self.decomp_runtime_ms)))

        torch.save(self.model, 'result/decomposed/' + str(self.constrain) + '_alexnet_model')

    def conv_decomposition(self):

        #remain_budget = 0.0

        for i, key in enumerate(self.model.features._modules.keys()):
            if(i == 0):
                continue
            if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer_to_decompose = self.model.features._modules[key]
                #print('\nTravis conv_layer_to_decompose rank1:{}, rank2:{}'.format(conv_layer_to_decompose.in_channels, conv_layer_to_decompose.out_channels))
                print('\n{}Layer Info{}'.format('\033[31m', '\033[0m'))
                print('{}-----------------------------------------{}'.format('\033[94m', '\033[0m'))
                print('Rank1:{}, Rank2:{}'.format(conv_layer_to_decompose.in_channels, conv_layer_to_decompose.out_channels))
                #print(self.estimate(conv_layer_to_decompose, key))
                self.decomposed_layer_info['key'] = key
                self.decomposed_layer_info['image_size'] = self.model_image_size[key][0]
                self.decomposed_layer_info['kernel_size'] = conv_layer_to_decompose.kernel_size[0]
                self.decomposed_layer_info['stride'] = conv_layer_to_decompose.stride[0]
                self.decomposed_layer_info['padding'] = conv_layer_to_decompose.padding[0]

                self.conv_target_rate = 0.0
                iteration_count = 0
                while(True):
                    print('{}Iteration{}'.format('\033[31m', '\033[0m'))
                    print('{}-----------------------------------------{}'.format('\033[94m', '\033[0m'))
                    print('Iteration {}{}{} Target_Rate: {}{}{}'.format('\033[32m', iteration_count, '\033[0m', '\033[32m', self.conv_target_rate, '\033[0m'))
                    iteration_count += 1
                    ranks = self.conv_bayesian([conv_layer_to_decompose.in_channels, conv_layer_to_decompose.out_channels])
                    tmp_ms_1 = self.estimate_with_config([self.decomposed_layer_info['image_size'], \
                                                        conv_layer_to_decompose.in_channels, ranks[0], \
                                                        1, \
                                                        self.decomposed_layer_info['stride'], \
                                                        self.decomposed_layer_info['padding']])
                    tmp_ms_2 = self.estimate_with_config([self.decomposed_layer_info['image_size'], \
                                                        ranks[0], ranks[1], \
                                                        self.decomposed_layer_info['kernel_size'], \
                                                        self.decomposed_layer_info['stride'], \
                                                        self.decomposed_layer_info['padding']])
                    tmp_ms_3 = self.estimate_with_config([self.decomposed_layer_info['image_size'], \
                                                        ranks[1], conv_layer_to_decompose.out_channels, \
                                                        1, \
                                                        self.decomposed_layer_info['stride'], \
                                                        self.decomposed_layer_info['padding']])
                    tmp_runtime_ms = tmp_ms_1 + tmp_ms_2 + tmp_ms_3
                    print('Travis tmp_ms_1: {}, tmp_ms_2: {}, tmp_ms_3: {}'.format(tmp_ms_1, tmp_ms_2, tmp_ms_3))
                    '''
                    print('Travis tmp_ms_1: {}, image_size: {}, in_channel: {}, out_channel: {}, kernel_size: {}, stride: {}, padding: {}'.format( \
                                            tmp_ms_1, self.decomposed_layer_info['image_size'], conv_layer_to_decompose.in_channels, ranks[0], \
                                            1, self.decomposed_layer_info['stride'], self.decomposed_layer_info['padding']))
                    print('Travis tmp_ms_2: {}, image_size: {}, in_channel: {}, out_channel: {}, kernel_size: {}, stride: {}, padding: {}'.format( \
                                            tmp_ms_2, self.decomposed_layer_info['image_size'], ranks[0], ranks[1], \
                                            self.decomposed_layer_info['kernel_size'], self.decomposed_layer_info['stride'], self.decomposed_layer_info['padding']))
                    print('Travis tmp_ms_3: {}, image_size: {}, in_channel: {}, out_channel: {}, kernel_size: {}, stride: {}, padding: {}'.format( \
                                            tmp_ms_3, self.decomposed_layer_info['image_size'], ranks[1], conv_layer_to_decompose.out_channels, \
                                            1, self.decomposed_layer_info['stride'], self.decomposed_layer_info['padding']))
                    '''
                    assert (tmp_runtime_ms - self.VBMF_layer_runtime['conv_'+key]) > 0
                    if(self.constrain == 0 or (tmp_runtime_ms - self.VBMF_layer_runtime['conv_'+key]) <= (self.layer_budget['conv_'+ key] + self.remain_budget)):
                        print('Travis Constrain: {}, Search_runtime: {}, Layer_budget: {}, Search_layer_runtime: {}, VBMF_layer_runtime: {}, Remain_budget: {}'.format(self.constrain, \
                                                                tmp_runtime_ms-self.VBMF_layer_runtime['conv_'+key], \
                                                                self.layer_budget['conv_'+key], tmp_runtime_ms, self.VBMF_layer_runtime['conv_'+key], self.remain_budget))
                        print('Travis Layer_budget + Remain_budget: {}'.format(self.layer_budget['conv_'+key]+self.remain_budget))
                        self.remain_budget = (self.layer_budget['conv_'+key] + self.remain_budget) - (tmp_runtime_ms - self.VBMF_layer_runtime['conv_'+key])
                        assert self.remain_budget >= 0
                        print('Updated Remain Budget: {}'.format(self.remain_budget))
                        break
                    else:
                        print('Update the objective function ...')
                        self.conv_target_rate += 0.05
                        print('Travis Constrain: {}, Search_runtime: {}, Layer_budget: {}, Search_layer_runtime: {}, VBMF_layer_runtime: {}, Remain_budget: {}'.format(self.constrain, \
                                                                tmp_runtime_ms-self.VBMF_layer_runtime['conv_'+key], \
                                                                self.layer_budget['conv_'+key], tmp_runtime_ms, self.VBMF_layer_runtime['conv_'+key], self.remain_budget))
                        print('Travis Layer_budget + Remain_budget: {}'.format(self.layer_budget['conv_'+key]+self.remain_budget))
                    
                print('{}Fine Tune{}'.format('\033[31m', '\033[0m'))
                print('{}-----------------------------------------{}'.format('\033[94m', '\033[0m'))
                ranks[0], ranks[1] = ranks[1], ranks[0]
                decompose = tucker_decomposition_conv_layer_without_rank(self.model.features._modules[key],ranks)
                self.model.features._modules[key] = decompose
                fine_tune(self.model, 10)
                fine_tune_test(self.model, self.testloader, True)
    
    def fc_decomposition(self):
        #remain_budget = 0.0
        N_classifier = len(self.model.classifier._modules.keys())
        for i, key in enumerate(self.model.classifier._modules.keys()):
            if i >= N_classifier - 2:
                break
            if isinstance(self.model.classifier._modules[key], torch.nn.modules.linear.Linear):
                fc_layer_to_decompose = self.model.classifier._modules[key]
                #print('Travis fc_layer_to_decompose rank1:{}, rank2:{}'.format(fc_layer_to_decompose.in_features, fc_layer_to_decompose.out_features))
                print('\n{}Layer Info{}'.format('\033[31m', '\033[0m'))
                print('{}-----------------------------------------{}'.format('\033[94m', '\033[0m'))
                print('Rank1:{}, Rank2:{}'.format(fc_layer_to_decompose.in_features, fc_layer_to_decompose.out_features))
                print(self.estimate(fc_layer_to_decompose, key))
                self.decomposed_layer_info['key'] = key

                self.fc_target_rate = 0
                iteration_count = 0
                while(True):
                    print('{}Iteration{}'.format('\033[31m', '\033[0m'))
                    print('{}-----------------------------------------{}'.format('\033[94m', '\033[0m'))
                    print('Iteration {}{}{} Target_Rate: {}{}{}'.format('\033[32m', iteration_count, '\033[0m', '\033[32m', self.fc_target_rate, '\033[0m'))
                    iteration_count += 1
                    rank = self.fc_bayesian(fc_layer_to_decompose.in_features*fc_layer_to_decompose.out_features//(fc_layer_to_decompose.in_features+fc_layer_to_decompose.out_features))
                    tmp_ms_1 = self.estimate_with_config([fc_layer_to_decompose.in_features, rank[0]])
                    tmp_ms_2 = self.estimate_with_config([rank[0], fc_layer_to_decompose.out_features])
                    tmp_runtime_ms = tmp_ms_1 + tmp_ms_2
                    assert (tmp_runtime_ms - self.VBMF_layer_runtime['fc_'+key]) > 0
                    if(self.constrain == 0 or (tmp_runtime_ms - self.VBMF_layer_runtime['fc_'+key]) <= (self.layer_budget['fc_'+ key] + self.remain_budget)):
                        print('Travis Constrain: {}, Search_runtime: {}, Layer_budget: {}, Search_layer_runtime: {}, VBMF_layer_runtime: {}, Remain_budget: {}'.format(self.constrain, \
                                                                tmp_runtime_ms-self.VBMF_layer_runtime['fc_'+key], \
                                                                self.layer_budget['fc_'+key], tmp_runtime_ms, self.VBMF_layer_runtime['fc_'+key], self.remain_budget))
                        print('Travis Layer_budget + Remain_budget: {}'.format(self.layer_budget['fc_'+key]+self.remain_budget))
                        self.remain_budget = (self.layer_budget['fc_'+key] + self.remain_budget) - (tmp_runtime_ms - self.VBMF_layer_runtime['fc_'+key])
                        assert self.remain_budget >= 0
                        print('Updated Remain budget: {}'.format(self.remain_budget))
                        break
                    else:
                        print('Update the objective function ...')
                        self.fc_target_rate += 0.1
                        print('Travis Constrain: {}, Search_runtime: {}, Layer_budget: {}, Search_layer_runtime: {}, VBMF_layer_runtime: {}, Remain_budget: {}'.format(self.constrain, \
                                                                tmp_runtime_ms-self.VBMF_layer_runtime['fc_'+key], \
                                                                self.layer_budget['fc_'+key], tmp_runtime_ms, self.VBMF_layer_runtime['fc_'+key], self.remain_budget))
                        print('Travis Layer_budget + Remain_budget: {}'.format(self.layer_budget['fc_'+key]+self.remain_budget))
                decompose = tucker_decomposition_fc_layer_without_rank(self.model.classifier._modules[key],rank)
                self.model.classifier._modules[key] = decompose
                fine_tune(self.model, 10)
                fine_tune_test(self.model, self.testloader, True)
                #print('Travis Model: {}'.format(self.model))

    def conv_bayesian(self, ranks):
        print('{}Travis{} : {} Rank: {} {}'.format('\33[33m', '\33[0m', self.decomposed_layer_info['key'], self.VBMF_layer_rank['conv_'+self.decomposed_layer_info['key']][0], \
                                                                                                        self.VBMF_layer_rank['conv_'+self.decomposed_layer_info['key']][1]))
        bo = BayesianOptimization(self.conv_target, {'in_channel': (self.VBMF_layer_rank['conv_'+self.decomposed_layer_info['key']][0]+1, ranks[0]), \
                                                    'out_channel': (self.VBMF_layer_rank['conv_'+self.decomposed_layer_info['key']][1]+1, ranks[1])})
        gp_params = {'kernel': None,
                    'alpha': 1e-5,
                    'kappa': 2}
        bo.maximize(init_points=2, n_iter=0, acq='ucb', **gp_params)
        bo.maximize(init_points=0, n_iter=100, acq='ucb', **gp_params)
        print(bo.res['max'])
        return [int(bo.res['max']['max_params']['in_channel']), int(bo.res['max']['max_params']['out_channel'])]

    def fc_bayesian(self, rank):
        print(self.decomposed_layer_info['key'])
        print(self.VBMF_layer_rank['fc_'+self.decomposed_layer_info['key']][0], type(self.VBMF_layer_rank['fc_'+self.decomposed_layer_info['key']]))
        print('{}Travis{} : {} Rank: {}'.format('\33[33m', '\33[0m', self.decomposed_layer_info['key'], self.VBMF_layer_rank['fc_'+self.decomposed_layer_info['key']][0]))
        bo = BayesianOptimization(self.fc_target, {'rank': (self.VBMF_layer_rank['fc_'+self.decomposed_layer_info['key']][0]+1, rank)})
        #bo = BayesianOptimization(self.fc_target, {'rank': (2 , rank)})
        gp_params = {'kernel': None,
                    'alpha': 1e-5,
                    'kappa': 2}
        bo.maximize(init_points=2, n_iter=0, acq='ucb', **gp_params)
        bo.maximize(init_points=0, n_iter=25, acq='ucb', **gp_params)
        print(bo.res['max'])
        return [int(bo.res['max']['max_params']['rank'])]

    def conv_target(self, in_channel, out_channel):

        assert int(self.decomposed_layer_info['key']) >= 0
        assert int(self.decomposed_layer_info['image_size']) > 0
        assert int(self.decomposed_layer_info['kernel_size']) > 0
        assert int(self.decomposed_layer_info['stride']) >= 0
        assert int(self.decomposed_layer_info['padding']) >= 0
        #print('Travis key: {}, rank1: {}, rank2: {}'.format(self.decomposed_layer_info['key'], in_channel, out_channel))
        rank = [int(out_channel), int(in_channel)]
        runtime_ms = 0.0

        decomp_model = copy.deepcopy(self.model)
        decomp_model.cuda()

        # Tucker decomposed
        decompose = tucker_decomposition_conv_layer_without_rank(decomp_model.features._modules[self.decomposed_layer_info['key']],rank)
        assert isinstance(decompose, torch.nn.modules.container.Sequential)
        for i in decompose:
            assert isinstance(i, torch.nn.modules.conv.Conv2d)
            tmp_ms = self.estimate_with_config([self.decomposed_layer_info['image_size'], i.in_channels, i.out_channels, i.kernel_size[0], i.stride[0], i.padding[0]])
            runtime_ms += tmp_ms
        decomp_model.features._modules[self.decomposed_layer_info['key']] = decompose
        decomp_model.cuda()
        decomp_model.eval()
        #print('Travis decomp_model: {}'.format(decomp_model))

        test_loss = 0
        correct = 0
        total = 0
        beta = 1.5

        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs = decomp_model(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += float(predicted.eq(targets.data).cpu().sum())
        decomp_model.train()
        acc = 100.*correct/total
        #print('Travis acc: {}, Travis runtime: {}, Value: {}'.format(acc, runtime_ms, -1*(100-acc)*(runtime_ms**0.1)))
        assert (runtime_ms**self.conv_target_rate) > 0
        print('{}Travis{} runtime: {}'.format('\033[36m', '\033[0m', runtime_ms))
        assert runtime_ms > 0
        return -1 * ((100-acc)**2) * (runtime_ms**self.conv_target_rate)

    def fc_target(self, rank):

        rank = [int(rank)]
        runtime_ms = 0.0

        decomp_model = copy.deepcopy(self.model)
        decomp_model.cuda()

        # Tucker decomposed
        decompose = tucker_decomposition_fc_layer_without_rank(decomp_model.classifier._modules[self.decomposed_layer_info['key']],rank)
        assert isinstance(decompose, torch.nn.modules.container.Sequential)
        for i in decompose:
            assert isinstance(i, torch.nn.modules.linear.Linear)
            tmp_ms = self.estimate_with_config([i.in_features, i.out_features])
            runtime_ms += tmp_ms
        decomp_model.classifier._modules[self.decomposed_layer_info['key']] = decompose
        decomp_model.cuda()
        decomp_model.eval()
        #print('Travis decomp_model: {}'.format(decomp_model))

        test_loss = 0
        correct = 0
        total = 0
        beta = 1.5

        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
            outputs = decomp_model(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += float(predicted.eq(targets.data).cpu().sum())
        decomp_model.train()
        acc = 100.*correct/total
        #print('Travis acc: {}, Travis runtime: {}, Value: {}'.format(acc, runtime_ms, -1*(100-acc)*(runtime_ms**0.1)))
        assert (runtime_ms**self.fc_target_rate) > 0
        assert runtime_ms > 0
        return -1 * (100-acc) * (runtime_ms**self.fc_target_rate)

    def estimate(self, layer, key):
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            runtime = self.perf_model.conv_predict(self.model_image_size[key][0], layer.in_channels, layer.out_channels ,\
                    layer.kernel_size[0], layer.stride[0], layer.padding[0])
        elif isinstance(layer, torch.nn.modules.linear.Linear):
            runtime = self.perf_model.fc_predict(layer.in_features, layer.out_features)
        return runtime

    def estimate_with_config(self, layer_config):
        if(len(layer_config) == 6):
            runtime = self.perf_model.conv_predict(layer_config[0], layer_config[1], layer_config[2] ,\
                    layer_config[3], layer_config[4], layer_config[5])
        elif(len(layer_config) == 2):
            runtime = self.perf_model.fc_predict(layer_config[0], layer_config[1])
        return runtime

    def get_layer_budget(self):

        N_classifier = len(self.model.classifier._modules.keys())
        decomp_model = copy.deepcopy(self.model)
        decomp_model.cuda()
        tmp_VBMF_layer_runtime = {}
        tmp_VBMF_model_runtime = 0.0
        tmp_VBMF_layer_rank = {}

        VBMF_model = torch.load('checkpoint/VBMF_alexnet_model')
        #print('Travis Decomp_model_with_VBMF: {}'.format(VBMF_model))

        for i, k in enumerate(VBMF_model.features._modules.keys()):
            #print('{}Travis{} : {}'.format('\033[32m', '\033[0m', VBMF_model.features._modules[k]))
            if isinstance(VBMF_model.features._modules[k], torch.nn.modules.container.Sequential):
                conv_container = VBMF_model.features._modules[k]
                for idx, conv_layer in enumerate(conv_container):
                    if(idx == 1):
                        tmp_VBMF_layer_rank['conv_'+k] = [conv_layer.in_channels, conv_layer.out_channels]
            if isinstance(VBMF_model.features._modules[k], torch.nn.modules.conv.Conv2d):
                conv_layer = VBMF_model.features._modules[k]
                tmp_VBMF_layer_rank['conv_'+k] = [conv_layer.in_channels, conv_layer.out_channels]
        for i, k in enumerate(VBMF_model.classifier._modules.keys()):
            #print('{}Travis{} : {}'.format('\033[32m', '\033[0m', VBMF_model.classifier._modules[k]))
            if isinstance(VBMF_model.classifier._modules[k], torch.nn.modules.container.Sequential):
                fc_container = VBMF_model.classifier._modules[k]
                for idx, fc_layer in enumerate(fc_container):
                    if(idx == 0):
                        tmp_VBMF_layer_rank['fc_'+k] = [fc_layer.out_features]
            if isinstance(VBMF_model.classifier._modules[k], torch.nn.modules.linear.Linear):
                fc_layer = VBMF_model.classifier._modules[k]
                tmp_VBMF_layer_rank['fc_'+k] = [fc_layer.in_features, fc_layer.out_features]

        tmp_VBMF_model_runtime, tmp_VBMF_layer_runtime = self.get_model_predict_runtime(VBMF_model)

        #print('Travis tmp_VBMF_model_runtime: {}, tmp_VBMF_layer_runtime: {}'.format(tmp_VBMF_model_runtime, tmp_VBMF_layer_runtime))
        print('{}Tmp_VBMF_model_runtime{}: {}'.format('\033[36m', '\033[0m', tmp_VBMF_model_runtime))
        print('{}Tmp_VBMF_layer_runtime{}: {}'.format('\033[36m', '\033[0m', tmp_VBMF_layer_runtime))
        print('{}Tmp_VBMF_layer_rank{}: {}'.format('\033[36m', '\033[0m', tmp_VBMF_layer_rank))

        print('{}VBMF model save ...{}'.format('\033[36m', '\033[0m'))

        budget = self.constrain - tmp_VBMF_model_runtime

        #print('Travis Runtime: {}, Budget: {}, Layer_mac: {}'.format(runtime_ms, self.user_budget, self.layer_mac))
        #print('Travis Layer MAC: {}'.format(self.layer_mac))

        tmp_layer_budget = {}
        for key, value in self.layer_importance.items():
            tmp_layer_budget[key] = float(value * budget)
        #print('Travis tmp_layer_budget: {}'.format(tmp_layer_budget))

        return tmp_layer_budget, tmp_VBMF_layer_runtime, tmp_VBMF_model_runtime, tmp_VBMF_layer_rank

    def get_model_mac_weight(self, model):

        total_mac = 0
        total_weight = 0

        for i, key in enumerate(model.features._modules.keys()):
            if isinstance(model.features._modules[key], torch.nn.modules.container.Sequential):
                sequential = model.features._modules[key]
                for conv_layer in sequential:
                    assert isinstance(conv_layer, torch.nn.modules.conv.Conv2d)
                    total_mac += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*(self.model_image_size[key][1]**2)*conv_layer.out_channels
                    total_weight += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*conv_layer.out_channels
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                total_mac += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*(self.model_image_size[key][1]**2)*conv_layer.out_channels
                total_weight += (conv_layer.kernel_size[0]**2)*conv_layer.in_channels*conv_layer.out_channels
            #print('Travis key: {}, total_mac: {}'.format(key, total_mac))
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
            #print('Travis key: {}, total_mac: {}'.format(key, total_mac))

        return total_mac, total_weight

    def get_model_predict_runtime(self, model):

        #print('Travis get model predict runtime')

        total_runtime = 0.0
        layer_runtime = {}
        N_classifier = len(model.classifier._modules.keys())

        for i, key in enumerate(model.features._modules.keys()):
            if isinstance(model.features._modules[key], torch.nn.modules.container.Sequential):
                sequential_layer = model.features._modules[key]
                tmp_runtime_ms = 0.0
                for container in sequential_layer:
                    assert isinstance(container, torch.nn.modules.conv.Conv2d)
                    tmp_ms = self.estimate_with_config([self.model_image_size[key][0], container.in_channels, container.out_channels, container.kernel_size[0], container.stride[0], container.padding[0]])
                    tmp_runtime_ms += tmp_ms
                layer_runtime['conv_'+key] = tmp_runtime_ms
                total_runtime += tmp_runtime_ms
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                tmp_ms = self.estimate_with_config([self.model_image_size[key][0], conv_layer.in_channels, conv_layer.out_channels, \
                        conv_layer.kernel_size[0], conv_layer.stride[0], conv_layer.padding[0]])
                layer_runtime['conv_'+key] = tmp_ms
                total_runtime += tmp_ms

        for i, key in enumerate(model.classifier._modules.keys()):
            if isinstance(model.classifier._modules[key], torch.nn.modules.container.Sequential):
                sequential_layer = model.classifier._modules[key]
                tmp_runtime_ms = 0.0
                for container in sequential_layer:
                    assert isinstance(container, torch.nn.modules.linear.Linear)
                    tmp_ms = self.estimate_with_config([container.in_features, container.out_features])
                    tmp_runtime_ms += tmp_ms
                layer_runtime['fc_'+key] = tmp_runtime_ms
                total_runtime += tmp_runtime_ms
            if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
                fc_layer = model.classifier._modules[key]
                tmp_ms = self.estimate_with_config([fc_layer.in_features, fc_layer.out_features])
                layer_runtime['fc_'+key] = tmp_ms
                total_runtime += tmp_ms

        return total_runtime, layer_runtime

    def get_layer_importance(self):

        layer_importance = {}
        layer_runtime = {}
        total_runtime = 0.0
        N_classifier = len(self.model.classifier._modules.keys())

        for i, key in enumerate(self.model.features._modules.keys()):
            if i == 0:
                continue
            if isinstance(self.model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = self.model.features._modules[key]
                tmp_ms = self.estimate_with_config([self.model_image_size[key][0], conv_layer.in_channels, conv_layer.out_channels, \
                        conv_layer.kernel_size[0], conv_layer.stride[0], conv_layer.padding[0]])
                layer_runtime['conv_'+key] = tmp_ms
                total_runtime += tmp_ms

        for i, key in enumerate(self.model.classifier._modules.keys()):
            if i >= N_classifier - 2:
                break
            if isinstance(self.model.classifier._modules[key], torch.nn.modules.linear.Linear):
                fc_layer = self.model.classifier._modules[key]
                tmp_ms = self.estimate_with_config([fc_layer.in_features, fc_layer.out_features])
                layer_runtime['fc_'+key] = tmp_ms
                total_runtime += tmp_ms

        for key, value in layer_runtime.items():
            layer_importance[key] = float(value/total_runtime)

        #print('Travis get_layer_importance layer_importance: {}'.format(layer_importance))

        return layer_importance

def test(**kwargs):
    opt.parse(kwargs)
    decomposer = Decomposer(**kwargs)
    #decomposer.print_model()
    #decomposer.fine_tune()
    #decomposer.test()
    decomposer.decompose()
    #decomposer.decompose_human()

if __name__ == '__main__':
    fire.Fire()
