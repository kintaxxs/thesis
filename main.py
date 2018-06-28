import fire
import torch
from config import opt
import torchvision
from utils.deploy import export_onnx_model, deploy_by_rpc
import torchvision.transforms as transforms
from utils.fine_tune import fine_tune, fine_tune_test
from utils.estimator import Estimator
import os

def print_model(**kwargs):
    opt.parse(kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    if opt.load_model_path:
            model = torch.load(opt.load_model_path)
            print(model)
    else:
        import sys
        print('set the load_model_path')
        sys.exit(1)

    if (type(model) is dict):
        model = model['net']

    model = model.cuda()
    model, acc = fine_tune(model, True)
    best_acc = fine_tune_test(model, testloader, True)
    print(best_acc)

    #Get the runtime before prunning
    tmp_save_model_name = export_onnx_model(model)
    origin_runtime = deploy_by_rpc(tmp_save_model_name)
    print('Real Runtime: {}ms'.format(origin_runtime*1000))
    os.remove(tmp_save_model_name)

    #total_mac, total_weight = get_model_mac_weight(model)
    #print('Mac: {}, Weight: {}'.format(total_mac, total_weight))

    # Calculate Image_size for each layer
    model_image_size = {'0': [32, 16], '2': [16, 8], '3': [8, 8], '5': [8, 4], '6': [4, 4], '8': [4, 4], '10': [4, 4], '12': [4, 2]}
    '''
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
    '''
    print('{}Image_Size{}: {}'.format('\033[36m', '\033[0m', model_image_size))

    #Get the Predicted Runtime
    predicted_model_runtime, _ = get_model_runtime(model, model_image_size)
    print('Predicted Runtime: {}ms'.format(predicted_model_runtime))

def get_model_runtime(model, model_image_size):

    total_runtime = 0.0
    layer_runtime = {}
    N_classifier = len(model.classifier._modules.keys())

    for i, key in enumerate(model.features._modules.keys()):
        if isinstance(model.features._modules[key], torch.nn.modules.container.Sequential):
            sequential_layer = model.features._modules[key]
            tmp_runtime_ms = 0.0
            for container in sequential_layer:
                assert isinstance(container, torch.nn.modules.conv.Conv2d)
                tmp_ms = estimate_with_config([model_image_size[key][0], container.in_channels, container.out_channels, \
                                                            container.kernel_size[0], container.stride[0], container.padding[0]])
                tmp_runtime_ms += tmp_ms
            layer_runtime['conv_'+key] = tmp_runtime_ms
            total_runtime += tmp_runtime_ms
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = model.features._modules[key]
            tmp_ms = estimate_with_config([model_image_size[key][0], conv_layer.in_channels, conv_layer.out_channels, \
                                    conv_layer.kernel_size[0], conv_layer.stride[0], conv_layer.padding[0]])
            layer_runtime['conv_'+key] = tmp_ms
            total_runtime += tmp_ms

    for i, key in enumerate(model.classifier._modules.keys()):
        if isinstance(model.classifier._modules[key], torch.nn.modules.container.Sequential):
            sequential_layer = model.classifier._modules[key]
            tmp_runtime_ms = 0.0
            for container in sequential_layer:
                assert isinstance(container, torch.nn.modules.linear.Linear)
                tmp_ms = estimate_with_config([container.in_features, container.out_features])
                tmp_runtime_ms += tmp_ms
            layer_runtime['fc_'+key] = tmp_runtime_ms
            total_runtime += tmp_runtime_ms
        if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
            fc_layer = model.classifier._modules[key]
            tmp_ms = estimate_with_config([fc_layer.in_features, fc_layer.out_features])
            layer_runtime['fc_'+key] = tmp_ms
            total_runtime += tmp_ms

    return total_runtime, layer_runtime

def estimate_with_config(layer_config):
	# Load Perf Model            
	perf_model = Estimator()

	if(len(layer_config) == 6):
		runtime = perf_model.conv_predict(layer_config[0], layer_config[1], layer_config[2] ,\
				layer_config[3], layer_config[4], layer_config[5])
	elif(len(layer_config) == 2):
		runtime = perf_model.fc_predict(layer_config[0], layer_config[1])
	return runtime

if __name__ == '__main__':
    fire.Fire()
