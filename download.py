import json, os, sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['net_id_list', 'build_model', 'download_tflite']

""" Note: all the memory and latency profiling is done with TinyEngine """
NET_INFO = {
    ##### imagenet models ######
    # mcunet models
    'mcunet-in0': {
        'net_name': 'mcunet-10fps_imagenet',
        'description': 'MCUNet model that runs 10fps on STM32F746 (ImageNet)'
    },
    'mcunet-in1': {
        'net_name': 'mcunet-5fps_imagenet',
        'description': 'MCUNet model that runs 5fps on STM32F746 (ImageNet)'
    },
    'mcunet-in2': {
        'net_name': 'mcunet-256kb-1mb_imagenet',
        'description': 'MCUNet model that fits 256KB SRAM and 1MB Flash (ImageNet)',
    },
    'mcunet-in3': {
        'net_name': 'mcunet-320kb-1mb_imagenet',
        'description': 'MCUNet model that fits 320KB SRAM and 1MB Flash (ImageNet)',
    },
    'mcunet-in4': {
        'net_name': 'mcunet-512kb-2mb_imagenet',
        'description': 'MCUNet model that fits 512KB SRAM and 2MB Flash (ImageNet)',
    },
    # baseline models
    'mbv2-w0.35': {
        'net_name': 'mbv2-w0.35-r144_imagenet',
        'description': 'scaled MobileNetV2 that fits 320KB SRAM and 1MB Flash (ImageNet)',
    },
    'proxyless-w0.3': {
        'net_name': 'proxyless-w0.3-r176_imagenet',
        'description': 'scaled ProxylessNet that fits 320KB SRAM and 1MB Flash (ImageNet)'
    },

    ##### vww models ######
    'mcunet-vww0': {
        'net_name': 'mcunet-10fps_vww',
        'description': 'MCUNet model that runs 10fps on STM32F746 (VWW)'
    },
    'mcunet-vww1': {
        'net_name': 'mcunet-5fps_vww',
        'description': 'MCUNet model that runs 5fps on STM32F746 (VWW)'
    },
    'mcunet-vww2': {
        'net_name': 'mcunet-320kb-1mb_vww',
        'description': 'MCUNet model that fits 320KB SRAM and 1MB Flash (VWW)'
    },

    ##### detection demo model ######
    # NOTE: we have tf-lite only for this model
    'person-det': {
        'net_name': 'person-det',
        'description': 'person detection model used in our demo'
    },
}

net_id_list = list(NET_INFO.keys())

url_base = "https://hanlab18.mit.edu/projects/tinyml/mcunet/release/"

def download_url(url, model_dir='~/.torch/mcunet', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
        return None

def download_tflite(net_id):
    assert net_id in NET_INFO, 'Invalid net_id! Select one from {})'.format(list(NET_INFO.keys()))
    net_info = NET_INFO[net_id]
    tflite_url = url_base + net_info['net_name'] + ".tflite"
    return download_url(tflite_url)  # the file path of the downloaded tflite model

if __name__ == "__main__":
    download_tflite("mcunet-vww1")