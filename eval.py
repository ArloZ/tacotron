import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer

# sentences = [
#   # From July 8, 2017 New York Times:
#   'Scientists at the CERN laboratory say they have discovered a new particle.',
#   'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#   'President Trump met with other leaders at the Group of 20 conference.',
#   'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
#   # From Google's Tacotron example page:
#   'Generative adversarial network or variational auto-encoder.',
#   'The buses aren\'t the problem, they actually provide a solution.',
#   'Does the quick brown fox jump over the lazy dog?',
#   'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
# ]


sentences = [
    'ye1 feng1 bu4 jin4 lian2 xiang3 qi3 zi4',
    'ye1 feng1 bu4 jin4 lian2 xiang3 qi3 zi4 ji3 ying2 rao3 zai4 xin1 tou2 de5 wei4 le5 xin1 shi4 nan2 dao4 ta1 ye3 he2 wo3 yi1 yang4 wei2 ai4 fu4 chou2',
    'ke3 dui4 yu2 zhou1 yun2 lai2 shuo1 zhei4 xie1 wen4 ti2 ta1 dou1 jue2 de2 shi2 fen1 shen1 ao4 zhei4 me5 da4 de5 tie3 qiao4 zi5 chuan2 zen3 me5 neng2 fu2 zai4 shui3 mian4 shang4 xiang4 qi4 che1 yi1 yang4 xing2 shi4 ni2',
    'ta1 jing3 ti4 de5 xia4 le5 chuang2 gei3 liang3 ge5 sun1 zi5 ye4 hao3 bei4 zi5 you4 na2 guo4 yi1 ba3 da4 yi3 zi5 ba3 jie3 mei4 lia3 dang3 zhu4 gang1 zou3 dao4 ke4 ting1 jiu4 bei4 ren2 lan2 yao1 bao4 zhu4 le5',
    'wei1 xin4 zhi1 fu4 zhang1 xiao3 long2 han3 jian4 lou4 mian4 cheng1 wei1 xin4 bu4 hui4 cha2 kan4 yong4 hu4 liao2 tian1 ji4 lu4 yi4 si an4 feng4 zhi1 fu4 bao3 , ben3 wen2 lai2 zi4 teng2 xun4 ke1 ji4 .',
    'da4 hui4 zhi3 re4 nao5 tou2 liang3 tian1 yue4 hou4 yue4 song1 kua3 zui4 zhong1 chu1 ben3 lun4 wen2 ji2 jiu4 suan4 yuan2 man3 wan2 cheng2 ren4 wu5',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
