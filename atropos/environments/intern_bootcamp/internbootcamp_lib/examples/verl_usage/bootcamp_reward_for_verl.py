# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):

    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math_data
        res = math_data.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source.startswith("bootcamp/"):    
        try:
            import importlib
            import json
            bootcamp_name = data_source.split('/')[1]
            class_name = bootcamp_name[0].upper() + bootcamp_name[1:] + "bootcamp"
            module = importlib.import_module(f"internbootcamp")
            ground_truth = json.loads(ground_truth)
            return getattr(module, class_name).verify_score(solution_str, ground_truth, format_score=0)
        except Exception as e:
            print("Something woring with bootcamp reward because of ",e)
            return 0
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


    # print('solution string (generated raw output)', solution_str.replace('\n', ''))