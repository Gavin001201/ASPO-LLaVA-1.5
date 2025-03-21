import argparse
import json
import os

import openai
openai.api_base = "https://api.kwwai.top/v1"
openai.api_key = "sk-Lvo1tVEiJohx6N36G8YsJklr5CExP4Qbi1aCrdGId0oZEgng"
import time

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question', default='/home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/questions.jsonl')
    parser.add_argument('-c', '--context', default='/home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/context.jsonl')
    parser.add_argument('-a', '--answer-list', nargs='+', default=['/home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl',\
                                                                    '/home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/answers/bz_10-lr_2e-6-lora_r_1024-scaling_factor_1-new_data-perplexity-13b-2.jsonl'])
    parser.add_argument('-r', '--rule', default='/home/data/wyy/projects/SeVa/seva/llava/eval/table/rule.json')
    parser.add_argument('-o', '--output', default='/home/data/wyy/projects/SeVa/seva/playground/data/eval/llava-bench-in-the-wild/reviews/bz_10-lr_2e-6-lora_r_1024-scaling_factor_1-new_data-perplexity-13b-2.jsonl')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
