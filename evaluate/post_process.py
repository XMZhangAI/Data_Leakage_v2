import json
import os
import re
import pdb

# def truncate(completion):

#     def find_re(string, pattern, start_pos):
#         m = pattern.search(string, start_pos)
#         return m.start() if m else -1

#     terminals = [
#         re.compile(r, re.MULTILINE)
#         for r in
#         [
#             '^#',
#             re.escape('<|endoftext|>'),
#             "^'''",
#             '^"""',
#             '\n\n\n',
#             re.escape('[DONE]'),
#             re.escape('[BEGIN]')
#         ]
#     ]

#     prints = list(re.finditer('^print', completion, re.MULTILINE))
#     # print(prints)
#     if len(prints) > 1:
#         completion = completion[:prints[1].start()] # 只取第一个print

#     # defs = list(re.finditer('^def', completion, re.MULTILINE))
#     # print(defs)
#     # if len(defs) > 1:
#     #     completion = completion[:defs[1].start()] # 有多个def只取第一个def, 可能存在问题

#     start_pos = 0

#     terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
#     if len(terminals_pos) > 0:
#         return completion[:min(terminals_pos)]
#     else:
#         return completion

def build_test_method_for_apps(input_output, test_case_limit = 5):
    test = "def check(candidate):\n"
    for idx, (input, output) in enumerate(zip(input_output['inputs'], input_output['outputs'])):
        if idx >= test_case_limit:
            break
        try:
            test += "\tassert candidate(%r) == %r \n" % (input.strip(), output.strip())
        except:
            test += "\tassert candidate(%s) == %s \n" % (input, output)
    return test

def truncate(d):
    d = d.split('\n\n')
    s = d[0] + '\n\n'
    if len(d)>1:
        for i in d[1:]:
            if 'def' not in i and i and '__main__' not in i:
                s += i + '\n\n'
            else:
                break
    return s

# 计算最小缩进
def minimum_indent(lines):
    m_indent = 100
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if indent > 0 and indent < m_indent:
            m_indent = indent
    return m_indent

# 检查是否需要全部缩进
def check_overall_indent(lines):
    def check_indent(lines):
        for line in lines:
            if "def" not in line and "print" not in line and "__name__" not in line and line[0] != '#' and len(line) - len(line.lstrip()) == 0:
                return True
        return False
    m_indent = minimum_indent(lines) # 最小缩进 （ > 0）
    if len(lines) <= 1:
        return False
    elif len(lines[0]) - len(lines[0].lstrip()) == 0:
        if lines[0].strip()[-1] == ':':
            space_num = len(lines[1]) - len(lines[1].lstrip())
            if space_num == m_indent:
                return True
        elif check_indent(lines[1:]):
            return True
    return False


def post_process_code(prompt, code, func_name, m_indent):
    assert type(code) == str
    # truncate
    if f"def {func_name}(" in code:
        return code
    truncation = truncate(code).replace('\r', '\n')
    truncation = re.sub('\n+', '\n', truncation)
    lines = truncation.split('\n')

    # 去除空行和与func_name相同的行，只考虑一个子函数，如果有多个子函数（例如HumanEval/39），需要修改
    # lines = list(filter(lambda x: x.strip() != "" and func_name not in x, lines))
    lines = list(filter(lambda x: x.strip() != "", lines))
    # 将tab替换，保持tab和空格一致性
    lines = list(map(lambda x: x.replace('\t', m_indent), lines))

    if len(lines) == 0:
        pass
    else:
        if check_overall_indent(lines):
            # 需要全部缩进
            for i in range(len(lines)):
                lines[i] = m_indent + lines[i] 
        elif len(lines[0]) - len(lines[0].lstrip()) == 0:
            # 仅首行缩进
            lines[0] = m_indent + lines[0]
        else:
            pass
    return prompt.replace('\t', m_indent)+'\n'.join(lines)