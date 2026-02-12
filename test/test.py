import json
import re

def extract_question_and_reference(input_file, output_file):
    """
    从原始JSONL文件中提取问题和reference，生成新的JSONL文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        idx = 1
        for line in infile:
            try:
                data = json.loads(line.strip())
                
                # 1. 提取问题（从input字段中提取<问题>之后的内容）
                input_text = data.get('input', '')
                
                # 查找<问题>标记，提取后面的内容
                question_match = re.search(r'<问题>：\s*(.*?)$', input_text, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                else:
                    # 如果没有找到<问题>标记，则使用整个input作为问题
                    question = input_text.strip()
                
                # 2. 合并reference为单个字符串
                references = data.get('reference', [])
                reference_text = ' '.join(references)  # 用空格合并
                
                # 3. 创建新的JSON对象
                new_data = {
                    'question': question.replace('\r\n', '').replace('\n\r', ''),
                    'reference': reference_text.replace('\r\n', '').replace('\n\r', ''),
                    'id': idx
                }
                idx += 1
                
                # 4. 写入新的JSONL文件
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}, 行内容: {line[:100]}")
            except Exception as e:
                print(f"处理行时发生错误: {e}")

def main():
    # 输入和输出文件路径
    input_file = '/home/RAG_agent/test/DISC-Law-SFT-Triplet-QA-released.jsonl'  # 请替换为你的输入文件名
    output_file = '/home/RAG_agent/test/RAG_target.jsonl'  # 请替换为你想要的输出文件名
    
    print(f"开始处理文件: {input_file}")
    extract_question_and_reference(input_file, output_file)
    print(f"处理完成！结果已保存到: {output_file}")

if __name__ == "__main__":
    main()