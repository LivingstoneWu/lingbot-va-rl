import json
import os
import shutil
from pathlib import Path

def process_episodes_file(input_file: str, output_file: str):
    """处理episodes.jsonl文件，添加action_config字段"""
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 读取并处理所有行
    episodes = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            ep['action_config'] = [{
                "start_frame": 0,
                "end_frame": ep.get('length', 0),
                "action_text": ep.get('tasks', [''])[0],
                "skill": ""
            }]
            episodes.append(ep)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')
    
    print(f"处理完成！已保存 {len(episodes)} 个episode到 {output_file}")

def main():
    """主函数"""
    base_dir = "/liujinxin/code/lhc/lingbot-va/datasets/robochallenge/turn_on_faucet_trim/meta"
    input_file = os.path.join(base_dir, "episodes.jsonl")
    backup_file = os.path.join(base_dir, "episodes_ori.jsonl")
    output_file = os.path.join(base_dir, "episodes_new.jsonl")
    
    # 备份原始文件
    if os.path.exists(input_file) and not os.path.exists(backup_file):
        shutil.copy2(input_file, backup_file)
        print(f"原始文件已备份到: {backup_file}")
    
    # 处理文件
    process_episodes_file(input_file, output_file)
    
    # 显示第一个episode示例
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            print("\n处理后的第一个episode示例:")
            print(json.dumps(json.loads(f.readline()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()